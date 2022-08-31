#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
from _warnings import warn
import torch
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.generic_SEUNet import Generic_SEUNet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.training.network_training.nnUNetTrainerV2_S5_D3_W64 import nnUNetTrainerV2_S5_D3_W64
from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn
import numpy as np
from tqdm import trange
from time import time, sleep
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.training.dataloading.dataset_loading import unpack_dataset
from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.dataloading.dataset_loading import DataLoader3D, DataLoader2D
from collections import OrderedDict
from sklearn.model_selection import KFold
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from torch.cuda.amp import autocast
import torch.backends.cudnn as cudnn
from nnunet.network_architecture.adw_ResUNet import AdwResUNet
from nnunet.training.learning_rate.poly_lr import poly_lr
from torch.optim.swa_utils import AveragedModel, SWALR
import io


def rand_bbox3d(size, lam):
    W = size[2]
    H = size[3]
    D = size[4]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)
    cut_d = np.int(D * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    cz = np.random.randint(D)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbz1 = np.clip(cz - cut_d // 2, 0, D)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    bbz2 = np.clip(cz + cut_d // 2, 0, D)

    return bbx1, bby1, bbz1, bbx2, bby2, bbz2


# def save_checkpoint_to_ceph(client, fname, model): # for large checkpoint
#     with io.BytesIO() as f:
#         torch.save(model, f)
#         f.seek(0)
#         client.put('s3://{}'.format(fname), f)


class nnUNetTrainerV2_amos(nnUNetTrainerV2_S5_D3_W64):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                            deterministic, fp16)
        self.bs = 4
        self.norm_op = nn.InstanceNorm3d
        self.norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        self.multi_modal_per_batch = False
        self.mix_mode = None # (Mixup, CutMix, None)
        # self.mix_mode = "CutMix" # (Mixup, CutMix, None)
        # self.mix_mode = "Mixup" # (Mixup, CutMix, None)
        self.mix_beta = 1.0
        self.mix_prob = 1.0
        self.mix_lam = 0.5
        self.network_type = "Default" # Default | SE | ResUNet4
        self.max_num_epochs = 1000   
        self.save_best_checkpoint = False
        self.arch_list = []
        self.ema_network = None
        self.ema_scheduler =None
        self.use_ema = False
        self.swa_start_epoch = 500
        self.use_swa_lr = False


    def initialize(self, training=True, force_load_plans=False):
        """
        - replaced get_default_augmentation with get_moreDA_augmentation
        - enforce to only run this code once
        - loss function wrapper for deep supervision

        :param training:
        :param force_load_plans:
        :return:
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans)
            self.batch_size = self.bs

            self.setup_DA_params()

            ################# Here we wrap the loss for deep supervision ############
            # we need to know the number of outputs of the network
            net_numpool = len(self.net_num_pool_op_kernel_sizes)

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
            weights[~mask] = 0
            weights = weights / weights.sum()
            self.ds_loss_weights = weights
            # now wrap the loss
            self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)
            ################# END ###################

            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                      "_stage%d" % self.stage)
            if training:
                self.dl_tr, self.dl_val = self.get_basic_generators()
                # [Note] split modality-specific dataloader
                if(self.multi_modal_per_batch): self.dl_tr_CT, self.dl_tr_MR = self.dl_tr
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")
                if(self.multi_modal_per_batch):
                    self.tr_gen_CT, self.val_gen = get_moreDA_augmentation(
                        self.dl_tr_CT, self.dl_val,
                        self.data_aug_params[
                            'patch_size_for_spatialtransform'],
                        self.data_aug_params,
                        deep_supervision_scales=self.deep_supervision_scales,
                        pin_memory=self.pin_memory,
                        use_nondetMultiThreadedAugmenter=False
                    )
                    self.tr_gen_MR, _ = get_moreDA_augmentation(
                        self.dl_tr_MR, self.dl_val,
                        self.data_aug_params[
                            'patch_size_for_spatialtransform'],
                        self.data_aug_params,
                        deep_supervision_scales=self.deep_supervision_scales,
                        pin_memory=self.pin_memory,
                        use_nondetMultiThreadedAugmenter=False
                    )
                    self.tr_gen = (self.tr_gen_CT, self.tr_gen_MR)
                else:
                    self.tr_gen, self.val_gen = get_moreDA_augmentation(
                        self.dl_tr, self.dl_val,
                        self.data_aug_params[
                            'patch_size_for_spatialtransform'],
                        self.data_aug_params,
                        deep_supervision_scales=self.deep_supervision_scales,
                        pin_memory=self.pin_memory,
                        use_nondetMultiThreadedAugmenter=False
                    )

                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                       also_print_to_console=False)
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

            assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                         momentum=0.99, nesterov=True)
        self.lr_scheduler = None
        if(self.use_swa_lr):
            self.ema_scheduler = SWALR(self.optimizer, swa_lr=0.05)

    def maybe_update_lr(self, epoch=None):
        """ add EMA scheduler strategy """
        if(self.ema_scheduler is not None and self.ema_network is not None and self.epoch > self.swa_start_epoch):
            self.ema_network.update_parameters(self.network)
            self.ema_scheduler.step(self.epoch+1)
        else:
            if epoch is None:
                ep = self.epoch + 1
            else:
                ep = epoch
            self.optimizer.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.initial_lr, 0.9)
        self.print_to_log_file("lr:", np.round(self.optimizer.param_groups[0]['lr'], decimals=6))

    def save_checkpoint(self, fname, save_optimizer=True):
        start_time = time()
        if(self.ema_network is not None):
            state_dict = self.ema_network.state_dict()
        else: 
            state_dict = self.network.state_dict()

        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()
        lr_sched_state_dct = None
        if self.lr_scheduler is not None and hasattr(self.lr_scheduler,
                                                     'state_dict'):  # not isinstance(self.lr_scheduler, lr_scheduler.ReduceLROnPlateau):
            lr_sched_state_dct = self.lr_scheduler.state_dict()
            # WTF is this!?
            # for key in lr_sched_state_dct.keys():
            #    lr_sched_state_dct[key] = lr_sched_state_dct[key]
        if save_optimizer:
            optimizer_state_dict = self.optimizer.state_dict()
        else:
            optimizer_state_dict = None

        self.print_to_log_file("saving checkpoint...")
        save_this = {
            'epoch': self.epoch + 1,
            'state_dict': state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            'lr_scheduler_state_dict': lr_sched_state_dct,
            'plot_stuff': (self.all_tr_losses, self.all_val_losses, self.all_val_losses_tr_mode,
                           self.all_val_eval_metrics),
            'best_stuff' : (self.best_epoch_based_on_MA_tr_loss, self.best_MA_tr_loss_for_patience, self.best_val_eval_criterion_MA)}
        if self.amp_grad_scaler is not None:
            save_this['amp_grad_scaler'] = self.amp_grad_scaler.state_dict()

        # import pdb; pdb.set_trace()
        # try:
        #     ceph_fname = 'nnUNet_trained_models' + fname.split('nnUNet_trained_models')[1]
        #     save_checkpoint_to_ceph(self.client, ceph_fname, save_this)
        #     self.print_to_log_file('save to ceph...')
        # except:
        #     print ('save ceph failed...')
            torch.save(save_this, fname)
        # self.print_to_log_file('save to local path...')
        self.print_to_log_file("done, saving took %.2f seconds" % (time() - start_time))

    def initialize_network(self):
        if(self.network_type!="ResUNet4"):
            self.base_num_features = 64  
            self.conv_per_stage = 3
            self.stage_num = 5
            self.max_num_features = 512

            original_stage_num = len(self.net_conv_kernel_sizes)
            if original_stage_num > self.stage_num:
                self.net_conv_kernel_sizes = self.net_conv_kernel_sizes[:self.stage_num]
                self.net_num_pool_op_kernel_sizes = self.net_num_pool_op_kernel_sizes[:self.stage_num-1]

            if self.threeD:
                conv_op = nn.Conv3d
                dropout_op = nn.Dropout3d
                norm_op = self.norm_op

            else:
                conv_op = nn.Conv2d
                dropout_op = nn.Dropout2d
                norm_op = nn.InstanceNorm2d

            norm_op_kwargs = self.norm_op_kwargs
            dropout_op_kwargs = {'p': 0, 'inplace': True}
            net_nonlin = nn.LeakyReLU
            net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}

        if(self.network_type=="SE"):
            self.network = Generic_SEUNet(self.num_input_channels, self.base_num_features, self.num_classes,
                                        len(self.net_num_pool_op_kernel_sizes),
                                        self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                                        net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                        self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True, self.max_num_features,
                                        encoder_se=True, encoder_residual=True, 
                                        decoder_se=True, decoder_residual=True)
            self.print_to_log_file('use ESR_DSR network')
        elif(self.network_type=="ResUNet4"):
            self.arch_list = [[128]*1, [256]*1, [512]*2, [1024]*2, [2048]*3, [2048]*3, [2048]*3, [1024]*2, [512]*2, [256]*1, [128]*1]
            self.network = AdwResUNet(self.num_input_channels, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.arch_list,
                                    self.net_num_pool_op_kernel_sizes, 
                                    self.net_conv_kernel_sizes)
            self.print_to_log_file('ResUNet4-current arch:', self.arch_list)
        else:
            self.network = Generic_UNet(self.num_input_channels, self.base_num_features, self.num_classes,
                                        len(self.net_num_pool_op_kernel_sizes),
                                        self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                                        net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                        self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True, self.max_num_features)
            self.print_to_log_file('use Default GenericUNet network')

        if(self.use_ema): 
            self.ema_network = AveragedModel(self.network)
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper


    def do_split(self):
        """
        In AMOS, filename is like ‘amos_0598.nii.gz’, id>499 (MR)
        """
        splits_file = join(self.dataset_directory, "splits_final.pkl")
        if not isfile(splits_file):
            print("balanced split of CT and MR does not exist")
            # raise RuntimeError("no split")
            self.print_to_log_file("Creating new 5-fold cross-validation split...")
            splits = []
            all_keys_sorted = np.sort(list(self.dataset.keys()))
            kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
            for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
                train_keys = np.array(all_keys_sorted)[train_idx]
                test_keys = np.array(all_keys_sorted)[test_idx]
                splits.append(OrderedDict())
                splits[-1]['train'] = train_keys
                splits[-1]['val'] = test_keys
            save_pickle(splits, splits_file)

        self.print_to_log_file("Using splits from existing split file:", splits_file)
        splits = load_pickle(splits_file)
        self.print_to_log_file("The split file contains %d splits." % len(splits))

        if self.fold == "all":
            tr_keys = val_keys = list(self.dataset.keys())
        else:
            tr_keys = splits[self.fold]['train']
            val_keys = splits[self.fold]['val']

        tr_keys.sort()
        val_keys.sort()

        # [Note] split CT and MR in dataset_tr
        self.dataset_tr = OrderedDict()
        self.dataset_tr_CT = OrderedDict()
        self.dataset_tr_MR = OrderedDict()
        for i in tr_keys:
            self.dataset_tr[i] = self.dataset[i]
            key_id = int(i.split('_')[-1])
            if(key_id<500): #CT
                self.dataset_tr_CT[i] = self.dataset[i]
            else:
                self.dataset_tr_MR[i] = self.dataset[i]
        
        self.dataset_val = OrderedDict()
        for i in val_keys:
            self.dataset_val[i] = self.dataset[i]

    def get_basic_generators(self):
        self.load_dataset()
        self.do_split()

        if self.multi_modal_per_batch:
            if self.threeD:
                dl_tr_CT = DataLoader3D(self.dataset_tr_CT, self.basic_generator_patch_size, self.patch_size, self.batch_size[0],
                                    False, oversample_foreground_percent=self.oversample_foreground_percent,
                                    pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
                dl_tr_MR = DataLoader3D(self.dataset_tr_MR, self.basic_generator_patch_size, self.patch_size, self.batch_size[1],
                                    False, oversample_foreground_percent=self.oversample_foreground_percent,
                                    pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
                dl_val = DataLoader3D(self.dataset_val, self.patch_size, self.patch_size, sum(self.batch_size), False,
                                    oversample_foreground_percent=self.oversample_foreground_percent,
                                    pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
            else:
                dl_tr_CT = DataLoader2D(self.dataset_tr_CT, self.basic_generator_patch_size, self.patch_size, self.batch_size[0],
                                    oversample_foreground_percent=self.oversample_foreground_percent,
                                    pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
                dl_tr_MR = DataLoader2D(self.dataset_tr_MR, self.basic_generator_patch_size, self.patch_size, self.batch_size[1],
                                    oversample_foreground_percent=self.oversample_foreground_percent,
                                    pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
                dl_val = DataLoader2D(self.dataset_val, self.patch_size, self.patch_size, sum(self.batch_size),
                                    oversample_foreground_percent=self.oversample_foreground_percent,
                                    pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
            dl_tr = (dl_tr_CT, dl_tr_MR)
        else:
            if self.threeD:
                dl_tr = DataLoader3D(self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                                     False, oversample_foreground_percent=self.oversample_foreground_percent,
                                     pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
                dl_val = DataLoader3D(self.dataset_val, self.patch_size, self.patch_size, self.batch_size, False,
                                    oversample_foreground_percent=self.oversample_foreground_percent,
                                    pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
            else:
                dl_tr = DataLoader2D(self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                                    oversample_foreground_percent=self.oversample_foreground_percent,
                                    pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
                dl_val = DataLoader2D(self.dataset_val, self.patch_size, self.patch_size, self.batch_size,
                                    oversample_foreground_percent=self.oversample_foreground_percent,
                                    pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
        return dl_tr, dl_val

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        if self.multi_modal_per_batch and do_backprop:
            # [Note] combine two data_dict of CT and MR
            try:
                data_gen_CT, data_gen_MR = data_generator
                data_dict_CT = next(data_gen_CT)
                data_dict_MR = next(data_gen_MR)
                if isinstance(data_dict_CT['data'], list):
                    data = data_dict_CT['data'] + data_dict_MR['data']
                else:
                    data = torch.cat([data_dict_CT['data'], data_dict_MR['data']], 0)
                if isinstance(data_dict_CT['target'], list):
                    target = [torch.cat([ct, mr], 0) for ct, mr in zip(data_dict_CT['target'], data_dict_MR['target'])]
                else:
                    target = torch.cat([data_dict_CT['target'], data_dict_MR['target']], 0)            
            except:
                data_dict = next(data_generator)
                data = data_dict['data']
                target = data_dict['target']                
        else:
            data_dict = next(data_generator)
            data = data_dict['data']
            target = data_dict['target']

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        # mixing augmentation
        if(self.mix_mode):
            assert self.mix_mode in ("Mixup", "CutMix", "MaskedCutMix")
            if(self.mix_mode=="CutMix"):
                r = np.random.rand(1)
                if self.mix_beta > 0 and r < self.mix_prob:
                    lam = np.random.beta(self.mix_beta, self.mix_beta)
                    rand_index = torch.randperm(data.size()[0])
                    bbx1, bby1, bbz1, bbx2, bby2, bbz2 = rand_bbox3d(data.size(), lam)
                    data[:, :, bbx1:bbx2, bby1:bby2, bbz1:bbz2] = data[rand_index, :, bbx1:bbx2, bby1:bby2, bbz1:bbz2]
                    # CutMix for each class
                    for i in range(len(target)):
                        target[i][:, :, bbx1:bbx2, bby1:bby2, bbz1:bbz2] = target[i][rand_index, :, bbx1:bbx2, bby1:bby2, bbz1:bbz2]                                       
            elif(self.mix_mode=="Mixup"):
                assert self.mix_lam>=0.0 and self.mix_lam<=1.0
                rand_index = torch.randperm(data.size()[0])
                data = self.mix_lam * data[rand_index] + (1-self.mix_lam)* data
                for i in range(len(target)):
                    target[i] = self.mix_lam * target[i][rand_index] + (1-self.mix_lam)* target[i]
            else:
                raise ValueError(f"Do not support the mixing mode {self.mix_mode}")

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                output = self.network(data)
                del data
                l = self.loss(output, target)

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            output = self.network(data)
            del data
            l = self.loss(output, target)

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target

        return l.detach().cpu().numpy()

    def run_training(self):
        if not torch.cuda.is_available():
            self.print_to_log_file("WARNING!!! You are attempting to run training on a CPU (torch.cuda.is_available() is False). This can be VERY slow!")

        if self.multi_modal_per_batch:
            _ = self.tr_gen[0].next()
            _ = self.tr_gen[1].next()
        else:
            _ = self.tr_gen.next()

        _ = self.val_gen.next()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._maybe_init_amp()

        maybe_mkdir_p(self.output_folder)        
        self.plot_network_architecture()

        if cudnn.benchmark and cudnn.deterministic:
            warn("torch.backends.cudnn.deterministic is True indicating a deterministic training is desired. "
                 "But torch.backends.cudnn.benchmark is True as well and this will prevent deterministic training! "
                 "If you want deterministic then set benchmark=False")

        if not self.was_initialized:
            self.initialize(True)

        while self.epoch < self.max_num_epochs:
            self.print_to_log_file("\nepoch: ", self.epoch)
            epoch_start_time = time()
            train_losses_epoch = []

            # train one epoch
            self.network.train()

            if self.use_progress_bar:
                with trange(self.num_batches_per_epoch) as tbar:
                    for b in tbar:
                        tbar.set_description("Epoch {}/{}".format(self.epoch+1, self.max_num_epochs))

                        l = self.run_iteration(self.tr_gen, True)

                        tbar.set_postfix(loss=l)
                        train_losses_epoch.append(l)
            else:
                for _ in range(self.num_batches_per_epoch):
                    l = self.run_iteration(self.tr_gen, True)
                    train_losses_epoch.append(l)

            self.all_tr_losses.append(np.mean(train_losses_epoch))
            self.print_to_log_file("train loss : %.4f" % self.all_tr_losses[-1])

            with torch.no_grad():
                # validation with train=False
                self.network.eval()
                val_losses = []
                for b in range(self.num_val_batches_per_epoch):
                    l = self.run_iteration(self.val_gen, False, True)
                    val_losses.append(l)
                self.all_val_losses.append(np.mean(val_losses))
                self.print_to_log_file("validation loss: %.4f" % self.all_val_losses[-1])

                if self.also_val_in_tr_mode:
                    self.network.train()
                    # validation with train=True
                    val_losses = []
                    for b in range(self.num_val_batches_per_epoch):
                        l = self.run_iteration(self.val_gen, False)
                        val_losses.append(l)
                    self.all_val_losses_tr_mode.append(np.mean(val_losses))
                    self.print_to_log_file("validation loss (train=True): %.4f" % self.all_val_losses_tr_mode[-1])

            self.update_train_loss_MA()  # needed for lr scheduler and stopping of training

            continue_training = self.on_epoch_end()

            epoch_end_time = time()

            if not continue_training:
                # allows for early stopping
                break

            self.epoch += 1
            self.print_to_log_file("This epoch took %f s\n" % (epoch_end_time - epoch_start_time))

        self.epoch -= 1  # if we don't do this we can get a problem with loading model_final_checkpoint.

        if self.save_final_checkpoint: self.save_checkpoint(join(self.output_folder, "model_final_checkpoint.model"))
        # now we can delete latest as it will be identical with final
        if isfile(join(self.output_folder, "model_latest.model")):
            os.remove(join(self.output_folder, "model_latest.model"))
        if isfile(join(self.output_folder, "model_latest.model.pkl")):
            os.remove(join(self.output_folder, "model_latest.model.pkl"))

    def load_checkpoint_ram(self, checkpoint, train=True):
        """
        used for if the checkpoint is already in ram
        :param checkpoint:
        :param train:
        :return: 
        """
        if not self.was_initialized:
            self.initialize(train)


        if(not self.use_ema):
            new_state_dict = OrderedDict()
            curr_state_dict_keys = list(self.network.state_dict().keys())
            # if state dict comes form nn.DataParallel but we use non-parallel model here then the state dict keys do not
            # match. Use heuristic to make it match
            for k, value in checkpoint['state_dict'].items():
                key = k
                if key not in curr_state_dict_keys and key.startswith('module.'):
                    key = key[7:]
                new_state_dict[key] = value
        else:
            print("loading ema_network...")
            self.ema_network.load_state_dict(checkpoint['state_dict'])
            print("done loading ema_network")
            new_state_dict = OrderedDict()
            curr_state_dict_keys = list(self.network.state_dict().keys())
            # if state dict comes form nn.DataParallel but we use non-parallel model here then the state dict keys do not
            # match. Use heuristic to make it match
            for k, value in checkpoint['state_dict'].items():
                if k == "n_averaged": continue # ignore n_averaged in ema
                key = k
                if key not in curr_state_dict_keys and key.startswith('module.'):
                    key = key[7:]
                new_state_dict[key] = value


        if self.fp16:
            self._maybe_init_amp()
            if train:
                if 'amp_grad_scaler' in checkpoint.keys():
                    self.amp_grad_scaler.load_state_dict(checkpoint['amp_grad_scaler'])

        self.network.load_state_dict(new_state_dict)
        self.epoch = checkpoint['epoch']
        if train:
            optimizer_state_dict = checkpoint['optimizer_state_dict']
            if optimizer_state_dict is not None:
                self.optimizer.load_state_dict(optimizer_state_dict)

            if self.lr_scheduler is not None and hasattr(self.lr_scheduler, 'load_state_dict') and checkpoint[
                'lr_scheduler_state_dict'] is not None:
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

            if issubclass(self.lr_scheduler.__class__, _LRScheduler):
                self.lr_scheduler.step(self.epoch)

        self.all_tr_losses, self.all_val_losses, self.all_val_losses_tr_mode, self.all_val_eval_metrics = checkpoint[
            'plot_stuff']

        # load best loss (if present)
        if 'best_stuff' in checkpoint.keys():
            self.best_epoch_based_on_MA_tr_loss, self.best_MA_tr_loss_for_patience, self.best_val_eval_criterion_MA = checkpoint[
                'best_stuff']

        # after the training is done, the epoch is incremented one more time in my old code. This results in
        # self.epoch = 1001 for old trained models when the epoch is actually 1000. This causes issues because
        # len(self.all_tr_losses) = 1000 and the plot function will fail. We can easily detect and correct that here
        if self.epoch != len(self.all_tr_losses):
            self.print_to_log_file("WARNING in loading checkpoint: self.epoch != len(self.all_tr_losses). This is "
                                   "due to an old bug and should only appear when you are loading old models. New "
                                   "models should have this fixed! self.epoch is now set to len(self.all_tr_losses)")
            self.epoch = len(self.all_tr_losses)
            self.all_tr_losses = self.all_tr_losses[:self.epoch]
            self.all_val_losses = self.all_val_losses[:self.epoch]
            self.all_val_losses_tr_mode = self.all_val_losses_tr_mode[:self.epoch]
            self.all_val_eval_metrics = self.all_val_eval_metrics[:self.epoch]

        self._maybe_init_amp()

