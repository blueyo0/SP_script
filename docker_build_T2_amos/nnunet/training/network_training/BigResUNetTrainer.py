import torch
from nnunet.network_architecture.adw_ResUNet import AdwResUNet
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.utilities.nd_softmax import softmax_helper


class AdwResUNetTrainer(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):

        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)

        self.max_num_epochs = 1000   
        self.save_best_checkpoint = False
        self.arch_list = []
        
        
    def initialize_network(self):

        self.network = AdwResUNet(self.num_input_channels, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.arch_list,
                                    self.net_num_pool_op_kernel_sizes, 
                                    self.net_conv_kernel_sizes)
        print('current arch:', self.arch_list)
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper
        
    def setup_DA_params(self):
        super().setup_DA_params()
        self.data_aug_params["gamma_range"] = (0.3, 3.0)

        
class BigResUNetTrainer1(AdwResUNetTrainer):
    # 结构与 BigUNet1 类似
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):

        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.arch_list = [[64]*1, [128]*1, [256]*2, [512]*2, [1024]*3, [1024]*3, [1024]*3, [512]*2, [256]*2, [128]*1, [64]*1]

        
class BigResUNetTrainer2(AdwResUNetTrainer):
    # 结构与 BigUNet2 类似
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):

        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.arch_list = [[64]*1, [128]*2, [256]*3, [512]*4, [1024]*5, [1024]*6, [1024]*5, [512]*4, [256]*3, [128]*2, [64]*1]
        
class BigResUNetTrainer3(AdwResUNetTrainer):
    # 结构与 BigUNet3 类似
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):

        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.arch_list = [[64]*1, [128]*2, [256]*4, [512]*8, [1024]*16, [1024]*32, [1024]*16, [512]*8, [256]*4, [128]*2, [64]*1]
        
        
class BigResUNetTrainer4(AdwResUNetTrainer):
    # 结构与 BigUNet4 类似
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):

        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.arch_list = [[128]*1, [256]*1, [512]*2, [1024]*2, [2048]*3, [2048]*3, [2048]*3, [1024]*2, [512]*2, [256]*1, [128]*1]
        
