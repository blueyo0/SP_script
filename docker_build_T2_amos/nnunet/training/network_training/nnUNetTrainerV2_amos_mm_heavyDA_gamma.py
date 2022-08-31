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
import torch
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.training.network_training.nnUNetTrainerV2_amos import nnUNetTrainerV2_amos
from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn
import numpy as np

class nnUNetTrainerV2_amos_mm_heavyDA_gamma(nnUNetTrainerV2_amos):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
            unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                            deterministic, fp16)
        self.bs = (5,1)
        self.multi_modal_per_batch = True
        self.mix_mode = "CutMix"
    
    def setup_DA_params(self):
        super().setup_DA_params()
        self.data_aug_params["gamma_range"] = (0.6, 1.6)

