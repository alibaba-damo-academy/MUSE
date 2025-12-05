import os
import torch
import timm
import numpy as np
from torch import nn
from .nuclei_extractor import MultiLayerFeatureExtractorHead


class SupViTFeatureExtractor(nn.Module):

    def __init__(self, model_name):
        super().__init__()

        if model_name == 'sup_vit-s-16':
            model_weights = './weights/supervised/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'
            self.basic_model = timm.create_model('vit_small_patch16_224', pretrained=False)
            self.basic_model.load_pretrained(model_weights)
            self.out_dim = 384
        elif model_name == 'sup_vit-b-16':
            model_weights = './weights/supervised/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz'
            self.basic_model = timm.create_model('vit_base_patch16_224', pretrained=False)
            self.basic_model.load_pretrained(model_weights)
            self.out_dim = 768
        else:
            raise ValueError('Invalid model name')

        self.dense_feature_extractor = MultiLayerFeatureExtractorHead()

    def forward(self, input_img, input_coords):
        input_size = input_img.shape[-1]

        out = self.basic_model(input_img)
        h, w = int(input_img.shape[2] / 16), int(input_img.shape[3] / 16)
        out = out.reshape(out.shape[0], h, w, -1).permute(0, 3, 1, 2)
        
        return self.dense_feature_extractor([out], input_coords, input_size)

