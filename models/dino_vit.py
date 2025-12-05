import os
import torch
from torch import nn
from .dino_vision_transformer import vit_small, vit_base
from .nuclei_extractor import MultiLayerFeatureExtractorHead


class DINOViTDenseFeatureExtractor(nn.Module):

    def __init__(self, model_name):
        super(DINOViTDenseFeatureExtractor, self).__init__()

        if model_name == 'dino_vit-s-16':
            model_weights = './weights/DINO/dino_deitsmall16_pretrain.pth'
            self.basic_model = vit_small(patch_size=16)
            self.out_dim = 384
        elif model_name == 'dino_vit-s-8':
            model_weights = './weights/DINO/dino_deitsmall8_pretrain.pth'
            self.basic_model = vit_small(patch_size=8)
            self.out_dim = 384
        elif model_name == 'dino_vit-b-16':
            model_weights = './weights/DINO/dino_vitbase16_pretrain.pth'
            self.basic_model = vit_base(patch_size=16)
            self.out_dim = 768
        elif model_name == 'dino_vit-b-8':
            model_weights = './weights/DINO/dino_vitb8_pretrain.pth'
            self.basic_model = vit_base(patch_size=8)
            self.out_dim = 768
        elif model_name == 'dino_kang_bench_vit-s-16':
            model_weights = './weights/kang_bench/dino_vit_small_patch16_ep200.torch'
            self.basic_model = vit_small(patch_size=16)
            self.out_dim = 384
        else:
            raise ValueError('Invalid model name')
        
        self.basic_model.load_state_dict(torch.load(model_weights, map_location='cpu'))

        self.dense_feature_extractor = MultiLayerFeatureExtractorHead()

    def forward(self, input_img, input_coords):
        input_size = input_img.shape[-1]

        out = self.basic_model.get_intermediate_layers(input_img, n=1)[0]
        out = out[:, 1:, :]  # remove cls token
        h, w = int(input_img.shape[2] / self.basic_model.patch_embed.patch_size), int(input_img.shape[3] / self.basic_model.patch_embed.patch_size)
        out = out.reshape(out.shape[0], h, w, -1).permute(0, 3, 1, 2)
        
        return self.dense_feature_extractor([out], input_coords, input_size)
