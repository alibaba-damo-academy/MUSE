import os
import torch
import timm
from torch import nn
from .nuclei_extractor import MultiLayerFeatureExtractorHead


class GigaPathFeatureExtractor(nn.Module):

    def __init__(self):
        super(GigaPathFeatureExtractor, self).__init__()

        model_weights = './weights/prov-gigapath/pytorch_model.bin'
        
        timm_kwargs = {
            'model_name': 'vit_giant_patch14_dinov2',
            'img_size': 224, 
            'patch_size': 16, 
            'depth': 40,
            'num_heads': 24,
            'init_values': 1e-5, 
            'embed_dim': 1536,
            'mlp_ratio': 5.33334,
            'num_classes': 0, 
        }

        self.basic_model = timm.create_model(pretrained=False, **timm_kwargs)
        self.out_dim = 1536

        self.basic_model.load_state_dict(torch.load(model_weights, map_location="cpu"))
        self.dense_feature_extractor = MultiLayerFeatureExtractorHead()

    def forward(self, input_img, input_coords):
        input_size = input_img.shape[-1]

        out = self.basic_model(input_img)
        h, w = int(input_img.shape[2] / 16), int(input_img.shape[3] / 16)
        out = out.reshape(out.shape[0], h, w, -1).permute(0, 3, 1, 2)
        
        return self.dense_feature_extractor([out], input_coords, input_size)
