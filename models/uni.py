import os
import torch
import timm
from torch import nn
from .nuclei_extractor import MultiLayerFeatureExtractorHead


class UNIFeatureExtractor(nn.Module):

    def __init__(self):
        super().__init__()

        model_weights = './weights/UNI/pytorch_model.bin'

        self.basic_model = timm.create_model(
            "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
        )

        self.basic_model.load_state_dict(torch.load(model_weights, map_location="cpu"))

        self.dense_feature_extractor = MultiLayerFeatureExtractorHead()
        self.out_dim = 1024

    def forward(self, input_img, input_coords):
        input_size = input_img.shape[-1]

        out = self.basic_model(input_img)
        h, w = int(input_img.shape[2] / 16), int(input_img.shape[3] / 16)
        out = out.reshape(out.shape[0], h, w, -1).permute(0, 3, 1, 2)
        
        return self.dense_feature_extractor([out], input_coords, input_size)

