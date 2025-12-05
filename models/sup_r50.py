import os
import torch
from torch import nn
import timm
from .nuclei_extractor import MultiLayerFeatureExtractorHead


class SupR50DenseFeatureExtractor(nn.Module):

    def __init__(self):
        super().__init__()

        self.basic_model = timm.create_model('resnet50', pretrained=True, num_classes=0, global_pool='')

        self.dense_feature_extractor = MultiLayerFeatureExtractorHead()
        self.out_dim = sum([256, 512, 1024, 2048])

    def forward(self, input_img, input_coords):

        input_size = input_img.shape[-1]

        feats = self.basic_model(input_img)

        return self.dense_feature_extractor(feats, input_coords, input_size)
