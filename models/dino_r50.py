import os
import torch
from torch import nn
import timm
from .nuclei_extractor import MultiLayerFeatureExtractorHead


class DINOR50DenseFeatureExtractor(nn.Module):

    def __init__(self, model_name):
        super().__init__()

        if model_name == 'dino_r50':
            model_weights = './weights/DINO/dino_resnet50_pretrain.pth'
        elif model_name == 'kang_bench_r50':
            model_weights = './weights/kang_bench/mocov2_rn50_ep200.torch'

        self.basic_model = timm.create_model('resnet50', pretrained=False, num_classes=0, global_pool='')

        self.basic_model.load_state_dict(torch.load(model_weights, map_location='cpu'))

        self.dense_feature_extractor = MultiLayerFeatureExtractorHead()
        self.out_dim = sum([256, 512, 1024, 2048])

    def forward(self, input_img, input_coords):

        input_size = input_img.shape[-1]

        feats = self.basic_model(input_img)

        return self.dense_feature_extractor(feats, input_coords, input_size)
