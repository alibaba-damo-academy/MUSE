import os
import torch
from torch import nn
import timm
from .nuclei_extractor import MultiLayerFeatureExtractorHead
from .conch.open_clip_custom import create_model_from_pretrained


class CONCHFeatureExtractor(nn.Module):

    def __init__(self):
        super().__init__()

        model, _ = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path="./weights/CONCH/pytorch_model.bin")

        # we only use the vision encoder
        self.basic_model = model.visual.trunk
        self.out_dim = 768

        self.dense_feature_extractor = MultiLayerFeatureExtractorHead()

    def forward(self, input_img, input_coords):
        input_size = input_img.shape[-1]

        out = self.basic_model(input_img)[:, 1:, :]
        h, w = int(input_img.shape[2] / self.basic_model.patch_embed.patch_size[0]), int(input_img.shape[3] / self.basic_model.patch_embed.patch_size[0])
        out = out.reshape(out.shape[0], h, w, -1).permute(0, 3, 1, 2)

        return self.dense_feature_extractor([out], input_coords, input_size)
