import os
import torch
from torch import nn
from transformers import AutoModel, Dinov2Config
from .nuclei_extractor import MultiLayerFeatureExtractorHead


class DINOV2ViTDenseFeatureExtractor(nn.Module):

    def __init__(self, model_name):
        super(DINOV2ViTDenseFeatureExtractor, self).__init__()

        if model_name == 'dinov2_vit-s':
            self.basic_model = AutoModel.from_pretrained('facebook/dinov2-small')
            self.out_dim = 384
        elif model_name == 'dinov2_vit-b':
            self.basic_model = AutoModel.from_pretrained('facebook/dinov2-base')
            self.out_dim = 768
        elif model_name == 'dinov2_vit-b-path':
            from dinov2.models.vision_transformer import vit_base
            self.basic_model = vit_base(patch_size=16, block_chunks=4, init_values=1.0e-05)
            model_weights = './weights/DINOV2/dinov2_vitb16_pathology.pth'
            fully_weights = torch.load(model_weights, map_location='cpu')['teacher']
            loaded_weights = dict()
            for k, v in fully_weights.items():
                if k.startswith('backbone.'):
                    new_k = k.replace('backbone.', '')
                    loaded_weights[new_k] = v
            self.basic_model.load_state_dict(loaded_weights)
            self.out_dim = 768
        elif model_name == 'dinov2_vit-s-path':
            from dinov2.models.vision_transformer import vit_small
            self.basic_model = vit_small(patch_size=16, block_chunks=4, init_values=1.0e-05)
            model_weights = './weights/DINOV2/dinov2_vits16_pathology.pth'
            fully_weights = torch.load(model_weights, map_location='cpu')['teacher']
            loaded_weights = dict()
            for k, v in fully_weights.items():
                if k.startswith('backbone.'):
                    new_k = k.replace('backbone.', '')
                    loaded_weights[new_k] = v
            self.basic_model.load_state_dict(loaded_weights)
            self.out_dim = 384
        else:
            raise ValueError('Invalid model name')

        self.dense_feature_extractor = MultiLayerFeatureExtractorHead()

        self.model_name = model_name

    def forward(self, input_img, input_coords):
        input_size = input_img.shape[-1]

        if self.model_name in ['dinov2_vit-s', 'dinov2_vit-b']:
            # official
            out = self.basic_model(input_img).last_hidden_state[:,1:,:]
            h, w = int(input_img.shape[2] / 14), int(input_img.shape[3] / 14)
            out = out.reshape(out.shape[0], h, w, -1).permute(0, 3, 1, 2)
        else:
            # path pretraining
            out = self.basic_model.get_intermediate_layers(input_img, n=1, reshape=True, return_class_token=False)[0]
        
        return self.dense_feature_extractor([out], input_coords, input_size)
