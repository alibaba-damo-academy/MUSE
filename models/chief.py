import os
import torch
from torch import nn
import timm
from timm.models.layers.helpers import to_2tuple
from .nuclei_extractor import MultiLayerFeatureExtractorHead


class ConvStem(nn.Module):

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, flatten=True):
        super().__init__()

        assert patch_size == 4
        assert embed_dim % 8 == 0

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        stem = []
        input_dim, output_dim = 3, embed_dim // 8
        for l in range(2):
            stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

def ctranspath():
    model = timm.create_model('swin_tiny_patch4_window7_224', embed_layer=ConvStem, pretrained=False, num_classes=0)
    model_weights = './weights/CHIEF/CHIEF_CTransPath.pth'
    model.load_state_dict(torch.load(model_weights, map_location='cpu')['model'])
    return model


class CHIEFPathFeatureExtractor(nn.Module):

    def __init__(self):
        super().__init__()

        model_weights = None

        self.basic_model = ctranspath()

        self.dense_feature_extractor = MultiLayerFeatureExtractorHead()
        # self.out_dim = sum([192, 384, 768, 768])
        self.out_dim = 768
        self.out_shape = [28, 14, 7, 7]
    
    def forward(self, input_img, input_coords):
        input_size = input_img.shape[-1]

        out = self.basic_model(input_img)  # [[784, 192], [196, 384], [49, 768]]

        # reshape out
        for i in range(len(out)):
            out[i] = out[i].reshape(out[i].shape[0], self.out_shape[i], self.out_shape[i], -1).permute(0, 3, 1, 2)
        
        # ============= IMPORTANT ============= 
        # as CTransPath is not train with multi-level, we notice only use the last layer is better
        return self.dense_feature_extractor([out[-1]], input_coords, input_size)
