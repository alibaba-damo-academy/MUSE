"""
This version define the model with: encoder - decoder - hyperfeats - basic linear
Specifically, we use the following:
- encoder: vit-based
- decoder: cnn-based
- hyperfeats: linear conv
- basic linear: linear: map to the target dims
"""

import math
import torch
import timm
import torch.nn as nn
import models.dino_vision_transformer as vits
import torch.nn.functional as F
from .nuclei_extractor import MultiLayerFeatureExtractorHead


class ClsTokenLayer(nn.Module):

    def __init__(self, merge_type='proj', dim=768):
        super().__init__()

        self.merge_type = merge_type
        if merge_type == 'proj':
            self.mlp = nn.Linear(dim*2, dim)
        elif merge_type == 'ignore':
            self.mlp = nn.Identity()
        elif merge_type == 'add':
            self.mlp = nn.Identity()
        else:
            raise ValueError('Invalid merge type: %s' % merge_type)
    
    def forward(self, x):
        b, n, c = x.shape
        # x: [B, N, C]
        if self.merge_type == 'proj':
            cls_token = x[:, 0:1]  # [B, 1, C]
            # expand cls token
            cls_token = cls_token.expand(b, n-1, c)  # [B, N-1, C]
            x = torch.cat([cls_token, x[:, 1:]], dim=-1) # [B, N-1, 2C]
            x = self.mlp(x)  # [B, N-1, C]
        elif self.merge_type == 'ignore':
            x = x[:, 1:]
        elif self.merge_type == 'add':
            cls_token = x[:, 0:1]
            x = cls_token + x[:, 1:]  # [B, N-1, C]
        
        return x


class Encoder(nn.Module):

    def __init__(self, arch, img_size=224, patch_size=16, num_classes=0, extract_layers=(2, 5, 8, 11), cls_token_merge='proj'):
        super().__init__()

        self.encoder = vits.__dict__[arch](patch_size=patch_size, num_classes=0, img_size=[img_size])

        self.cls_token_layers = list()
        for i in range(len(extract_layers)):
            self.cls_token_layers.append(ClsTokenLayer(cls_token_merge, self.encoder.embed_dim))
        self.cls_token_layers = nn.ModuleList(self.cls_token_layers)

        self.extract_layers = extract_layers
        self.patch_size = self.encoder.patch_embed.patch_size
        self.embed_dim = self.encoder.embed_dim
    
    def forward(self, x):
        # get feats
        feats = self.encoder.get_specific_layer(x, self.extract_layers)
        last_cls_token = feats[-1][:, 0]  # [B, C]

        # merge token
        out_feats = list()
        for i in range(len(self.extract_layers)):
            temp_feat = self.cls_token_layers[i](feats[i])
            h, w = int(x.shape[2] / self.patch_size), int(x.shape[3] / self.patch_size)
            out_feats.append(temp_feat.reshape(temp_feat.shape[0], h, w, -1).permute(0, 3, 1, 2).contiguous())
        out_feats.append(last_cls_token.unsqueeze(-1).unsqueeze(-1))
        
        return out_feats


class ConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, act_layer=nn.ReLU):
        super(ConvLayer, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            act_layer(inplace=True),
        )
    
    def forward(self, x):
        return self.layer(x)


class BasicResBlock(nn.Module):

    def __init__(self, in_channels, channels, out_channels, act_layer=nn.ReLU, attn=None):
        super(BasicResBlock, self).__init__()

        self.conv1 = ConvLayer(in_channels, channels, 3, 1, 1, act_layer)
        self.conv2 = ConvLayer(channels, out_channels, 3, 1, 1, nn.Identity)
        self.attn_layer = attn(out_channels) if attn else nn.Identity()

        if in_channels == out_channels:
            self.shortcut_conv = nn.Identity()
        else:
            self.shortcut_conv = ConvLayer(in_channels, out_channels, 1, 1, 0, nn.Identity)

        self.act_out = act_layer(inplace=True)
    
    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attn_layer(x)

        shortcut = self.shortcut_conv(shortcut)
        return self.act_out(x + shortcut)


class UnetBlock(nn.Module):

    def __init__(self, in_channels, channels, out_channels, n_block, act_layer=nn.ReLU, attn=None, interpolate=True):
        super(UnetBlock, self).__init__()

        self.first_block = ConvLayer(
            in_channels=in_channels,
            out_channels=channels,
            act_layer=act_layer
        )
        self.last_block = BasicResBlock(
            in_channels=channels,
            channels=out_channels,
            out_channels=out_channels,
            act_layer=act_layer,
            attn=attn 
        )

        self.mid_block = nn.Sequential(
            *[BasicResBlock(channels, channels, channels, act_layer, attn) for _ in range(n_block - 2)]
        )

        self.interpolate = interpolate
    
    def forward(self, x_de, x_en):
        if self.interpolate:
            x_de = F.interpolate(x_de, size=x_en.shape[-1], mode='bilinear', align_corners=True)

        if x_en is not None:
            x = torch.cat([x_de, x_en], 1)
        else:
            x = x_de

        x = self.first_block(x)
        x = self.mid_block(x)
        x = self.last_block(x)
        return x


class HyperFeats(nn.Module):

    def __init__(self, in_channels, out_ch):
        super().__init__()

        self.conv_list = nn.ModuleList()

        for in_ch in in_channels:
            self.conv_list.append(
                ConvLayer(in_ch, out_ch, 1, 1, 0, nn.Identity)
            )
    
    def forward(self, in_feats):

        out_feats = list()

        for i, in_feat in enumerate(in_feats):
            out_feats.append(
                F.interpolate(self.conv_list[i](in_feat), scale_factor=2**i, mode='bilinear', align_corners=True),
            )
        
        return torch.cat(out_feats, 1)


class DropSkip(nn.Module):

    def __init__(self, p=0.0):
        super().__init__()
        self.p = p
    
    def forward(self, enc_feat):
        if self.training:
            # for each sample, try to drop
            mask = torch.bernoulli(torch.ones(enc_feat.shape[0], device=enc_feat.device) * (1.0 - self.p))
            return enc_feat * mask.view(-1, 1, 1, 1)
        else:
            return enc_feat


class MyLN(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class MyRMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        """
        https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py
            Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super().__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(p=2, dim=1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return (self.scale * x_normed.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class Decoder(nn.Module):

    def __init__(self, n_block, input_dim, target_dims, target_size_ratio, act_layer=nn.ReLU, attn=None, last_norm=None, skip_drop_ratio=[0.0, 0.0, 0.0], mask_multi_level=False):
        super().__init__()

        # init linear dim map layers
        self.dim_map_layers = self.build_dim_map_layers(input_dim, target_dims[:-1])
        self.target_size_ratio = target_size_ratio[:-1]

        # init fusion layers
        self.decoder3 = UnetBlock(target_dims[-2] + target_dims[-3], target_dims[-3], target_dims[-3], n_block, act_layer, attn, interpolate=True)
        self.drop_skip3 = DropSkip(skip_drop_ratio[2])
        self.decoder2 = UnetBlock(target_dims[-3] + target_dims[-4], target_dims[-4], target_dims[-4], n_block, act_layer, attn, interpolate=True)
        self.drop_skip2 = DropSkip(skip_drop_ratio[1])
        self.decoder1 = UnetBlock(target_dims[-4] + target_dims[-5], target_dims[-5], target_dims[-5], n_block, act_layer, attn, interpolate=True)
        self.drop_skip1 = DropSkip(skip_drop_ratio[0])

        self.hyper_feats = HyperFeats(
            in_channels = [target_dims[-5], target_dims[-4], target_dims[-3], target_dims[-2]],
            out_ch = int(input_dim / 4)
        )

        if last_norm is None:
            self.last_norm = nn.Identity()
        elif last_norm == 'LN':
            self.last_norm = MyLN(int(input_dim / 4) * 4)
        elif last_norm == 'RMS':
            self.last_norm = MyRMSNorm(int(input_dim / 4) * 4)
        else:
            raise ValueError('last_norm must be None, LN or RMS')
        if mask_multi_level:
            self.output_mask = torch.zeros(1, int(input_dim / 4) * 4, 1, 1)  # [C, 1, 1]
            self.output_mask[0, :int(input_dim / 4)] = 1 # only the output of the last decoder block is used
        else:
            self.output_mask = torch.ones(1, int(input_dim / 4) * 4, 1, 1)    
    def build_dim_map_layers(self, input_dim, target_dims):

        map_layers = list()
        for target_dim in target_dims:
            map_layers.append(nn.Conv2d(input_dim, target_dim, 1, bias=False))
        
        return nn.ModuleList(map_layers)

    def forward(self, feats, merge_out=False):

        encoder_last_feat = feats[-2]

        # map to target dims & target sizes
        n_feat = len(feats)
        for i in range(n_feat-1):
            feats[i] = self.dim_map_layers[i](feats[i])
            if self.target_size_ratio[i] != 1:
                feats[i] = F.interpolate(feats[i], scale_factor=self.target_size_ratio[i], mode='bilinear', align_corners=True)
        
        # fusion
        e1, e2, e3, e4, cls_token = feats

        d3 = self.decoder3(e4, e3)
        d2 = self.decoder2(d3, e2)
        d1 = self.decoder1(d2, e1)
        # d3 = self.decoder3(*self.drop_skip3(e4, e3))
        # d2 = self.decoder2(*self.drop_skip2(d3, e2))
        # d1 = self.decoder1(*self.drop_skip1(d2, e1))

        res_feats = self.hyper_feats([d1, d2, d3, e4])
        res_feats = self.last_norm(res_feats)

        output_mask = self.output_mask.to(cls_token.device)
        
        return [res_feats * output_mask], cls_token, encoder_last_feat
        # return [d1, d2, d3, e4, cls_token], encoder_last_feat


class OnlyEncoderViTBackbone(nn.Module):

    def __init__(self, vit_arch, patch_size, drop_path_rate=0.0, img_size=224):

        super().__init__()

        self.encoder = Encoder(
            arch=vit_arch,
            patch_size=patch_size,
            extract_layers=[11, ],  # only the last layer
            cls_token_merge='ignore',
            drop_path_rate=drop_path_rate,
            img_size=img_size
        )

        self.decoder_dims = [self.encoder.embed_dim, self.encoder.embed_dim]
    
    def forward(self, x):
        return self.encoder(x)


class ViTBackbone(nn.Module):

    def __init__(
        self, 
        vit_arch, patch_size, extract_layers=(2, 5, 8, 11), cls_token_merge='ignore', 
        target_dim_ratio=(0.25, 0.50, 1.0, 2.0), target_size_ratio=(4, 8, 16, 32), last_norm=None, img_size=224, mask_multi_level=False, only_encoder=False, decoder_type='default'
    ):
        """
        This class is used to extract features from vit backbone.

        1. Encoder: extract features from vit backbone
        2. Decoder: rebuild feature map from encoder features

        args:
        vit_arch: vit architecture
        patch_size: patch size
        extract_layers: layers to extract features
        cls_token_merge: how to merge cls token
        target_dim_ratio: target dim ratio (related to the encoder dim)
        target_size_ratio: target size ratio (related to the patch_size).
        """
        super().__init__()

        if only_encoder:
            self.encoder = Encoder(
                arch=vit_arch,
                patch_size=patch_size,
                extract_layers=[11, ],  # only the last layer
                cls_token_merge='ignore',
                img_size=img_size
            )
            self.decoder = nn.Identity()
        else:
            # init encoder
            self.encoder = Encoder(
                arch=vit_arch,
                patch_size=patch_size,
                extract_layers=extract_layers,
                cls_token_merge=cls_token_merge,
                img_size=img_size
            )

            encoder_dim = self.encoder.embed_dim
            decoder_target_dims = [int(encoder_dim * target_dim_ratio[i]) for i in range(len(target_dim_ratio))] + [encoder_dim]  # the last is cls token
            decoder_target_ratio = [patch_size / target_size_ratio[i] for i in range(len(target_size_ratio))] + [1]  # the last is cls token

            if decoder_type == 'default':
                self.decoder = Decoder(
                    n_block=2,
                    input_dim=self.encoder.embed_dim,
                    target_dims=decoder_target_dims,
                    target_size_ratio=decoder_target_ratio,
                    attn=None,
                    last_norm=last_norm,
                    skip_drop_ratio=(0.00, 0.00, 0.00),
                    mask_multi_level=False
                )
            elif decoder_type == 'fpn':
                self.decoder_layers = nn.ModuleList()
                for in_ch in range(len(extract_layers)):
                    self.decoder_layers.append(
                        BasicResBlock(self.encoder.embed_dim, self.encoder.embed_dim, self.encoder.embed_dim // 4)
                    )
                self.decoder_layers.append(
                    ConvLayer(self.encoder.embed_dim, self.encoder.embed_dim, 1, 1, 0, nn.Identity)
                )

        self.decoder_dims = [int(self.encoder.embed_dim / 4) * 4]
        self.decoder_type = decoder_type
    
    def forward(self, x):

        feats = self.encoder(x)
        if self.decoder_type == 'default':
            return self.decoder(feats)
        elif self.decoder_type == 'fpn':
            e1, e2, e3, e4, cls_token = feats

            e1 = self.decoder_layers[0](e1)
            e2 = self.decoder_layers[1](e2)
            e3 = self.decoder_layers[2](e3)
            e4 = self.decoder_layers[3](e4)

            return [self.decoder_layers[-1](torch.cat([e1, e2, e3, e4], dim=1))], cls_token, feats[-2]


class MUSEViTFeatureExtractor(nn.Module):

    def __init__(self, model_name):
        super(MUSEViTFeatureExtractor, self).__init__()

        mask_multi_level = False
        self.only_encoder = False
        decoder_type = 'default'

        if model_name == 'muse_vit-s-16':
            model_weights = '/path/to/muse-vit_s_16-224.pth'
            arch = 'vit_small'
            patch_size = 16
            last_norm = None
            print(model_weights)
            img_size = 224
        elif model_name == 'muse_vit-b-16':
            model_weights = '/path/to/muse-vit_b_16-224.pth'
            arch = 'vit_base'
            patch_size = 16
            last_norm = None
            print(model_weights)
            img_size = 224
        elif model_name == 'lfov_muse_vit-b-16':
            model_weights = '/path/to/muse-vit_b_16-512.pth'
            arch = 'vit_base'
            patch_size = 16
            last_norm = None
            print(model_weights)
            img_size = 512
        elif model_name == 'lfov_muse_vit-s-16':
            model_weights = '/path/to/muse-vit_s_16-512.pth'
            arch = 'vit_small'
            patch_size = 16
            last_norm = None
            print(model_weights)
            img_size = 512
        else:
            raise ValueError('Unknown model name: {}'.format(model_name))

        fully_weights = torch.load(model_weights, map_location='cpu')['teacher']
        # filter
        loaded_weights = dict()
        for k, v in fully_weights.items():
            if k.startswith('module.backbone.'):
                new_k = k.replace('module.backbone.', '')
                if new_k == 'encoder.encoder.pos_embed':
                    v = self.interpolate_pos_encoding(v, patch_size, (img_size ** 2) // patch_size, img_size, img_size)
                loaded_weights[new_k] = v

        self.basic_model = ViTBackbone(arch, patch_size, last_norm=last_norm, img_size=img_size, mask_multi_level=mask_multi_level, only_encoder=self.only_encoder, decoder_type=decoder_type)
        self.basic_model.load_state_dict(loaded_weights)
        self.dense_feature_extractor = MultiLayerFeatureExtractorHead()
        
        # the last is the token of the image
        # for dense prediction, we remove this token
        self.out_dim = sum(self.basic_model.decoder_dims)
    
    def forward(self, input_img, input_coords):

        input_size = input_img.shape[-1]

        if self.only_encoder:
            feats, last_encoder_cls_token = self.basic_model(input_img)
            feats = [feats]
        else:
            feats, last_encoder_cls_token, last_encoder_feat = self.basic_model(input_img)
        return self.dense_feature_extractor(feats, input_coords, input_size)

    @staticmethod
    def interpolate_pos_encoding(pos_embed_params, patch_size, n_new_patch, target_w, target_h):
        npatch = n_new_patch
        N = pos_embed_params.shape[1] - 1
        if npatch == N and w == h:
            return pos_embed_params
        class_pos_embed = pos_embed_params[:, 0]
        patch_pos_embed = pos_embed_params[:, 1:]
        dim = patch_pos_embed.shape[-1]
        w0 = target_w // patch_size
        h0 = target_h // patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)


if __name__ == '__main__':

    test_arch = 'vit_tiny'
    specific_layers = [3, 7, 9, 11]

    test_model = ViTBackbone(test_arch, 16)
    print(test_model)

    test_input = torch.randn(1, 3, 224, 224)
    outs = test_model(test_input)
    for out in outs:
        print(out.shape)
