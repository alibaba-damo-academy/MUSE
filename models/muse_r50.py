import torch
from torch import nn
import timm
import torch.nn.functional as F
import numpy as np
from .nuclei_extractor import MultiLayerFeatureExtractorHead


# ===============================
# Define the UNet model
# In this case, we use the timm library to build the Encoder
# ===============================
class Encoder(nn.Module):

    def __init__(self, model_name):
        super().__init__()

        self.encoder = timm.create_model(model_name, pretrained=False, num_classes=0, global_pool='')
    
    def forward(self, x):
        return self.encoder(x)


# ===============================
# Define the Res Decoder
# ===============================
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

        # self.first_block = ConvLayer(
        #     in_channels=in_channels, 
        #     channels=in_channels,
        #     out_channels=channels,
        #     act_layer=act_layer,
        #     attn=attn
        # )
        self.first_block = ConvLayer(
            in_channels=in_channels,
            out_channels=channels,
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

        for in_ch_id, in_ch in enumerate(in_channels):
            self.conv_list.append(
                ConvLayer(in_ch, out_ch, 1, 1, 0, nn.Identity)
            )
    
    def forward(self, in_feats):

        out_feats = list()
        
        for i, in_feat in enumerate(in_feats):
            # should interpolate the feat to targe size
            out_feats.append(
                F.interpolate(self.conv_list[i](in_feat), size=in_feats[0].shape[-1], mode='bilinear', align_corners=True)
            )
        
        return torch.cat(out_feats, 1)


class CattedDecoder(nn.Module):

    def __init__(self, n_block=2, act_layer=nn.ReLU, attn=None, encoder_channels=None, skip_drop_ratio=[0.0, 0.0, 0.0], learnable_drop=False):
        super().__init__()

        self.decoder3 = UnetBlock(encoder_channels[-1] + encoder_channels[-2], encoder_channels[-2], encoder_channels[-2], n_block, act_layer, attn)
        self.drop_skip3 = DropSkip(encoder_channels[-2], skip_drop_ratio[2], learnable_drop)
        self.decoder2 = UnetBlock(encoder_channels[-2] + encoder_channels[-3], encoder_channels[-3], encoder_channels[-3], n_block, act_layer, attn)
        self.drop_skip2 = DropSkip(encoder_channels[-3], skip_drop_ratio[1], learnable_drop)
        self.decoder1 = UnetBlock(encoder_channels[-3] + encoder_channels[-4], encoder_channels[-4], encoder_channels[-4], n_block, act_layer, attn)
        self.drop_skip1 = DropSkip(encoder_channels[-4], skip_drop_ratio[0], learnable_drop)

        self.hyper_feats = HyperFeats(
            in_channels = [encoder_channels[-4], encoder_channels[-3], encoder_channels[-2], encoder_channels[-1]],
            out_ch = encoder_channels[-1] // 8,  # we suppose the out_ch is equal to half of the image-level dim of the encoder
        )

        self.decoder_dims = [(encoder_channels[-1] // 8) * 4, encoder_channels[-1]]
    
    def forward(self, feats, merge_out=False):
        e2,e3,e4,e5 = feats
        d4 = self.decoder3(e5, e4) # /16
        d3 = self.decoder2(d4, e3) # /8
        d2 = self.decoder1(d3, e2) # /4

        feats = self.hyper_feats([d2, d3, d4, e5])
        return [feats, e5]  # the e5 is used to build the image-level loss/feature


class DropSkip(nn.Module):

    def __init__(self, n_c, p=0.0, learnable=False):
        super().__init__()
        self.p = p
        self.learnable = learnable
        if self.learnable:
            self.param = nn.Parameter(torch.randn(n_c))
        else:
            self.param = torch.zeros(n_c)
    
    def forward(self, enc_feat):
        if self.training:
            # for each sample, try to drop
            mask = torch.bernoulli(torch.ones(enc_feat.shape[0], device=enc_feat.device) * (1.0 - self.p)).view(-1, 1, 1, 1).contiguous()
            return enc_feat * mask
        else:
            return enc_feat


class ConvBackbone(nn.Module):

    def __init__(
        self, 
        model_name,
        skip_drop_ratio=(0.15, 0.10, 0.05), learnable_drop=False):
        super().__init__()

        # init encoder
        self.encoder = Encoder(model_name)
        assert model_name == 'resnet50', f'Create encoder {model_name} successfully, but only support resnet50 in this version'

        if model_name == 'resnet50':
            encoder_dims = [256, 512, 1024, 2048]
        self.embed_dims = [1024, 2048]

        self.decoder = CattedDecoder(
            n_block=2,
            act_layer=nn.ReLU,
            attn=None,
            encoder_channels=encoder_dims,
            skip_drop_ratio=skip_drop_ratio,
            learnable_drop=learnable_drop
        )
        self.decoder_dims = self.decoder.decoder_dims

    def forward(self, x):

        feats = self.encoder(x)
        return self.decoder(feats)


class MUSEConvFeatureExtractor(nn.Module):

    def __init__(self, model_name):
        super(MUSEConvFeatureExtractor, self).__init__()

        if model_name == 'muse_r50':
            model_weights = '/path/to/muse-r50-224.pth'
            print(model_weights)
        elif model_name == 'lfov_muse_r50':
            model_weights = '/path/to/muse-r50-512.pth'
            print(model_weights)

        self.basic_model = ConvBackbone(
            model_name='resnet50',
            skip_drop_ratio=(0.0, 0.0, 0.0),  # no skip for evaluation
        )

        fully_weights = torch.load(model_weights, map_location='cpu')['teacher']
        # filter
        loaded_weights = dict()
        for k, v in fully_weights.items():
            if k.startswith('module.backbone.'):
                new_k = k.replace('module.backbone.', '')
                loaded_weights[new_k] = v

        self.basic_model.load_state_dict(loaded_weights)
        self.dense_feature_extractor = MultiLayerFeatureExtractorHead()
        
        self.out_dim = self.basic_model.decoder_dims[0]

        self.dense_feature_extractor = MultiLayerFeatureExtractorHead()
    
    def forward(self, input_img, input_coords):

        input_size = input_img.shape[-1]

        feats = self.basic_model(input_img)

        return self.dense_feature_extractor([feats[0]], input_coords, input_size)
