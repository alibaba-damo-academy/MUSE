import torch
from torch import nn
import torch.nn.functional as F


class MultiLayerFeatureExtractorHead(nn.Module):
    """
    We need to extract each layer features and cat.
    """

    def __init__(self):
        super().__init__()
    
    def forward(self, input_feats, input_coords, input_size):
        dense_feats = list()
        n_level = len(input_feats)
        n_bs = input_feats[0].shape[0]
        device = input_feats[0].device
        for ib in range(n_bs):
            
            sample_feats = list()

            query_coords = input_coords[ib].to(device)
            query_coords = 2 * query_coords / input_size - 1
            for il in range(n_level):
                _dense_cell_feats = F.grid_sample(
                    input_feats[il][ib:ib+1],
                    query_coords.unsqueeze(0).unsqueeze(0),
                    mode='bilinear',
                    align_corners=True,
                ).permute(0, 2, 3, 1).squeeze(0, 1)

                sample_feats.append(_dense_cell_feats)
            dense_feats.append(torch.cat(sample_feats, dim=-1))
        
        return dense_feats
