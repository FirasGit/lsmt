""" This code is adapted from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py"""

import torch
from torch import nn
from einops import rearrange, repeat
import timm
import random
import math
from classification.models.utils.ehr_module import EHRModule


class PositionalEncoding(nn.Module):
    """
    Taken form https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """ 
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.permute(1, 0, 2) 
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]

        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, *, input_dim, dim, depth, heads, mlp_dim, pool = 'cls', dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_embedding = nn.Sequential(
            nn.Linear(input_dim, dim),
        )

        self.pos_embedding = PositionalEncoding(d_model=dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

    def forward(self, x):
        x = self.to_embedding(x)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_embedding(x)
        x = self.dropout(x)

        x = self.transformer(x)

        #x = x.mean(dim = 1) if self.pool == 'mean' else x[:, :1]

        return x

class LSMT(nn.Module):
    def __init__(self, *, model_name, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pretrained, ehr_module, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., p_visual_dropout=0., p_feature_dropout=0., p_modality_dropout=0., deep_supervision=False, pretrain_cxr=False, deep_supervised_merge=False):
        super().__init__()

        self.feature_extractor_1 = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            in_chans=channels,
            img_size=image_size,
        )

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_lab_embedding = EHRModule(num_freq_bands=ehr_module.num_freq_bands, depth=ehr_module.depth, max_freq=ehr_module.max_freq, input_channels=ehr_module.input_channels, input_axis=ehr_module.input_axis, num_latents=ehr_module.num_latents, latent_dim=ehr_module.latent_dim, cross_heads=ehr_module.cross_heads, latent_heads=ehr_module.latent_heads, cross_dim_head=ehr_module.cross_dim_head, latent_dim_head=ehr_module.latent_dim_head, attn_dropout=ehr_module.attn_dropout, ff_dropout=ehr_module.ff_dropout, weight_tie_layers=ehr_module.weight_tie_layers, fourier_encode_data=ehr_module.fourier_encode_data, self_per_cross_attn=ehr_module.self_per_cross_attn, final_classifier_head=ehr_module.final_classifier_head)
        self.to_cxr_dimensionality = nn.Sequential(
            nn.Linear(ehr_module.latent_dim, dim),
        )

        num_patches = (image_height // patch_height) * (image_width // patch_width) + 1 
        num_patches += 1 # Account for lab value

        self.pos_embedding = PositionalEncoding(d_model=dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.p_visual_dropout = p_visual_dropout
        self.p_feature_dropout = p_feature_dropout
        self.p_modality_dropout = p_modality_dropout

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        self.deep_supervision = deep_supervision
        self.deep_supervised_merge = deep_supervised_merge
        if self.deep_supervision:
            self.mlp_head_ehr = nn.Sequential(nn.LayerNorm(ehr_module.latent_dim), nn.Linear(ehr_module.latent_dim, num_classes))
            self.mlp_head_cxr = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))
        
            if self.deep_supervised_merge:
                self.mlp_deep_supervised_merge = nn.Sequential(nn.LayerNorm(3*num_classes), nn.Linear(3*num_classes, num_classes))
        
        self.pretrain_cxr = pretrain_cxr
        self.mlp_head_cxr_pt = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, self.pretrain_cxr.num_cxr_targets))

    def forward(self, data):
        img_1, ehr = data[0], data[1]

        # Visual dropout
        if self.training:
            if random.random() <= self.p_visual_dropout:
                img_1 = torch.zeros_like(img_1)

        # Feature dropout
        if self.training:
            if random.random() <= self.p_feature_dropout:
                ehr = torch.zeros_like(ehr)

        # Modality dropout
        if self.training:
            if random.random() <= self.p_modality_dropout:
                img_1 = torch.zeros_like(img_1)
            elif random.random() <= self.p_modality_dropout:
                ehr = torch.zeros_like(ehr)
        x_image1 = self.feature_extractor_1.forward_features(img_1)

		# B, T, D
        ehr = ehr
        x_ehr = self.to_lab_embedding(ehr)
        x_ehr_dim = self.to_cxr_dimensionality(x_ehr)

        x = torch.cat((x_image1, x_ehr_dim), dim=1)

        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)

        x += self.pos_embedding(x)

        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)

        if self.deep_supervision:
            if self.deep_supervised_merge:
                out = self.mlp_deep_supervised_merge(torch.concat([self.mlp_head(x), self.mlp_head_ehr(x_ehr.mean(dim=1)), self.mlp_head_cxr(x_image1.mean(dim=1))], dim=1))
            else:
                out = [self.mlp_head(x), self.mlp_head_ehr(x_ehr.mean(dim=1)), self.mlp_head_cxr(x_image1.mean(dim=1))]

        else:
            out = self.mlp_head(x)
        
        if self.pretrain_cxr.is_true:
            out = self.mlp_head_cxr_pt(x)

        return out