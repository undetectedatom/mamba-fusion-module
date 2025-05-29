import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedAttentionFusion(nn.Module):
    """
    Fusion method using learnable weights and channel attention
    """
    def __init__(self, channels):
        super().__init__()
        self.mamba_weight = nn.Parameter(torch.ones(1))
        self.conv_weight = nn.Parameter(torch.ones(1))
        
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(channels // 4, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x_mamba, x_conv):
        channel_weights = self.channel_attention(x_mamba + x_conv)        
        mamba_weight = torch.sigmoid(self.mamba_weight)
        conv_weight = torch.sigmoid(self.conv_weight)
        total_weight = mamba_weight + conv_weight
        mamba_weight = mamba_weight / total_weight
        conv_weight = conv_weight / total_weight
        return (x_mamba * mamba_weight + x_conv * conv_weight) * channel_weights

class FusionAdd(nn.Module):
    def __init__(self, channels):
        super().__init__()
    
    def forward(self, x_mamba, x_conv):
        return x_mamba + x_conv

class FusionConcat(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel_reduction = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.SiLU()
        )

    def forward(self, x_mamba, x_conv):
        x_cat = torch.cat([x_mamba, x_conv], dim=1)
        return self.channel_reduction(x_cat)
    
class FusionSE(nn.Module):
    def __init__(self, channels, r=8):
        super().__init__()
        self.fc1 = nn.Conv2d(channels*2, (channels*2)//r, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d((channels*2)//r, channels*2, kernel_size=1, bias=True)
        self.act = nn.SiLU()

    def forward(self, x_mamba, x_conv):
        B, C, H, W = x_conv.shape

        s_conv  = x_conv.mean((-2, -1), keepdim=True)   # (B, C, 1, 1)
        s_mamba = x_mamba.mean((-2, -1), keepdim=True)
        s = torch.cat([s_conv, s_mamba], dim=1)         # (B, 2C, 1, 1)
        w = self.act(self.fc1(s))                       # (B, 2C/r, 1, 1)
        w = self.fc2(w)                                 # (B, 2C,   1, 1)
        w = w.view(B, 2, C, 1, 1).softmax(dim=1)       # (B, 2, C, 1, 1)

        fused = w[:,0] * x_conv + w[:,1] * x_mamba     # (B, C, H, W)
        return fused

class FusionCrossSpatial(nn.Module):
    def __init__(self, channels): 
        super().__init__()      
        self.channels = channels
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.softmax_for_channel_weights = nn.Softmax(dim=-1)

    def forward(self, x_to_modulate, x_conv, x_mamba):
        B, C, H, W = x_conv.shape
        
        assert x_conv.shape[1] == x_mamba.shape[1], f"Channel dimensions must match: conv={x_conv.shape[1]}, mamba={x_mamba.shape[1]}"
        assert x_to_modulate.shape[1] == C, f"Input channels must match: input={x_to_modulate.shape[1]}, expected={C}"

        pooled_mamba = self.avgpool(x_mamba)             # (B, C, 1, 1)
        pooled_mamba = pooled_mamba.squeeze(-1).squeeze(-1)  # (B, C)
        pooled_mamba = pooled_mamba.unsqueeze(1)         # (B, 1, C)
        att_weights_from_mamba = self.softmax_for_channel_weights(pooled_mamba) # (B, 1, C)
        x_conv_flat = x_conv.reshape(B, C, H * W)       # (B, C, H*W)
        m_c_contribution = torch.matmul(att_weights_from_mamba, x_conv_flat) # (B, 1, H*W)
        pooled_conv = self.avgpool(x_conv)               # (B, C, 1, 1)
        pooled_conv = pooled_conv.squeeze(-1).squeeze(-1)  # (B, C)
        pooled_conv = pooled_conv.unsqueeze(1)           # (B, 1, C)
        att_weights_from_conv = self.softmax_for_channel_weights(pooled_conv) # (B, 1, C)
        x_mamba_flat = x_mamba.reshape(B, C, H * W)     # (B, C, H*W)
        c_m_contribution = torch.matmul(att_weights_from_conv, x_mamba_flat) # (B, 1, H*W)
        combined_spatial_map = (m_c_contribution + c_m_contribution).reshape(B, 1, H, W)
        spatial_modulation_weights = torch.sigmoid(combined_spatial_map) # (B, 1, H, W)

        return x_to_modulate * spatial_modulation_weights
    
def get_fusion_module(fusion_type: str, channels: int) -> nn.Module:
    fusion_modules = {
        'FusionSE': FusionSE,
        'FusionCrossSpatial': FusionCrossSpatial,
        'FusionAdd': FusionAdd,
        'FusionConcat': FusionConcat,
        'WeightedAttentionFusion': WeightedAttentionFusion
    }
    
    if fusion_type not in fusion_modules:
        raise ValueError(f"Unsupported fusion type: {fusion_type}. Available types: {list(fusion_modules.keys())}")
    
    return fusion_modules[fusion_type](channels)
