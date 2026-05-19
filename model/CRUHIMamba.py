import math
import torch
from torch import nn
from mamba_ssm import Mamba
import torch.nn.functional as F
import numpy as np
from calflops import calculate_flops

class SSMB_Spe(nn.Module):    # Spectrum Mamba Module
    def __init__(self, channels, Spagroup_num=1, use_residual=True, GNgroup_num=4):
        super(SSMB_Spe, self).__init__()
        self.Spagroup_num = Spagroup_num
        self.use_residual = use_residual

        self.group_channel_num = math.ceil(channels/Spagroup_num)
        self.channel_num = self.Spagroup_num * self.group_channel_num

        self.mamba = Mamba( # This module uses roughly 3 * expand * d_model^2 parameters
                            d_model=self.group_channel_num,  # Model dimension d_model
                            d_state=16,  # SSM state expansion factor
                            d_conv=4,  # Local convolution width
                            expand=1,  # Block expansion factor
                            )

        self.proj = nn.Sequential(
            nn.GroupNorm(GNgroup_num, self.channel_num),
            nn.SiLU()
        )

    def padding_feature(self, x):
        B, C, H, W = x.shape
        if C < self.channel_num:
            pad_c = self.channel_num - C
            pad_features = torch.zeros((B, pad_c, H, W)).to(x.device)
            cat_features = torch.cat([x, pad_features], dim=1)
            return cat_features
        else:
            return x

    def forward(self, x):
        x_pad = self.padding_feature(x)
        x_pad = x_pad.permute(0, 2, 3, 1).contiguous()
        B, H, W, C_pad = x_pad.shape
        x_flat = x_pad.view(B * H * W, self.Spagroup_num, self.group_channel_num)
        x_flat = self.mamba(x_flat)
        x_recon = x_flat.view(B, H, W, C_pad)
        x_recon = x_recon.permute(0, 3, 1, 2).contiguous()
        x_proj = self.proj(x_recon)
        if self.use_residual:
            return x + x_proj
        else:
            return x_proj


class SSMB_Spa(nn.Module):   # Spatial Mamba Module
    def __init__(self, channels, use_residual=True, GNgroup_num=4, use_proj=True):
        super(SSMB_Spa, self).__init__()
        self.use_residual = use_residual
        self.use_proj = use_proj
        self.mamba = Mamba(  # This module uses roughly 3 * expand * d_model^2 parameters
                           d_model=channels,  # Model dimension d_model
                           d_state=16,  # SSM state expansion factor
                           d_conv=4,  # Local convolution width
                           expand=4,  # Block expansion factor
                           )
        if self.use_proj:
            self.proj = nn.Sequential(
                nn.GroupNorm(GNgroup_num, channels),
                nn.SiLU()
            )

    def forward(self, x):
        x_re = x.permute(0, 2, 3, 1).contiguous()
        B, H, W, C = x_re.shape
        x_flat = x_re.view(1, -1, C)
        x_flat = self.mamba(x_flat)

        x_recon = x_flat.view(B, H, W, C)
        x_recon = x_recon.permute(0, 3, 1, 2).contiguous()
        if self.use_proj:
            x_recon = self.proj(x_recon)
        if self.use_residual:
            return x_recon + x
        else:
            return x_recon


class SpaAddSpe(nn.Module):    # Spatial and spectral features are simply added together
    def __init__(self, channels, Spagroup_num, use_residual, GNgroup_num=4):
        super(SpaAddSpe, self).__init__()
        self.use_residual = use_residual
        self.SSMBspa = SSMB_Spa(channels, use_residual=use_residual, GNgroup_num=GNgroup_num)
        self.SSMBspe = SSMB_Spe(channels, Spagroup_num=Spagroup_num, use_residual=use_residual, GNgroup_num=GNgroup_num)

    def forward(self, x):
        spa_x = self.SSMBspa(x)
        spe_x = self.SSMBspe(x)
        fusion_x = spa_x + spe_x
        if self.use_residual:
            return fusion_x + x
        else:
            return fusion_x


class SSMBackbone(nn.Module):    # Spatial-Spectral Feature Adaptive Weighted Fusion
    def __init__(self, channels, Spagroup_num, use_residual, GNgroup_num=4, use_att=True, num_classes=10):
        super(SSMBackbone, self).__init__()
        self.use_att = use_att
        self.use_residual = use_residual
        if self.use_att:
            self.weights = nn.Parameter(torch.ones(2)/2)
            self.softmax = nn.Softmax(dim=0)

        self.SSMBspa = SSMB_Spa(channels, use_residual=use_residual, GNgroup_num=GNgroup_num)
        self.SSMBspe = SSMB_Spe(channels, Spagroup_num=Spagroup_num, use_residual=use_residual, GNgroup_num=GNgroup_num)

        self.spa_cls_head = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(GNgroup_num, 128),
            nn.SiLU(),
            nn.Conv2d(in_channels=128, out_channels=num_classes, kernel_size=1, stride=1, padding=0)
        )
        self.spe_cls_head = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(GNgroup_num, 128),
            nn.SiLU(),
            nn.Conv2d(in_channels=128, out_channels=num_classes, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        spa_x = self.SSMBspa(x)
        spe_x = self.SSMBspe(x)
        if self.use_att:
            weights = self.softmax(self.weights)
            fusion_x = spa_x * weights[0] + spe_x * weights[1]
        else:
            fusion_x = spa_x + spe_x
        if self.use_residual:
            fusion_x = fusion_x + x
        else:
            return fusion_x

        spa_logits = self.spa_cls_head(spa_x)
        spe_logits = self.spe_cls_head(spe_x)

        return fusion_x, spa_logits, spe_logits

class CRUHIMamba(nn.Module):  # CRUHI-Mamba;   mamba_type='AFDFM' / 'SSMBspa' / 'SSMBspe' / 'SpaAddSpe'
    # CRUHI-Mamba;   mamba_type='AFDFM' / 'SSMBspa' / 'SSMBspe' / 'SpaAddSpe'   # For ablation studies
    # mamba_type=‘AFDFM’ represents the complete structure of the CRUHI-Mamba model
    def __init__(self, in_channels=128, hidden_dim=64, num_classes=16, use_residual=True, mamba_type='AFDFM',
                 Spagroup_num=8, GNgroup_num=16, use_att=True):
        super(CRUHIMamba, self).__init__()
        self.mamba_type = mamba_type
        self.use_att = use_att
        if self.use_att:
            # Initialize weight parameters
            self.weights_3 = nn.Parameter(torch.ones(3) / 3)
            self.softmax_3 = nn.Softmax(dim=0)

        self.patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(GNgroup_num, hidden_dim),
            nn.SiLU()
        )

        if mamba_type == 'SSMBspa':
            self.mamba = nn.Sequential(
                SSMB_Spa(hidden_dim, use_residual=use_residual, GNgroup_num=GNgroup_num),
                )

        elif mamba_type == 'SSMBspe':
            self.mamba = nn.Sequential(
                SSMB_Spe(hidden_dim, Spagroup_num=Spagroup_num, use_residual=use_residual, GNgroup_num=GNgroup_num)
                )

        elif mamba_type == 'SpaAddSpe':
            self.mamba = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                SpaAddSpe(channels=hidden_dim, Spagroup_num=Spagroup_num, use_residual=use_residual, GNgroup_num=GNgroup_num)
            )

        elif mamba_type == 'AFDFM':
            self.mamba = nn.ModuleList([
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                SSMBackbone(channels=hidden_dim, Spagroup_num=Spagroup_num, use_residual=use_residual,
                            GNgroup_num=GNgroup_num, use_att=use_att, num_classes=num_classes)
            ])
        else:
            self.mamba = nn.Sequential()    # Add the appropriate modules based on mamba_type, and customize

        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(GNgroup_num, 128),
            nn.SiLU(),
            nn.Conv2d(in_channels=128, out_channels=num_classes, kernel_size=1, stride=1, padding=0)
        )


    def forward(self, x):
        x = self.patch_embedding(x)
        spa_logits, spe_logits = None, None

        if self.mamba_type == 'AFDFM':
            for module in self.mamba:
                if isinstance(module, SSMBackbone):
                    x, spa_logits, spe_logits = module(x)
                else:
                    x = module(x)
            fusion_x = x
            both_logits = self.cls_head(fusion_x)

            if self.use_att:
                weights = self.softmax_3(self.weights_3)
                fusion_x_logits = spa_logits * weights[0] + spe_logits * weights[1] + both_logits * weights[2]

            else:
                fusion_x_logits = (spa_logits + spe_logits + both_logits) / 3

            return fusion_x_logits

        elif self.mamba_type == 'SSMBspa':
            spa = self.mamba(x)
            SSMBspa_logits = self.cls_head(spa)
            return SSMBspa_logits

        elif self.mamba_type == 'SSMBspe':
            spe = self.mamba(x)
            SSMBspe_logits = self.cls_head(spe)
            return SSMBspe_logits

        elif self.mamba_type == 'SpaAddSpe':
            spaspe = self.mamba(x)
            SSMB_logits = self.cls_head(spaspe)
            return SSMB_logits

        else:
            x = self.mamba(x)
            logits = self.cls_head(x)
            return logits


# if __name__ == "__main__":
#     # Generate input
#     B, C, H, W = 1, 100, 256, 256
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     input_tensor = torch.randn(B, C, H, W, device=device)
#
#     # Create a model instance
#     model = CRUHIMamba(in_channels=C, num_classes=14, mamba_type='AFDFM').to(device)
#
#     optimized_logits = model(input_tensor)
#
#     model.eval()
#
#     # Two Methods for Calculating Model Complexity
#     flops, macs1, para = calculate_flops(model=model, input_shape=(1, input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3]))
#     print(flops, macs1, para)
#
#     def count_parameters(model):
#         total_params = sum(p.numel() for p in model.parameters())
#         trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#         return total_params, trainable_params
#     total_params, trainable_params = count_parameters(model=model)
#     print(total_params, trainable_params)

