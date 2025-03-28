# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        # self.double_conv = nn.Sequential(
        #     nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
        #     nn.GroupNorm(1, mid_channels),
        #     nn.GELU(),
        #     nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
        #     nn.GroupNorm(1, out_channels),
        # )
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            # nn.GroupNorm(1, mid_channels),
            # nn.GELU(),
            # nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            # return F.gelu(x + self.double_conv(x))
            return x + self.double_conv(x)
        else:
            return self.double_conv(x)



class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            # DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=4, stride=2, padding=1, output_padding=1)
        self.conv = nn.Sequential(
            # DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        if x.size()[-1] != skip_x.size()[-1]:
            # x = nn.functional.interpolate(x, size=(skip_x.size()[2], skip_x.size()[3]), mode='nearest')
            x = nn.functional.interpolate(x, size=(skip_x.size()[2], skip_x.size()[3]), mode='bilinear')
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet(nn.Module):
    def __init__(self, c_in_l=3, c_in_i=3, c_out_l=3, c_out_i=3, time_dim=256, img_size=None, device="cuda"):
        super().__init__()
        if img_size is None:
            img_size = [13, 6, 3]
        self.device = device
        self.time_dim = time_dim
        self.inc_l = DoubleConv(c_in_l, 32)
        self.inc_i = DoubleConv(c_in_i, 32)
        self.down1_l = Down(32, 64)
        self.down1_i = Down(32, 64)
        self.down2_l = Down(64, 64)
        self.down2_i = Down(64, 64)

        self.bot1 = DoubleConv(128, 128)
        self.bot2 = DoubleConv(128, 64)
        self.bot3 = DoubleConv(128, 64)

        self.up1_l = Up(128, 32)
        self.up1_i = Up(128, 32)
        self.up2_l = Up(64, 32)
        self.up2_i = Up(64, 32)
        self.Weight_Alpha = nn.Parameter(torch.ones(2), requires_grad=True)

        self.outc_l_1 = DoubleConv(32, 48)
        self.outc_i_1 = DoubleConv(32, 48)
        self.outc = DoubleConv(96, 48)

        self.outc_l_2 = DoubleConv(32, c_out_l)
        self.outc_i_2 = DoubleConv(32, c_out_i)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2).float() / channels)
        ).to(self.device)
        pos_enc_a = torch.sin(t.repeat(1, channels // 2).to(self.device) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2).to(self.device) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x_l, x_i, t_l, t_i):
        t_l = t_l.unsqueeze(-1).type(torch.float)
        t_l = self.pos_encoding(t_l, self.time_dim)
        t_i = t_i.unsqueeze(-1).type(torch.float)
        t_i = self.pos_encoding(t_i, self.time_dim)

        x1_l = self.inc_l(x_l)
        x1_i = self.inc_i(x_i)
        x2_l = self.down1_l(x1_l, t_l)
        x2_i = self.down1_i(x1_i, t_i)
        x3_l = self.down2_l(x2_l, t_l)
        x3_i = self.down2_i(x2_i, t_i)
        weight_alpha = F.softmax(self.Weight_Alpha, 0)
        x4 = self.bot1(torch.cat((x3_l * weight_alpha[0], x3_i * weight_alpha[1]), dim=1))

        # x4 = self.bot1(torch.cat((x3_l, x3_i), dim=1))
        # x4 = self.bot1(weight_alpha[0]*x3_l.add(weight_alpha[1]*x3_i))
        x4_l = self.bot2(x4)
        x4_i = self.bot3(x4)

        x5_l = self.up1_l(x4_l, x2_l, t_l)
        x5_i = self.up1_i(x4_i, x2_i, t_i)
        x5_l = self.up2_l(x5_l, x1_l, t_l)
        x5_i = self.up2_i(x5_i, x1_i, t_i)

        x_l_fus = self.outc_l_1(x5_l)
        x_i_fus = self.outc_i_1(x5_i)

        # weight_alpha = F.softmax(self.Weight_Alpha, 0)
        # output = torch.cat((x_l_fus * weight_alpha[0], x_i_fus * weight_alpha[1]), dim=1)
        output = torch.cat((x_l_fus, x_i_fus), dim=1)
        output = self.outc(output)

        output_l = self.outc_l_2(x5_l)
        output_i = self.outc_i_2(x5_i)

        return output, output_l, output_i


class UNet_conditional(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, num_classes=None, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 32)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 16)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 8)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, y):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        if y is not None:
            t += self.label_emb(y)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output



