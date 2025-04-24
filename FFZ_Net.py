import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

from functools import partial
# from .resnet import resnet34
from torchvision.models import resnet50
from .cflde_resnet import CFLDE_resnet34_S

from timm.models.layers import DropPath, to_2tuple, trunc_normal_



nonlinearity = partial(F.relu, inplace=True)


class ca_layer(nn.Module):
    def __init__(self, channel, k_size=3):
        super(ca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)

        return x * y.expand_as(x)



def BNReLU(num_features):
    
    return nn.Sequential(
                nn.BatchNorm2d(num_features),
                nn.ReLU()
            )



# ############################################## drop block ###########################################

class Drop(nn.Module):
    def __init__(self, drop_rate=0.1, block_size=2):
        super(Drop, self).__init__()
 
        self.drop_rate = drop_rate
        self.block_size = block_size
 
    def forward(self, x):
    
        if not self.training:
            return x
        
        if self.drop_rate == 0:
            return x
            
        gamma = self.drop_rate / (self.block_size**2)
        mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()
 
        mask = mask.to(x.device)

        block_mask = self._compute_block_mask(mask)
        out = x * block_mask[:, None, :, :]
        out = out * block_mask.numel() / block_mask.sum()
        return out
 
    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(input=mask[:, None, :, :],
                                  kernel_size=(self.block_size,
                                               self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)
        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]
        block_mask = 1 - block_mask.squeeze(1)
        return block_mask



# ################################ CFLDE at decoder stage ######################################################################


# PConv相关

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class PConv(nn.Module):  
    ''' Pinwheel-shaped Convolution using the Asymmetric Padding method. '''
    
    def __init__(self, c1, c2, k, s):
        super().__init__()

        # self.k = k
        p = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]
        self.pad = [nn.ZeroPad2d(padding=(p[g])) for g in range(4)]
        self.cw = Conv(c1, c2 // 4, (1, k), s=s, p=0)
        self.ch = Conv(c1, c2 // 4, (k, 1), s=s, p=0)
        self.cat = Conv(c2, c2, 2, s=1, p=0)

    def forward(self, x):
        yw0 = self.cw(self.pad[0](x))
        yw1 = self.cw(self.pad[1](x))
        yh0 = self.ch(self.pad[2](x))
        yh1 = self.ch(self.pad[3](x))
        return self.cat(torch.cat([yw0, yw1, yh0, yh1], dim=1))



class CFLDE_DecoderBlock_V2(nn.Module):
    def __init__(self, in_channels, n_filters, rla_channel=32, CA_size=5, reduction=16):
        super(CFLDE_DecoderBlock_V2, self).__init__()

        self.conv1_ct = nn.Conv2d(in_channels+rla_channel, in_channels // 4, 1)
        self.norm1_ct = nn.BatchNorm2d(in_channels // 4)
        self.silu1_ct = nn.SiLU()

        self.deconv2_ct = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2_ct = nn.BatchNorm2d(in_channels // 4)
        self.silu2_ct = nn.SiLU()

        self.conv3_ct = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3_ct = nn.BatchNorm2d(n_filters)
        self.silu3_ct = nn.SiLU()

        self.conv1_pt = PConv(in_channels+rla_channel, in_channels//4, 1, 1)
        self.norm1_pt = nn.BatchNorm2d(in_channels // 4)
        self.silu1_pt = nn.SiLU()

        self.deconv2_pt = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2_pt = nn.BatchNorm2d(in_channels // 4)
        self.silu2_pt = nn.SiLU()

        self.conv3_pt = PConv(in_channels//4, n_filters, 1, 1)
        self.norm3_pt = nn.BatchNorm2d(n_filters)
        self.silu3_pt = nn.SiLU()

        self.expansion = 1
        
        self.deconv_h = nn.ConvTranspose2d(rla_channel, rla_channel, 3, stride=2, padding=1, output_padding=1)
        self.deconv_x_ct = nn.ConvTranspose2d(in_channels, n_filters, 3, stride=2, padding=1, output_padding=1)
        self.deconv_x_pt = nn.ConvTranspose2d(in_channels, n_filters, 3, stride=2, padding=1, output_padding=1)
        
            
        self.ca = None
        if CA_size != None:
            self.ca_ct = ca_layer(n_filters * self.expansion, int(CA_size))
            self.ca_pt = ca_layer(n_filters * self.expansion, int(CA_size))
            
        self.conv_out_ct = nn.Conv2d(n_filters, rla_channel, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.conv_out_pt = nn.Conv2d(n_filters, rla_channel, kernel_size=(1, 1), stride=(1, 1), bias=False)

        self.conv_out_h = nn.Conv2d(rla_channel*3, rla_channel, 3, 1, 1) 
        self.norm4 = nn.BatchNorm2d(rla_channel)
        self.tanh = nn.Tanh()        
        
        
    def forward(self, ct_x, h, pt_x):
        ct_identity = ct_x              
        pt_identity = pt_x               
        
        ct_x_new = torch.cat((ct_x, h), dim=1)  
        pt_x_new = torch.cat((pt_x, h), dim=1)  
        
        for index in range(ct_x_new.shape[1]):
            if index % 2 == 0:
                ct_x_new[:, index, :, :] = ct_x[:, index//2, :, :]
                pt_x_new[:, index, :, :] = pt_x[:, index//2, :, :]
            else:
                ct_x_new[:, index, :, :] = h[:, index//2, :, :]
                pt_x_new[:, index, :, :] = h[:, index//2, :, :]
    
        ct_out = self.conv1_ct(ct_x_new)         
        ct_out = self.norm1_ct(ct_out)
        ct_out = self.silu1_ct(ct_out)
        
        ct_out = self.deconv2_ct(ct_out)   
        ct_out = self.norm2_ct(ct_out)
        ct_out = self.silu2_ct(ct_out)
        
        ct_out = self.conv3_ct(ct_out)  
        ct_out = self.norm3_ct(ct_out)

        pt_out = self.conv1_pt(pt_x_new)
        pt_out = self.norm1_pt(pt_out)
        pt_out = self.silu1_pt(pt_out)

        pt_out = self.deconv2_pt(pt_out)
        pt_out = self.norm2_pt(pt_out)
        pt_out = self.silu2_pt(pt_out)

        pt_out = self.conv3_pt(pt_out)
        pt_out = self.norm3_pt(pt_out)
        
        if self.ca != None:
            ct_out = self.ca_ct(ct_out)   
            pt_out = self.ca_pt(pt_out)

        
        ct_y = ct_out                    
        pt_y = pt_out
        ct_identity = self.deconv_x_ct(ct_identity)   
        pt_identity = self.deconv_x_pt(pt_identity)
        ct_out += ct_identity
        pt_out += pt_identity
        ct_out = self.silu3_ct(ct_out)                 
        pt_out = self.silu3_pt(pt_out)
        
        ct_y_h = self.conv_out_ct(ct_y)    
        pt_y_h = self.conv_out_pt(pt_y)
        h = self.deconv_h(h)         
        h = self.norm4(h)
        h = self.tanh(h)     

        return ct_out, h, pt_out


# Global Feature Enhance Module
class Mlp(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(input_channels, hidden_channels)
        self.silu = nn.SiLU()
        self.fc2 = nn.Linear(hidden_channels, output_channels)

    def forward(self, x):
        x = self.fc1(x)
        x = self.silu(x)
        x = self.fc2(x)
        return x
    

class Dynamic3DConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Dynamic3DConv, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=(kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2))
        self.dynamic_weight_net = nn.Sequential(
            nn.Linear(in_channels, out_channels*kernel_size[0]*kernel_size[1]*kernel_size[2]),
            nn.SiLU(),
            nn.Linear(out_channels*kernel_size[0]*kernel_size[1]*kernel_size[2], out_channels * out_channels * kernel_size[0] * kernel_size[1] * kernel_size[2])
        )

    def forward(self, x):
        batch_size, _, D, H, W = x.shape
        x_permute = x.permute(1, 0, 2, 3, 4)
        dynamic_weights = self.dynamic_weight_net(x_permute.mean(dim=[1, 2, 3, 4]))
        dynamic_weights = dynamic_weights.view(self.conv.out_channels, self.conv.out_channels, *self.kernel_size)

        x = nn.functional.conv3d(x, dynamic_weights, padding=0, stride=1)
        return x


class GFFE_M(nn.Module):
    def __init__(self, in_channels=49, dimension=128):
        super(GFFE_M, self).__init__()
        self.dimension = dimension

        self.ct_mlp = Mlp(in_channels, in_channels, in_channels)
        self.h_mlp = Mlp(in_channels, in_channels, in_channels)
        self.pt_mlp = Mlp(in_channels, in_channels, in_channels)

        self.silu = nn.SiLU()

        self.dynamic_conv = Dynamic3DConv(in_channels=128, out_channels=128, kernel_size=(3, 1, 1))
        self.bn_3d = nn.BatchNorm3d(128)
        self.dynamic_conv_out = Dynamic3DConv(in_channels=128, out_channels=128, kernel_size=(1, 1, 1))


    def forward(self, ct_fea, h_fea, pt_fea):
        b, c, h, w = ct_fea.size()
        ct_flatten = ct_fea.view(b, c, h*w)
        h_flatten = h_fea.view(b, c, h*w)
        pt_flatten = pt_fea.view(b, c, h*w)

        ct_key_value = self.ct_mlp(ct_flatten)
        h_query = self.h_mlp(h_flatten)
        pt_key_value = self.pt_mlp(pt_flatten)

        ct_key = ct_key_value.permute(0, 2, 1)
        ct_query_key = torch.matmul(h_query, ct_key)
        ct_map = (self.dimension ** -.5) * ct_query_key
        ct_map = F.softmax(ct_map, dim=-1)
        h_ct = torch.matmul(ct_map, ct_key_value) 

        pt_key = pt_key_value.permute(0, 2, 1)
        pt_query_key = torch.matmul(h_query,pt_key)
        pt_map = (self.dimension ** -.5) * pt_query_key
        pt_map = F.softmax(pt_map, dim=-1)
        h_pt = torch.matmul(pt_map, pt_key_value) 

        h_ct = h_ct.view(b, c, h, w) 
        h_pt = h_pt.view(b, c, h, w) 

        h_ct_pt = torch.stack([h_ct, h_pt, h_fea], dim=2)
        h_ct_pt_fuse1 = self.silu(self.bn_3d(self.dynamic_conv(h_ct_pt)))
        h_ct_pt_fuse2 = self.dynamic_conv_out(h_ct_pt_fuse1)
        h_ct_pt_fuse = torch.squeeze(h_ct_pt_fuse2, dim=2)

        ct_out = ct_fea + h_ct + h_ct_pt_fuse
        pt_out = pt_fea + h_pt + h_ct_pt_fuse
        return ct_out, h_ct_pt_fuse, pt_out


# ################################ FFZNet ######################################################################

class FFZ_Net(nn.Module):
    def __init__(self, classes=2, channels=3):
        super(FFZ_Net, self).__init__()
        
        self.rla_channel = 16
        filters = [16, 32, 64, 128]
        self.model = CFLDE_resnet34_S()  

        self.gffe_m = GFFE_M()

        self.conv_ct_out = nn.Conv2d(16, 1, 1)
        self.conv_pt_out = nn.Conv2d(16, 1, 1)
        self.conv_eh1 = nn.Conv2d(16, 1, 1)
        self.conv_eh2 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)
        self.conv_eh3 = nn.ConvTranspose2d(64, 1, kernel_size=8, stride=4, padding=2)
        self.conv_eh4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(64, 1, kernel_size=8, stride=4, padding=2)
            )
        
        self.drop_block = Drop(drop_rate=0.2, block_size=2)    

        self.decoder4 = CFLDE_DecoderBlock_V2(128, filters[2], rla_channel=128, CA_size=7)
        self.decoder3 = CFLDE_DecoderBlock_V2(filters[2], filters[1], rla_channel=64, CA_size=5)
        self.decoder2 = CFLDE_DecoderBlock_V2(filters[1], filters[0], rla_channel=32, CA_size=5)
        self.decoder1 = CFLDE_DecoderBlock_V2(filters[0], filters[0], rla_channel=16, CA_size=5)

        self.conv_bottleneck_h = nn.Conv2d(128, 1, 1)
        self.conv_dh4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(64, 1, kernel_size=8, stride=4, padding=2)
        )
        
        self.conv_dh3 = nn.ConvTranspose2d(64, 1, kernel_size=8, stride=4, padding=2)
        self.conv_dh2 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)
        self.conv_dh1 = nn.Conv2d(16, 1, 1)
        self.conv_dh0 = nn.Conv2d(16+16+16, 1, 2)
        
        self.dh3_conv = nn.Conv2d(128, 64, 1)
        self.dh3_bn = nn.BatchNorm2d(64)
        self.dh3_silu = nn.SiLU(inplace=True)

        self.dh2_conv = nn.Conv2d(64, 32, 1)
        self.dh2_bn = nn.BatchNorm2d(32)
        self.dh2_silu = nn.SiLU(inplace=True)

        self.dh1_conv = nn.Conv2d(32, 16, 1)
        self.dh1_bn = nn.BatchNorm2d(16)
        self.dh1_silu = nn.SiLU(inplace=True)

        self.dconv_43 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.conv_43 = nn.Conv2d(64+64, 64, 3, 1, 1)
        self.bn_43 = nn.BatchNorm2d(64)
        self.silu_43 = nn.SiLU(inplace=True)

        self.dconv_32 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.conv_32 = nn.Conv2d(32+32, 32, 3, 1, 1)
        self.bn_32 = nn.BatchNorm2d(32)
        self.silu_32 = nn.SiLU(inplace=True)

        self.dconv_21 = nn.ConvTranspose2d(32, 16, 4, 2, 1)
        self.conv_21 = nn.Conv2d(16+16, 16, 3, 1, 1)
        self.bn_21 = nn.BatchNorm2d(16)
        self.silu_21 = nn.SiLU(inplace=True)


        self.finaldeconv1 = nn.ConvTranspose2d(filters[0]+filters[0]+16, 16, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv3 = nn.Conv2d(16, classes, 3, padding=1)

    def forward(self, x):

        ct_x = x[:, 0:3, :, :]
        pt_x = x[:, 3:, :, :]

        x_ct, ct_e1, ct_e2, ct_e3, ct_e4, e_h1, e_h2, e_h3, e_h4, x_pt, pt_e1, pt_e2, pt_e3, pt_e4 = self.model(ct_x, pt_x)

        eh1_out = self.conv_eh1(e_h1)
        eh2_out = self.conv_eh2(e_h2)
        eh3_out = self.conv_eh3(e_h3)
        eh4_out = self.conv_eh4(e_h4)
    
        ct_e4_flat, eh4_flat, pt_e4_flat = self.gffe_m(ct_e4, e_h4, pt_e4)

        dh_4 = eh4_flat

        dh4_out = self.conv_dh4(dh_4)
        
        ct_d3, dh_3, pt_d3 = self.decoder4(ct_e4_flat, dh_4, pt_e4_flat)  
        dh_3 = self.dh3_silu(self.dh3_bn(self.dh3_conv(dh_3)))

        ct_d3 = ct_d3 + ct_e3
        pt_d3 = pt_d3 + pt_e3
        dh_3 = dh_3 + e_h3

        dh_43 = self.dconv_43(dh_4)
        dh_3 = self.silu_43(self.bn_43(self.conv_43(torch.cat((dh_43, dh_3), dim=1))))

        dh3_out = self.conv_dh3(dh_3)

        ct_d2, dh_2, pt_d2 = self.decoder3(ct_d3, dh_3, pt_d3)
        dh_2 = self.dh2_silu(self.dh2_bn(self.dh2_conv(dh_2)))
        
        ct_d2 = ct_d2 + ct_e2
        pt_d2 = pt_d2 + pt_e2
        dh_2 = dh_2 + e_h2

        dh_32 = self.dconv_32(dh_3)
        dh_2 = self.silu_32(self.bn_32(self.conv_32(torch.cat((dh_32, dh_2), dim=1))))

        dh2_out = self.conv_dh2(dh_2)

        ct_d1, dh_1, pt_d1 = self.decoder2(ct_d2, dh_2, pt_d2)
        dh_1 = self.dh1_silu(self.dh1_bn(self.dh1_conv(dh_1)))
        
        ct_d1 = ct_d1 + ct_e1
        pt_d1 = pt_d1 + pt_e1
        dh_1 = dh_1 + e_h1

        dh_21 = self.dconv_21(dh_2)
        dh_1 = self.silu_21(self.bn_21(self.conv_21(torch.cat((dh_21, dh_1), dim=1))))

        dh1_out = self.conv_dh1(dh_1)

        ct_d0, dh_0, pt_d0 = self.decoder1(ct_d1, dh_1, pt_d1)

        d0_out = torch.cat((ct_d0, dh_0, pt_d0), dim=1)
        dh0_out = self.conv_dh0(d0_out)
                
        out = self.finaldeconv1(d0_out)  
        out = self.finalrelu1(out)
        out = self.finalconv3(out)  

        x_ct_out = self.conv_ct_out(x_ct)
        x_pt_out = self.conv_pt_out(x_pt)

        return F.sigmoid(out), x_ct_out, x_pt_out, [eh1_out, eh2_out, eh3_out, eh4_out], [dh4_out, dh3_out, dh2_out, dh1_out, dh0_out]



if __name__ == '__main__':
    input = torch.rand(2, 3, 224, 224)
    model = FFZ_Net()
    out12 = model(input)
    print(out12.shape)