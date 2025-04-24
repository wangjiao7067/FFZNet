import torch.nn.functional as F
import math
import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


__all__ = ["rla_resnet34_eca_eh", "rla_resnet34_eca_eh_final", "rla_resnet34_eca_eh_tiny_final"]


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



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

    

#=========================== define bottleneck ============================

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



class CFLDE_BasicBlock_half(nn.Module):  
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, 
                 rla_channel=16, CA_size=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, reduction=16, pt=False):
        super(CFLDE_BasicBlock_half, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.inplanes = inplanes
        self.rla_channel = rla_channel
        self.planes = planes

        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv_pre = conv1x1(inplanes, planes)

        if not pt:
            self.conv1 = conv3x3(inplanes*2, planes, stride, groups=inplanes)
            self.bn1 = norm_layer(planes)
            self.silu = nn.SiLU(inplace=True)
            self.conv2 = conv3x3(planes, planes, groups=planes)
            self.bn2 = norm_layer(planes)
            self.downsample = downsample
            self.stride = stride
        elif pt:
            self.conv1 = conv3x3(inplanes*2, planes, stride, groups=inplanes)
            self.bn1 = norm_layer(planes)
            self.silu = nn.SiLU()
            self.conv2 = PConv(planes, planes, 3, 1)
            self.bn2 = norm_layer(planes)
            self.downsample = downsample
            self.stride = stride            
            
        self.averagePooling = None
        if downsample is not None and stride != 1:
            self.averagePooling = nn.AvgPool2d((2, 2), stride=(2, 2))
     
        self.ca = None
        if CA_size != None:
            self.ca = ca_layer(planes * self.expansion, int(CA_size))
            
    def forward(self, x, h):
        identity = x
        
        if h.shape[1] == x.shape[1]:
            pass
        else:
            h = self.conv_pre(h)

        x_new = torch.cat((x, h), dim=1)
        for index in range(x_new.shape[1]):
            if index % 2 == 0:
                x_new[:, index, :, :] = x[:, index//2, :, :]
            else:
                x_new[:, index, :, :] = h[:, index//2, :, :] 

        out = self.conv1(x_new) 
        out = self.bn1(out)  
        out = self.silu(out)

        out = self.conv2(out)  
        out = self.bn2(out)
            
        if self.ca != None:
            out = self.ca(out)
        
        y = out
        
        if self.downsample is not None:
            identity = self.downsample(identity)
        if self.averagePooling is not None:
            h = self.averagePooling(h)

        out += identity

        out = self.silu(out)

        return out, y, h



#=========================== define network ============================


class CFLDE_ResNet_S(nn.Module):
    def __init__(self, block, layers, num_classes=1000, 
                 rla_channel=16, CA=None, 
                 zero_init_last_bn=True,
                 groups=1, width_per_group=32, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(CFLDE_ResNet_S, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 16
        self.dilation = 1
        self.rla_channel = rla_channel
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        
        if CA is None:
            CA = [None] * 4
        elif len(CA) != 4:
            raise ValueError("argument CA should be a 4-element tuple, got {}".format(CA))
        
        self.flops = False
        self.groups = groups
        self.base_width = width_per_group

        self.conv1_ct = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_ct = norm_layer(self.inplanes)
        self.silu_ct = nn.SiLU(inplace=True)
        self.maxpool_ct = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv1_pt = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_pt = norm_layer(self.inplanes)
        self.silu_pt = nn.SiLU(inplace=True)
        self.maxpool_pt = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        conv_outs_ct = [None] * 4
        conv_hs_ct = [None] * 4
        stages_ct = [None] * 4
        stage_bns_ct = [None] * 4

        conv_outs_pt = [None] * 4
        conv_hs_pt = [None] * 4
        stages_pt = [None] * 4
        stage_bns_pt = [None] * 4
        
        stages_ct[0], stage_bns_ct[0], conv_outs_ct[0], conv_hs_ct[0] = self._make_layer(block, 16, layers[0], 
                                                                                     rla_channel=16, CA_size=CA[0], pt=False)
        stages_ct[1], stage_bns_ct[1], conv_outs_ct[1], conv_hs_ct[1] = self._make_layer(block, 32, layers[1], 
                                                                                     rla_channel=32, CA_size=CA[1], 
                                                                                     stride=2, dilate=replace_stride_with_dilation[0], pt=False)
        stages_ct[2], stage_bns_ct[2], conv_outs_ct[2], conv_hs_ct[2] = self._make_layer(block, 64, layers[2], 
                                                                                     rla_channel=64, CA_size=CA[2], 
                                                                                     stride=2, dilate=replace_stride_with_dilation[1], pt=False)
        stages_ct[3], stage_bns_ct[3], conv_outs_ct[3], conv_hs_ct[3] = self._make_layer(block, 128, layers[3], 
                                                                                     rla_channel=128, CA_size=CA[3], 
                                                                                     stride=2, dilate=replace_stride_with_dilation[2], pt=False)
        
        self.inplanes = 16
        
        stages_pt[0], stage_bns_pt[0], conv_outs_pt[0], conv_hs_pt[0] = self._make_layer(block, 16, layers[0], 
                                                                                     rla_channel=16, CA_size=CA[0], pt=True)
        stages_pt[1], stage_bns_pt[1], conv_outs_pt[1], conv_hs_pt[1] = self._make_layer(block, 32, layers[1], 
                                                                                 rla_channel=32, CA_size=CA[1], 
                                                                                     stride=2, dilate=replace_stride_with_dilation[0], pt=True)
        stages_pt[2], stage_bns_pt[2], conv_outs_pt[2], conv_hs_pt[2] = self._make_layer(block, 64, layers[2], 
                                                                                     rla_channel=64, CA_size=CA[2], 
                                                                                     stride=2, dilate=replace_stride_with_dilation[1], pt=True)
        stages_pt[3], stage_bns_pt[3], conv_outs_pt[3], conv_hs_pt[3] = self._make_layer(block, 128, layers[3], 
                                                                                     rla_channel=128, CA_size=CA[3], 
                                                                                     stride=2, dilate=replace_stride_with_dilation[2], pt=True)

        self.conv_outs_ct = nn.ModuleList(conv_outs_ct)
        self.conv_hs_ct = nn.ModuleList(conv_hs_ct)
        self.stages_ct = nn.ModuleList(stages_ct)
        self.stage_bns_ct = nn.ModuleList(stage_bns_ct)

        self.conv_outs_pt = nn.ModuleList(conv_outs_pt)
        self.conv_hs_pt = nn.ModuleList(conv_hs_pt)
        self.stages_pt = nn.ModuleList(stages_pt)
        self.stage_bns_pt = nn.ModuleList(stage_bns_pt)

        self.conv_h_1 = nn.Conv2d(16+16, 16, 3, 1, 1)
        self.bn_h1 = norm_layer(16)
        self.silu_h1 = nn.SiLU(inplace=True)
        self.conv_h_12 = nn.Conv2d(16, 32, 3, 1, 1)
        self.bn_h_12 = norm_layer(32)
        self.silu_h_12 = nn.SiLU(inplace=True)


        self.conv_h_2 = nn.Conv2d(32+32+32, 32, 3, 1, 1)
        self.bn_h2 = norm_layer(32)
        self.silu_h2 = nn.SiLU(inplace=True)

        self.conv_12 = nn.Conv2d(16, 32, 3, 2, 1)
        self.bn_12 = norm_layer(32)
        self.silu_12 = nn.SiLU(inplace=True)


        self.conv_h_3 = nn.Conv2d(64+64+64, 64, 3, 1, 1)
        self.bn_h3 = norm_layer(64)
        self.silu_h3 = nn.SiLU(inplace=True) 

        self.conv_23 = nn.Conv2d(32, 64, 3, 2, 1)
        self.bn_23 = norm_layer(64)
        self.silu_23 = nn.SiLU(inplace=True)


        self.conv_h_4 = nn.Conv2d(128+128+128, 128, 3, 1, 1)
        self.bn_h4 = norm_layer(128)
        self.silu_h4 = nn.SiLU(inplace=True)

        self.conv_34 = nn.Conv2d(64, 128, 3, 2, 1)
        self.bn_34 = norm_layer(128)
        self.silu_34 = nn.SiLU(inplace=True)
        
        self.tanh = nn.Tanh()
        
        self.bn2 = norm_layer(rla_channel)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(int(128 * block.expansion) + rla_channel, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, planes, blocks, 
                    rla_channel, CA_size, stride=1, dilate=False, pt=False):
        
        conv_out = conv1x1(int(planes * block.expansion), rla_channel)
        conv_h = conv1x1(int(planes / 2), planes)

        
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        

        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != int(planes * block.expansion):
            downsample = nn.Sequential(
                conv1x1(self.inplanes, int(planes * block.expansion), stride),
                norm_layer(int(planes * block.expansion)),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, 
                            rla_channel=rla_channel, CA_size=CA_size, groups=self.groups,
                            base_width=self.base_width, dilation=previous_dilation, norm_layer=norm_layer, pt=pt))
        self.inplanes = int(planes * block.expansion)
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, 
                                rla_channel=rla_channel, CA_size=CA_size, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, pt=pt))

        bns = [norm_layer(rla_channel) for _ in range(blocks)]

        return nn.ModuleList(layers), nn.ModuleList(bns), conv_out, conv_h
                                                                     
    
    def _get_one_layer(self, layers, bns, conv_out, conv_h, x, h):
        for layer, bn in zip(layers, bns):
            x, y, h = layer(x, h)
            
            y_out = conv_out(y)
            if h.shape[1] != y_out.shape[1]:
                h = conv_h(h)
            
            h = h + y_out
            h = bn(h)
            h = self.tanh(h)
    
        return x, h
        

    def _forward_impl(self, x_ct, x_pt):

        x_ct = self.conv1_ct(x_ct)
        x_ct = self.bn1_ct(x_ct)       
        x_ct = self.silu_ct(x_ct)
        x_ct = self.maxpool_ct(x_ct) 
        

        x_pt = self.conv1_pt(x_pt)
        x_pt = self.bn1_pt(x_pt)
        x_pt = self.silu_pt(x_pt)
        x_pt = self.maxpool_pt(x_pt)
        


        batch, _, height, width = x_ct.size()   

        # 初始融合特征
        h = torch.zeros(batch, 16, height, width, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # layer_0
        layer_0_ct = self.stages_ct[0]
        bns_0_ct = self.stage_bns_ct[0]
        conv_out_0_ct = self.conv_outs_ct[0]
        conv_h_0_ct = self.conv_hs_ct[0]

        layer_0_pt = self.stages_pt[0]
        bns_0_pt = self.stage_bns_pt[0]
        conv_out_0_pt = self.conv_outs_pt[0]
        conv_h_0_pt = self.conv_hs_pt[0]

        x_1_ct, h_1_ct = self._get_one_layer(layer_0_ct, bns_0_ct, conv_out_0_ct, conv_h_0_ct, x_ct, h)    # # output x_1  # [8, 256, 56, 56], [8, 32, 56, 56]
        x_1_pt, h_1_pt = self._get_one_layer(layer_0_pt, bns_0_pt, conv_out_0_pt, conv_h_0_pt, x_pt, h)

        h_1 = self.silu_h1(self.bn_h1(self.conv_h_1(torch.cat((h_1_ct, h_1_pt), dim=1))))
        

        # layer_1
        layer_1_ct = self.stages_ct[1]
        bns_1_ct = self.stage_bns_ct[1]
        conv_out_1_ct = self.conv_outs_ct[1]
        conv_h_1_ct = self.conv_hs_ct[1]

        layer_1_pt = self.stages_pt[1]
        bns_1_pt = self.stage_bns_pt[1]
        conv_out_1_pt = self.conv_outs_pt[1]
        conv_h_1_pt = self.conv_hs_pt[1]

        x_2_ct, h_2_ct = self._get_one_layer(layer_1_ct, bns_1_ct, conv_out_1_ct, conv_h_1_ct, x_1_ct, h_1)    # output x_2 [8, 512, 28, 28], [8, 32, 28, 28]
        x_2_pt, h_2_pt = self._get_one_layer(layer_1_pt, bns_1_pt, conv_out_1_pt, conv_h_1_pt, x_1_pt, h_1)
        

        h_12 = self.silu_12(self.bn_12(self.conv_12(h_1)))
        h_2 = self.silu_h2(self.bn_h2(self.conv_h_2(torch.cat((h_2_ct, h_2_pt, h_12), dim=1))))
        

        # layer_2
        layer_2_ct = self.stages_ct[2]
        bns_2_ct = self.stage_bns_ct[2]
        conv_out_2_ct = self.conv_outs_ct[2]
        conv_h_2_ct = self.conv_hs_ct[2]

        layer_2_pt = self.stages_pt[2]
        bns_2_pt = self.stage_bns_pt[2]
        conv_out_2_pt = self.conv_outs_pt[2]
        conv_h_2_pt = self.conv_hs_pt[2]
        
        x_3_ct, h_3_ct = self._get_one_layer(layer_2_ct, bns_2_ct, conv_out_2_ct, conv_h_2_ct, x_2_ct, h_2)    # output x_3 [8, 1024, 14, 14], [8, 32, 14, 14]
        x_3_pt, h_3_pt = self._get_one_layer(layer_2_pt, bns_2_pt, conv_out_2_pt, conv_h_2_pt, x_2_pt, h_2)
   
        h_23 = self.silu_23(self.bn_23(self.conv_23(h_2)))
        h_3 = self.silu_h3(self.bn_h3(self.conv_h_3(torch.cat((h_3_ct, h_3_pt, h_23), dim=1))))
       

        # layer_3
        layer_3_ct = self.stages_ct[3]
        bns_3_ct = self.stage_bns_ct[3]
        conv_out_3_ct = self.conv_outs_ct[3]
        conv_h_3_ct = self.conv_hs_ct[3]

        layer_3_pt = self.stages_pt[3]
        bns_3_pt = self.stage_bns_pt[3]
        conv_out_3_pt = self.conv_outs_pt[3]
        covn_h_3_pt = self.conv_hs_pt[3]
        
        x_4_ct, h_4_ct = self._get_one_layer(layer_3_ct, bns_3_ct, conv_out_3_ct, conv_h_3_ct, x_3_ct, h_3)    # output x_4 [8, 2048, 7, 7], [8, 32, 7, 7]
        x_4_pt, h_4_pt = self._get_one_layer(layer_3_pt, bns_3_pt, conv_out_3_pt, covn_h_3_pt, x_3_pt, h_3)

        h_34 = self.silu_34(self.bn_34(self.conv_34(h_3)))
        h_4 = self.silu_h4(self.bn_h4(self.conv_h_4(torch.cat((h_4_ct, h_4_pt, h_34), dim=1))))
        
        return x_ct, x_1_ct, x_2_ct, x_3_ct, x_4_ct, h_1, h_2, h_3, h_4, x_pt, x_1_pt, x_2_pt, x_3_pt, x_4_pt

       
    def forward(self, x_ct, x_pt):
        return self._forward_impl(x_ct, x_pt)



#=========================== available models ============================


def CFLDE_resnet34_S(rla_channel=16, k_size=[5, 5, 5, 7]): 
    print("Constructing MFII_resnet34_S......")
    model = CFLDE_ResNet_S(CFLDE_BasicBlock_half, [3, 4, 6, 3], CA=k_size)
    
    return model

