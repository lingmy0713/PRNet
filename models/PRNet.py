import torch
import torch.nn as nn
from models import model_base
import torch.nn.functional as F
from einops import rearrange
import numbers
from argument.argument import args


def pixel_shuffle(scale):
    return nn.PixelShuffle(upscale_factor=scale)


def rgb2raw(img):
    raw = torch.zeros([img.shape[0], 4, img.shape[2] // 2, img.shape[3] // 2], dtype=torch.float).cuda()
    img = img.squeeze(1)
    raw[:, 0, :, :] = img[:, 0, ::2, ::2]
    raw[:, 1, :, :] = img[:, 1, ::2, 1::2]
    raw[:, 2, :, :] = img[:, 1, 1::2, ::2]
    raw[:, 3, :, :] = img[:, 2, 1::2, 1::2]
    return raw


class RCAB(nn.Module):
    def __init__(self, in_channel, act=nn.LeakyReLU(negative_slope=0.2),):
        super(RCAB, self).__init__()
        self.body = nn.Sequential(*[
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, bias=True),
            act,
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, bias=True),
            CALayer(in_channel, True)
        ])

    def forward(self, x):
        out = self.body(x)
        out += x
        return out


class RG(nn.Module):
    def __init__(self, num_RCAB, inchannel):
        super(RG, self).__init__()
        body = []
        for i in range(num_RCAB):
            body.append(RCAB(in_channel=inchannel))
        body.append(nn.Conv2d(inchannel, inchannel, kernel_size=3, padding=(3 - 1) // 2, stride=1))
        self.body = nn.Sequential(*body)

    def forward(self, x):
        out = self.body(x)
        out += x
        return out


# Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, bias):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, 4, 1, padding=0, bias=bias),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(4, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class Lap_Pyramid_Conv(nn.Module):
    def __init__(self, num_high=3):
        super(Lap_Pyramid_Conv, self).__init__()

        self.num_high = num_high
        self.kernel = self.gauss_kernel()

    def gauss_kernel(self, device=torch.device('cuda'), channels=3):
        kernel = torch.tensor([[1., 4., 6., 4., 1],
                               [4., 16., 24., 16., 4.],
                               [6., 24., 36., 24., 6.],
                               [4., 16., 24., 16., 4.],
                               [1., 4., 6., 4., 1.]])
        kernel /= 256.
        kernel = kernel.repeat(channels, 1, 1, 1)
        kernel = kernel.to(device)
        return kernel

    def downsample(self, x):
        return x[:, :, ::2, ::2]

    def upsample(self, x):
        cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
        cc = cc.permute(0, 1, 3, 2)
        cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
        x_up = cc.permute(0, 1, 3, 2)
        return self.conv_gauss(x_up, 4 * self.kernel)

    def conv_gauss(self, img, kernel):
        img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
        out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
        return out

    def pyramid_decom(self, img):
        current = img
        pyr = []
        for _ in range(self.num_high):
            filtered = self.conv_gauss(current, self.kernel)
            down = self.downsample(filtered)
            up = self.upsample(down)
            if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
                up = nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]))
            diff = current - up
            pyr.append(diff)
            current = down
        pyr.append(current)
        return pyr

    def pyramid_recons(self, pyr):
        image = pyr[-1]
        for level in reversed(pyr[:-1]):
            up = self.upsample(image)
            if up.shape[2] != level.shape[2] or up.shape[3] != level.shape[3]:
                up = nn.functional.interpolate(up, size=(level.shape[2], level.shape[3]))
            image = up + level
        return image


class XrefCondNet(nn.Module):
    def __init__(self, num_RG):
        super(XrefCondNet, self).__init__()

        self.head = nn.Sequential(*[
            nn.Conv2d(3, 64, kernel_size=3, padding=(3 - 1) // 2, bias=True),
        ])

        modules_body = []
        for i in range(num_RG):
            modules_body.append(RG(num_RCAB=4, inchannel=64))
        modules_body.append(nn.Conv2d(64, 64, kernel_size=3, padding=(3 - 1) // 2, stride=1))
        self.body = nn.Sequential(*modules_body)

        if args.scale == 2:
            modules_tail = [InvertedResidualBlock(in_channels=64, out_channels=64, stride=2, expand_ratio=6),
                            nn.Sigmoid()]
        elif args.scale == 4:
            modules_tail = [nn.Conv2d(64, 64, kernel_size=3, padding=(3 - 1) // 2, stride=1),
                            nn.Sigmoid()]

        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        head = self.head(x)
        body = head + self.body(head)
        return self.tail(body)


class CGM(nn.Module):
    def __init__(self, in_channel, num):
        super(CGM, self).__init__()
        self.num_CGB = num
        sft_branch = []
        for i in range(self.num_CGB):
            sft_branch.append(TransformerBlock(dim=in_channel, num_heads=4, bias=False, LayerNorm_type='WithBias'))
        self.body = nn.Sequential(*sft_branch)
        self.fuse = nn.Conv2d(in_channel * (self.num_CGB+1), in_channel, kernel_size=1, padding=0, stride=1)

    def forward(self, fea, cond):
        out = [fea]
        for i in range(self.num_CGB):
            fea = self.body[i](fea, cond)
            out.append(fea)
        fea = self.fuse(torch.cat(out, dim=1))
        return fea


class Trans(nn.Module):
    def __init__(self, dim, num_CGM):
        super(Trans, self).__init__()

        self.CGM = CGM(in_channel=dim, num=num_CGM)

    def forward(self, fea, cond):
        fea = self.CGM(fea, cond)
        return fea


###############################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm_IN = nn.InstanceNorm2d(dim, affine=False)
        self.norm_c1 = LayerNorm(dim, LayerNorm_type)
        self.norm_s1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Restormer_cross(dim, num_heads, bias)
        self.norm_c2 = LayerNorm(dim, LayerNorm_type)
        self.FFN = FeedForward(dim, 2.66, bias)

    def forward(self, content, style):
        res = content
        scale, shift = self.attn(self.norm_c1(content), self.norm_s1(style))
        content = self.norm_IN(content) * scale + shift
        content = res + content
        content = content + self.FFN(self.norm_c2(content))
        return content


## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Restormer_cross(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Restormer_cross, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim * 1, bias=bias)

        self.project_scale = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.project_shift = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, content, style):
        b, c, h, w = content.shape

        kv = self.kv_dwconv(self.kv(style))
        k, v_scale, v_shift = kv.chunk(3, dim=1)
        q = self.q_dwconv(self.q(content))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_scale = rearrange(v_scale, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_shift = rearrange(v_shift, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        attn = attn.softmax(dim=-1)

        scale = (attn @ v_scale)
        shift = (attn @ v_shift)

        scale = rearrange(scale, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        shift = rearrange(shift, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        scale = self.project_scale(scale)
        shift = self.project_shift(shift)

        return scale, shift

## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

##########################################################################


class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, stride=1, expand_ratio=6, activation=nn.LeakyReLU(negative_slope=0.2, inplace=True)):
        super(InvertedResidualBlock, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        assert stride in [1, 2]
        hidden_dim = int(in_channels * expand_ratio)
        self.is_residual = self.stride == 1 and in_channels == out_channels
        self.conv = nn.Sequential(
            # pw Point-wise
            nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
            activation,
            # dw  Depth-wise
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            activation,
            # pw-linear, Point-wise linear
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),

        )

    def forward(self, x):
        if self.stride == 1 and self.in_channels == self.out_channels:
            res = self.conv(x)
            x = x + res
        else:
            x = self.conv(x)
        return x


class Restoration(model_base.ModelBase):
    def __init__(self, args):
        super(Restoration, self).__init__(args)

        r = args.scale

        # IR sub-network
        self.SFENet1 = nn.Conv2d(4, 64, 3, padding=(3 - 1) // 2, stride=1)

        self.RDBs = nn.ModuleList()
        for i in range(2):
            self.RDBs.append(
                RG(num_RCAB=4, inchannel=64)
            )

        self.GFF = nn.Sequential(*[
            nn.Conv2d(3 * 64, 64, 1, padding=0, stride=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 64, 3, padding=(3-1)//2, stride=1)
        ])

        # Up-sampling net
        self.output = nn.Sequential(*[
            nn.Conv2d(64, 3 * (2*r) * (2*r), kernel_size=3, padding=1, bias=True),
            pixel_shuffle(2*r)
            ])

        # ACG generator
        self.XrefCondNet = XrefCondNet(num_RG=1)

        # PR sub-network
        self.lap_pyramid = Lap_Pyramid_Conv(2)

        self.pyr_conv_3 = nn.Conv2d(3, 64, kernel_size=3, padding=(3 - 1) // 2, stride=1)
        self.trans_low = Trans(dim=64, num_CGM=4).cuda()
        self.deconv_3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
        )

        self.pyr_conv_2 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=(3 - 1) // 2, stride=1),
            nn.LeakyReLU(0.2, inplace=True),
            InvertedResidualBlock(in_channels=64, out_channels=64, stride=2, expand_ratio=6)
        )
        self.CGM_2 = Trans(dim=64, num_CGM=2).cuda()
        self.up_2 = nn.ConvTranspose2d(64, 64, kernel_size=4, padding=1, stride=2, bias=True)
        self.deconv_2 = nn.Sequential(*[
            nn.Conv2d(64, 32, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
        ])

        self.pyr_conv_1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=(3 - 1) // 2, stride=1),
            nn.LeakyReLU(0.2, inplace=True),
            InvertedResidualBlock(in_channels=32, out_channels=32, stride=2, expand_ratio=6)
        )
        self.CGM_1 = Trans(dim=32, num_CGM=2).cuda()
        self.up_1 = nn.ConvTranspose2d(32, 32, kernel_size=4, padding=1, stride=2, bias=True)

        self.fake_conv_3 = nn.Sequential(*[
            nn.Conv2d(64, 64, kernel_size=3, padding=(3 - 1) // 2, stride=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=(3 - 1) // 2, stride=1),
            ])
        self.fake_conv_2 = nn.Sequential(*[
            nn.Conv2d(64, 64, kernel_size=3, padding=(3 - 1) // 2, stride=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=(3 - 1) // 2, stride=1),
        ])
        self.fake_conv_1 = nn.Sequential(*[
            nn.Conv2d(32, 32, kernel_size=3, padding=(3 - 1) // 2, stride=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, padding=(3 - 1) // 2, stride=1),
        ])

    def forward(self, x, isp):
        # IR sub-network
        raw = rgb2raw(x)
        f__1 = self.SFENet1(raw)
        x = f__1
        RDBs_out = [x]
        for i in range(2):
            x = self.RDBs[i](x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out, 1))
        x += f__1
        x = self.output(x)

        # ACG Generator
        cond_3 = self.XrefCondNet(isp)

        # PR sub-network
        input_pyr_trans = []
        input_pyr = self.lap_pyramid.pyramid_decom(img=x)

        fea_3 = self.trans_low(self.pyr_conv_3(input_pyr[-1]), cond_3)
        fake_3 = self.fake_conv_3(fea_3) + input_pyr[-1]
        cond_2 = self.deconv_3(fea_3)
        del cond_3
        del fea_3

        fea_2 = self.up_2(self.CGM_2(self.pyr_conv_2(input_pyr[-2]), cond_2))
        fake_2 = self.fake_conv_2(fea_2) + input_pyr[-2]
        cond_1 = self.deconv_2(fea_2)
        del cond_2
        del fea_2

        fea_1 = self.up_1(self.CGM_1(self.pyr_conv_1(input_pyr[-3]), cond_1))
        fake_1 = self.fake_conv_1(fea_1) + input_pyr[-3]
        del cond_1
        del fea_1

        input_pyr_trans.append(fake_1)
        input_pyr_trans.append(fake_2)
        input_pyr_trans.append(fake_3)

        full = self.lap_pyramid.pyramid_recons(input_pyr_trans)

        return full, fake_1, fake_2, fake_3, full
