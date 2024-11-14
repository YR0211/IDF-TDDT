import torch
import torch.nn as nn
import torch.nn.functional as F


class TDPI_block(nn.Module):
    def __init__(self, embed_dim=512, inC=8, kernel_size=3):
        super(TDPI_block, self).__init__()
        self.kernel_size = kernel_size
        self.embed_dim = embed_dim
        self.inC = inC
        self.cppb = nn.Sequential(
            nn.Linear(512 + 2 * inC, 512),
            nn.Linear(512, self.kernel_size * self.kernel_size * self.inC * self.inC)
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.gap_conv = nn.Conv2d(inC, inC, kernel_size=3, stride=1, bias=True, padding=1)
        self.gmp_conv = nn.Conv2d(inC, inC, kernel_size=3, stride=1, bias=True, padding=1)
        self.zc = self.zero_convolution(inC, inC)
        self.bn_lrelu = nn.Sequential(
            nn.BatchNorm2d(inC),
            nn.LeakyReLU(0.2)
        )
        self.conv = nn.Sequential(nn.Conv2d(inC, inC * 2, kernel_size=3, stride=1, bias=True, padding=1),
                                  nn.BatchNorm2d(inC * 2),
                                  nn.LeakyReLU(0.2),
                                  nn.Conv2d(inC * 2, inC * 2, kernel_size=3, stride=1, bias=True, padding=1),
                                  nn.BatchNorm2d(inC * 2),
                                  nn.LeakyReLU(0.2),
                                  nn.Conv2d(inC * 2, inC, kernel_size=3, stride=1, bias=True, padding=1),
                                  nn.BatchNorm2d(inC),
                                  nn.LeakyReLU(0.2),
                                  )

    def forward(self, imgf, text_a):
        img_gap = self.gap(self.gap_conv(imgf)).squeeze(dim=-1).squeeze(dim=-1)
        img_gmp = self.gmp(self.gmp_conv(imgf)).squeeze(dim=-1).squeeze(dim=-1)
        img_text = torch.cat((img_gap, img_gmp, text_a), dim=1)
        conv_weight = self.cppb(img_text)
        bs, c, h, w = imgf.size()
        conv_weight1 = conv_weight.view(bs * self.inC, self.inC, self.kernel_size, self.kernel_size)
        dp = F.conv2d(imgf.view(1, bs * c, h, w), conv_weight1, stride=1, padding=self.kernel_size // 2, groups=bs)
        dp = dp.view(bs, c, h, w)
        dp = self.bn_lrelu(dp)
        dp = self.conv(dp)
        dp = self.zc(dp)
        imgf_h = dp + imgf
        return imgf_h

    def zero_convolution(self, inC, outC, kernel_size=1, stride=1):
        zc = nn.Conv2d(inC, outC, kernel_size=kernel_size, stride=stride, bias=True)
        nn.init.zeros_(zc.weight)
        nn.init.zeros_(zc.bias)
        return zc


class TDPI(nn.Module):
    def __init__(self, adapter, num_blocks=3, embed_dim=512, inC=8):
        super(TDPI, self).__init__()
        self.num_blocks = num_blocks

        self.adapter = adapter

        for i in range(num_blocks):
            setattr(self, f"dpi{i}", TDPI_block(embed_dim=embed_dim, inC=inC))

    def forward_obo(self, img_f, text_f, block_index):
        assert block_index <= self.num_blocks - 1 and block_index >= 0
        block = getattr(self, f"dpi{block_index}")
        text_a = self.adapter(text_f)
        out = block(img_f, text_a)
        return out


class T_OAR(nn.Module):
    def __init__(self, num_blocks=3, embed_dim=512, inC=8):
        super(T_OAR, self).__init__()
        self.num_blocks = num_blocks

        self.adapter = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.Linear(512, 512)
        )

        self.ir_t_dpi = TDPI(adapter=self.adapter, num_blocks=num_blocks, embed_dim=embed_dim, inC=inC)
        self.vi_t_dpi = TDPI(adapter=self.adapter, num_blocks=num_blocks, embed_dim=embed_dim, inC=inC)