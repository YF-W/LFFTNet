import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torchvision import models as resnet_model

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
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
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.layers.append(nn.ModuleList([
            PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
            PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
        ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class encoder(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3,
                 dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        self.channelTrans = nn.Conv2d(in_channels=65, out_channels=512, kernel_size=1, padding=0)

    def forward(self, x):
        x_vit = x
        x_vit = self.to_patch_embedding(x_vit)
        b, n, _ = x_vit.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x_vit = torch.cat((cls_tokens, x_vit), dim=1)
        x_vit = self.pos_embedding[:, :(n + 1)] + x_vit
        x_vit = self.dropout(x_vit)

        vit_layerInfo = []
        for i in range(4):
            x_vit = self.transformer(x_vit)
            vit_layerInfo.append(x_vit)

        return vit_layerInfo

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        return self.conv(x)

class Bottleneck(nn.Module):
    def __init__(self):
        super(Bottleneck, self).__init__()
        self.preconv = DoubleConv(1024, 1024)
        self.p1_1 = nn.ConvTranspose2d(1024, 1024, kernel_size=2, stride=2)
        self.p2_1 = nn.Conv2d(1024, 1024, 3, 1, 1, bias=False)
        self.p3_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.p1_2 = nn.Conv2d(1024, 1024, 3, 1, 1, bias=False)
        self.p2_2 = nn.Conv2d(1024, 1024, 5, 1, 2, bias=False)
        self.p3_2 = nn.Conv2d(1024, 1024, 7, 1, 3, bias=False)
        self.Up2Down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Down2Up = nn.ConvTranspose2d(1024, 1024, kernel_size=2, stride=2)
        self.p1_3 = nn.Conv2d(1024*2, 1024, 3, 1, 1, bias=False)
        self.p2_3 = nn.Conv2d(1024*2, 1024, 5, 1, 2, bias=False)
        self.p3_3 = nn.Conv2d(1024*2, 1024, 7, 1, 3, bias=False)
        self.p3_4 = nn.ConvTranspose2d(1024, 1024, kernel_size=2, stride=2)
        self.final = nn.Conv2d(1024, 1024, 3, padding=1, bias=False)
        self.bnrelu = nn.Sequential(
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=False))

    def forward(self, x):
        x = self.preconv(x)
        x = self.bnrelu(x)
        x_1 = x
        p1 = self.p1_1(x)
        p2 = self.p2_1(x)
        p2 = self.bnrelu(p2)
        p3 = self.p3_1(x)
        p1_2 = self.p1_2(p1)
        p1_2 = self.bnrelu(p1_2)
        p2_2 = self.p2_2(p2)
        p2_2 = self.bnrelu(p2_2)
        p3_2 = self.p3_2(p3)
        p3_2 = self.bnrelu(p3_2)
        p1To2 = self.Up2Down(p1_2)
        p2To3 = self.Up2Down(p2_2)
        p3To1 = self.Down2Up(p3_2)
        p3To1 = self.Down2Up(p3To1)
        p1_2 = torch.cat([p1_2, p3To1],dim=1)
        p2_2 = torch.cat([p2_2, p1To2],dim=1)
        p3_2 = torch.cat([p3_2, p2To3],dim=1)
        p1_3 = self.p1_3(p1_2)
        p1_3 = self.bnrelu(p1_3)
        p2_3 = self.p2_3(p2_2)
        p2_3 = self.bnrelu(p2_3)
        p3_3 = self.p3_3(p3_2)
        p3_3 = self.bnrelu(p3_3)
        p1 = self.p3_1(p1_3)
        p3 = self.p3_4(p3_3)
        p2 = p2_3
        p = p1 + p2 + p3
        p = self.final(p)
        p = self.bnrelu(p)
        return p

class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SingleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.conv(x)

class LFFTNet_SN6(nn.Module):
    def __init__(self, features=[64, 128, 256, 512], out_channels=1):
        super(LFFTNet_SN6, self).__init__()
        resnet = resnet_model.resnet34(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.encoder = encoder(image_size=224, patch_size=28, num_classes=2, dim=196, depth=6, heads=16, mlp_dim=2048)
        self.conv = nn.Conv2d(1024, 512, 1)
        self.bottleneck = Bottleneck()

        self.scm = nn.ModuleList()
        self.scm_11_module = nn.ModuleList()

        self.upconv_1 = DoubleConv(in_channels=512*2+ 65, out_channels=512)
        self.upconv_2 = DoubleConv(in_channels=256*2+ 65, out_channels=256)
        self.upconv_3 = DoubleConv(in_channels=128*2+ 65, out_channels=128)
        self.upconv_4 = DoubleConv(in_channels=64*2+ 65, out_channels=64)

        self.ups = nn.ModuleList()
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2
                )
            )
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature, feature, kernel_size=2, stride=2
                )
            )
        self.finalconv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        self.vitLayer_UpConv = nn.ConvTranspose2d(65, 65, kernel_size=2, stride=2)

        self.final_conv1 = nn.ConvTranspose2d(64, 32, 1)
        self.final_relu1 = nn.ReLU(inplace=False)
        self.final_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.final_relu2 = nn.ReLU(inplace=False)
        self.final_conv3 = nn.Conv2d(32, 1, 3, padding=1)

        self.downs = nn.ModuleList()
        self.predowns = nn.ModuleList()

        for feature in features:
            self.downs.append(DoubleConv(feature, feature))

        in_channels = 3
        for feature in features:
            self.predowns.append(SingleConv(in_channels, feature))
            in_channels = feature

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self, x):
        x_original = x

        branch_conv = x
        branch_sc = []
        count = 0
        for down in self.downs:
            branch_conv = self.predowns[count](branch_conv)
            count = count + 1
            residual_branchconv = branch_conv
            branch_conv = down(branch_conv)
            branch_conv = residual_branchconv + branch_conv
            branch_conv = self.pool(branch_conv)
            branch_sc.append(branch_conv)

        e0 = self.firstconv(x)
        e0 = self.firstbn(e0)
        e0 = self.firstrelu(e0)

        e1 = self.encoder1(e0)
        e1 = torch.add(e1, branch_sc[0])
        e2 = self.encoder2(e1)
        e2 = torch.add(e2, branch_sc[1])
        e3 = self.encoder3(e2)
        e3 = torch.add(e3, branch_sc[2])
        e4 = self.encoder4(e3)
        e4 = torch.add(e4, branch_sc[3])

        vit_layerInfo = self.encoder(x_original)

        bottleneck_in = torch.cat([e4, branch_sc[3]], dim=1)

        y = self.bottleneck(bottleneck_in)

        vit_layerInfo = vit_layerInfo[::-1]

        v0 = vit_layerInfo[0].view(4, 65, 14, 14)
        v1 = vit_layerInfo[1].view(4, 65, 14, 14)
        v2 = vit_layerInfo[2].view(4, 65, 14, 14)
        v3 = vit_layerInfo[3].view(4, 65, 14, 14)

        y = self.ups[0](y)
        e4 = self.ups[4](e4)
        v3 = self.vitLayer_UpConv(v3)
        y = torch.cat([y, e4, v3], dim=1)
        y = self.upconv_1(y)

        y = self.ups[1](y)
        e3 = self.ups[5](e3)
        v2 = self.vitLayer_UpConv(v2)
        v2 = self.vitLayer_UpConv(v2)
        y = torch.cat([y, e3, v2], dim=1)
        y = self.upconv_2(y)

        y = self.ups[2](y)
        e2 = self.ups[6](e2)
        v1 = self.vitLayer_UpConv(v1)
        v1 = self.vitLayer_UpConv(v1)
        v1 = self.vitLayer_UpConv(v1)
        y = torch.cat([y, e2, v1], dim=1)
        y = self.upconv_3(y)

        y = self.ups[3](y)
        e1 = self.ups[7](e1)
        v0 = self.vitLayer_UpConv(v0)
        v0 = self.vitLayer_UpConv(v0)
        v0 = self.vitLayer_UpConv(v0)
        v0 = self.vitLayer_UpConv(v0)
        y = torch.cat([y, e1, v0], dim=1)
        y = self.upconv_4(y)

        out1 = self.final_conv1(y)
        out1 = self.final_relu1(out1)
        out = self.final_conv2(out1)
        out = self.final_relu2(out)
        out = self.final_conv3(out)
        return out

x = torch.randn(4, 3, 224, 224)
model = LFFTNet_SN6()
preds = model(x)
print(x.shape)
print(preds.shape)
print("LFFTNet_SN6")