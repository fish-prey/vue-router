import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 基础组件 (LayerNorm2d, FeatureModulation, NoiseSensingModule 保持不变) ---
class LayerNorm2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        return (x - u) / torch.sqrt(s + 1e-6) * self.weight + self.bias

class FeatureModulation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv_gamma = nn.Sequential(
            nn.Conv2d(1, dim, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim, dim, 1)
        )
        self.conv_beta = nn.Sequential(
            nn.Conv2d(1, dim, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim, dim, 1)
        )

    def forward(self, x, noise_map):
        gamma = self.conv_gamma(noise_map)
        beta = self.conv_beta(noise_map)
        return x * (1 + gamma) + beta

class NoiseSensingModule(nn.Module):
    def __init__(self, in_c=3):
        super().__init__()
        self.sensing = nn.Sequential(
            nn.Conv2d(in_c, 16, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 8, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(8, 1, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.sensing(x)
    
class GatedBlock(nn.Module):
    def __init__(self, dim, expansion=1.8, k_size=3):
        super().__init__()
        self.norm = LayerNorm2d(dim)
        hidden_dim = int(dim * expansion)
        self.project_in = nn.Conv2d(dim, hidden_dim * 2, 1, bias=False)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, k_size, 1, k_size//2, groups=hidden_dim, bias=False)
        self.project_out = nn.Conv2d(hidden_dim, dim, 1, bias=False)

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        x = self.project_in(x)
        x1, x2 = x.chunk(2, dim=1)
        x = x1 * F.gelu(self.dwconv(x2)) 
        return shortcut + self.project_out(x)

def make_stage(dim, num_blocks, expansion, k_size):
    blocks = [GatedBlock(dim, expansion, k_size) for _ in range(num_blocks)]
    if num_blocks > 1:
        return nn.Sequential(*blocks, nn.Conv2d(dim, dim, 3, 1, 1))
    return blocks[0]

class MGANet(nn.Module):
    def __init__(self, in_c=3, out_c=3, base_c=24, expansion=1.5, k_size=3):
        super().__init__()
        self.noise_sensor = NoiseSensingModule(in_c)
        self.head = nn.Conv2d(in_c, base_c, 3, 1, 1)
        
        self.enc1 = GatedBlock(base_c, expansion, k_size)
        self.mod1 = FeatureModulation(base_c)
        self.down1 = nn.Conv2d(base_c, base_c * 2, 3, 2, 1)
        
        self.enc2 = GatedBlock(base_c * 2, expansion, k_size)
        self.mod2 = FeatureModulation(base_c * 2)
        self.down2 = nn.Conv2d(base_c * 2, base_c * 4, 3, 2, 1)
        
        self.bottleneck = GatedBlock(base_c * 4, expansion, k_size)
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.fuse1 = nn.Conv2d(base_c * 6, base_c * 2, 1)
        self.dec1 = GatedBlock(base_c * 2, expansion, k_size)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.fuse2 = nn.Conv2d(base_c * 3, base_c, 1)
        self.dec2 = GatedBlock(base_c, expansion, k_size)
        
        self.tail = nn.Conv2d(base_c, out_c, 3, 1, 1)

    def forward(self, x, return_feats=False):
        identity = x
        noise_map = self.noise_sensor(x) 
        
        x = self.head(x)
        e1 = self.mod1(self.enc1(x), noise_map) 
        
        e2_in = self.down1(e1)
        m2 = F.interpolate(noise_map, size=e2_in.shape[2:], mode='bilinear')
        e2 = self.mod2(self.enc2(e2_in), m2) 
        
        b = self.bottleneck(self.down2(e2)) 
        
        d1 = self.dec1(self.fuse1(torch.cat([self.up1(b), e2], dim=1)))
        d2 = self.dec2(self.fuse2(torch.cat([self.up2(d1), e1], dim=1)))
        
        out = torch.clamp(self.tail(d2) + identity, 0.0, 1.0)
        
        if return_feats:
            return out, {"bottleneck": b, "spatial_ref": d2}, noise_map 
        return out

class MGANet_L(nn.Module):
    def __init__(self, in_c=3, out_c=3, base_c=64, expansion=2, k_size=7, num_blocks=2):
        super().__init__()
        self.noise_sensor = NoiseSensingModule(in_c)
        self.head = nn.Conv2d(in_c, base_c, 3, 1, 1)
        
        self.enc1 = make_stage(base_c, num_blocks, expansion, k_size)
        self.mod1 = FeatureModulation(base_c)
        self.down1 = nn.Conv2d(base_c, base_c * 2, 3, 2, 1)
        
        self.enc2 = make_stage(base_c * 2, num_blocks, expansion, k_size)
        self.mod2 = FeatureModulation(base_c * 2)
        self.down2 = nn.Conv2d(base_c * 2, base_c * 4, 3, 2, 1)
        
        # Bottleneck 增加一倍深度作为核心约束
        self.bottleneck = make_stage(base_c * 4, num_blocks*2, expansion, k_size)
        
        # Decoder
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.fuse1 = nn.Conv2d(base_c * 6, base_c * 2, 1)
        self.dec1 = make_stage(base_c * 2, num_blocks, expansion, k_size)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.fuse2 = nn.Conv2d(base_c * 3, base_c, 1)
        self.dec2 = make_stage(base_c, num_blocks, expansion, k_size)
        
        self.tail = nn.Conv2d(base_c, out_c, 3, 1, 1)

    def forward(self, x, return_feats=False):
        identity = x
        noise_map = self.noise_sensor(x) 
        x = self.head(x)
        
        # 即使是 Sequential，forward 逻辑也保持同源一致
        e1 = self.mod1(self.enc1(x), noise_map) 
        
        e2_in = self.down1(e1)
        m2 = F.interpolate(noise_map, size=e2_in.shape[2:], mode='bilinear', align_corners=False)
        e2 = self.mod2(self.enc2(e2_in), m2) 
        
        b = self.bottleneck(self.down2(e2)) 
        
        d1 = self.dec1(self.fuse1(torch.cat([self.up1(b), e2], dim=1)))
        d2 = self.dec2(self.fuse2(torch.cat([self.up2(d1), e1], dim=1)))
        
        out = torch.clamp(self.tail(d2) + identity, 0.0, 1.0)
        
        if return_feats:
            return out, {"bottleneck": b, "spatial_ref": d2}, noise_map 
        return out

if __name__ == "__main__":
    from thop import profile
    x = torch.randn(1, 3, 256, 256)
    
    m_s = MGANet()
    m_l = MGANet_L()
    
    f_s, p_s = profile(m_s, inputs=(x,), verbose=False)
    f_l, p_l = profile(m_l, inputs=(x,), verbose=False)
    
    print(f"MGANet (Student) Params: {p_s/1e6:.4f}M | GFLOPs: {f_s/1e9:.2f}")
    print(f"MGANet_L (Teacher) Params: {p_l/1e6:.2f}M | GFLOPs: {f_l/1e9:.2f}")