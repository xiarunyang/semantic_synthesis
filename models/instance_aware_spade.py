import torch
import torch.nn as nn
import torch.nn.functional as F


class InstanceAwareSPADE(nn.Module):

    
    def __init__(self, norm_nc, label_nc, instance_nc=100, hidden_nc=128):
        super().__init__()
        
        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        
        self.semantic_mlp = nn.Sequential(
            nn.Conv2d(label_nc, hidden_nc, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.instance_mlp = nn.Sequential(
            nn.Conv2d(instance_nc, hidden_nc, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # 融合层
        self.gamma_conv = nn.Conv2d(hidden_nc * 2, norm_nc, kernel_size=3, padding=1)
        self.beta_conv = nn.Conv2d(hidden_nc * 2, norm_nc, kernel_size=3, padding=1)
    
    def forward(self, x, seg_map, instance_prob_map):
  
        # 归一化
        normalized = self.param_free_norm(x)
        
        # Resize条件图到特征图大小
        seg_map = F.interpolate(seg_map, size=x.size()[2:], mode='nearest')
        instance_prob_map = F.interpolate(instance_prob_map, size=x.size()[2:], mode='bilinear', align_corners=False)
        
        # 提取语义和实例特征
        semantic_feat = self.semantic_mlp(seg_map)
        instance_feat = self.instance_mlp(instance_prob_map)
        
        # 拼接特征
        combined_feat = torch.cat([semantic_feat, instance_feat], dim=1)
        
        # 生成调制参数
        gamma = self.gamma_conv(combined_feat)
        beta = self.beta_conv(combined_feat)
        
        # 应用仿射变换
        out = normalized * (1 + gamma) + beta
        
        return out


class InstanceProbabilisticSampler(nn.Module):
 
    
    def __init__(self, noise_level=0.1):
        super().__init__()
        self.noise_level = noise_level
    
    def forward(self, instance_map, training=True):
  
        if training and self.noise_level > 0:
            # 添加高斯噪声
            noise = torch.randn_like(instance_map) * self.noise_level
            noisy_map = instance_map + noise
        else:
            noisy_map = instance_map
        
        # Softmax归一化为概率分布
        prob_map = F.softmax(noisy_map, dim=1)
        
        return prob_map


class SPADEResBlock(nn.Module):
    
    def __init__(self, fin, fout, label_nc, instance_nc):
        super().__init__()
        
        fmiddle = min(fin, fout)
        
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        
        self.norm_0 = InstanceAwareSPADE(fin, label_nc, instance_nc)
        self.norm_1 = InstanceAwareSPADE(fmiddle, label_nc, instance_nc)
        
        if fin != fout:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)
            self.norm_s = InstanceAwareSPADE(fin, label_nc, instance_nc)
        else:
            self.conv_s = None
    
    def forward(self, x, seg, inst_prob):
        x_s = self.shortcut(x, seg, inst_prob)
        
        dx = self.conv_0(self.actvn(self.norm_0(x, seg, inst_prob)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg, inst_prob)))
        
        out = x_s + dx
        return out
    
    def shortcut(self, x, seg, inst_prob):
        if self.conv_s is not None:
            x_s = self.conv_s(self.norm_s(x, seg, inst_prob))
        else:
            x_s = x
        return x_s
    
    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


if __name__ == "__main__":
    # 测试模块
    batch_size = 2
    num_classes = 150
    num_instances = 100
    h, w = 64, 64
    
    # 创建模块
    spade = InstanceAwareSPADE(norm_nc=256, label_nc=num_classes, instance_nc=num_instances)
    sampler = InstanceProbabilisticSampler(noise_level=0.1)
    
    # 创建假数据
    x = torch.randn(batch_size, 256, h, w)
    seg_map = torch.randn(batch_size, num_classes, h, w)
    instance_map = torch.rand(batch_size, num_instances, h, w)
    
    # 前向传播
    prob_map = sampler(instance_map, training=True)
    out = spade(x, seg_map, prob_map)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Probability map shape: {prob_map.shape}")
    print(" Instance-Aware SPADE pass")
