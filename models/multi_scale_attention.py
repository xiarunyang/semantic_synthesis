import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List


class MultiScaleAttentionGuidance(nn.Module):

    
    def __init__(self, scales=[8, 16, 32, 64], guidance_weights=None):
 
        super().__init__()
        self.scales = scales
        
        if guidance_weights is None:

            guidance_weights = [s / max(scales) for s in scales]
        self.guidance_weights = guidance_weights
        

        self.attention_maps = {}
        self.hooks = []
    
    def register_attention_hooks(self, unet):
     
        def get_attention_hook(name, scale):
            def hook(module, input, output):
                # 保存注意力图
                if isinstance(output, tuple):
                    attn = output[0]
                else:
                    attn = output
                self.attention_maps[f"{name}_scale{scale}"] = attn
            return hook
        
        # 注册到不同层级的注意力模块
        for name, module in unet.named_modules():
            if 'attn' in name.lower() or 'attention' in name.lower():
                # 判断该层对应的特征图尺度
                for scale in self.scales:
                    if f"down_blocks" in name and f"{scale}" in name:
                        hook = module.register_forward_hook(
                            get_attention_hook(name, scale)
                        )
                        self.hooks.append(hook)
                    elif f"up_blocks" in name and f"{scale}" in name:
                        hook = module.register_forward_hook(
                            get_attention_hook(name, scale)
                        )
                        self.hooks.append(hook)
    
    def compute_layout_attention_loss(self, target_layout, small_object_mask=None):
      
        total_loss = 0.0
        num_attn_maps = 0
        
        for (name, attn_map), weight in zip(
            self.attention_maps.items(), 
            self.guidance_weights
        ):
            # 提取尺度信息
            scale = int(name.split('scale')[-1])
            
            # Resize目标布局到注意力图尺度
            target_resized = F.interpolate(
                target_layout, 
                size=(scale, scale), 
                mode='nearest'
            )
            
            # 计算注意力对齐损失
            # 这里使用KL散度或MSE
            if attn_map.dim() == 4:  # [B, heads, H*W, H*W]
                B, heads, _, _ = attn_map.shape
                # 对角线表示每个位置对自己的注意力
                self_attn = attn_map.diagonal(dim1=-2, dim2=-1)
                self_attn = self_attn.reshape(B, heads, scale, scale)
                
                # 平均所有注意力头
                self_attn = self_attn.mean(dim=1, keepdim=True)
                
                # 如果有小物体掩码，加大小物体区域的权重
                if small_object_mask is not None:
                    small_obj_resized = F.interpolate(
                        small_object_mask, 
                        size=(scale, scale), 
                        mode='bilinear'
                    )
                    # 小物体区域权重x2
                    loss_weight = 1.0 + small_obj_resized
                else:
                    loss_weight = 1.0
                
                # 计算加权MSE损失
                loss = F.mse_loss(
                    self_attn * loss_weight, 
                    target_resized * loss_weight,
                    reduction='mean'
                )
                
                total_loss += weight * loss
                num_attn_maps += 1
        
        if num_attn_maps > 0:
            total_loss = total_loss / num_attn_maps
        
        return total_loss
    
    def clear_attention_maps(self):

        self.attention_maps = {}
    
    def remove_hooks(self):

        for hook in self.hooks:
            hook.remove()
        self.hooks = []


class GuidedDenoisingStep:
   
    def __init__(self, guidance_module, guidance_scale=7.5):
        
        self.guidance_module = guidance_module
        self.guidance_scale = guidance_scale
    
    def __call__(self, 
                 unet, 
                 latents, 
                 timestep, 
                 encoder_hidden_states,
                 target_layout,
                 small_object_mask=None):
        
        # 清空之前的注意力图
        self.guidance_module.clear_attention_maps()
        
        # 设置梯度计算
        latents = latents.detach().requires_grad_(True)
        
        # 正向传播
        noise_pred = unet(
            latents, 
            timestep, 
            encoder_hidden_states=encoder_hidden_states
        ).sample
        
        # 计算注意力引导损失
        attn_loss = self.guidance_module.compute_layout_attention_loss(
            target_layout, 
            small_object_mask
        )
        
        # 计算梯度
        if attn_loss.requires_grad:
            grad = torch.autograd.grad(
                attn_loss, 
                latents,
                retain_graph=False
            )[0]
            
            # 应用引导
            noise_pred = noise_pred - self.guidance_scale * grad
        
        return noise_pred.detach()


def detect_small_objects(seg_map, threshold=32*32):
    
    B, C, H, W = seg_map.shape
    small_object_mask = torch.zeros(B, 1, H, W, device=seg_map.device)
    
    for b in range(B):
        for c in range(C):
            class_mask = seg_map[b, c]
            
            area = class_mask.sum()
            if 0 < area < threshold:
                small_object_mask[b, 0] += class_mask
    
    # 二值化
    small_object_mask = (small_object_mask > 0).float()
    
    return small_object_mask


if __name__ == "__main__":
    # 测试多尺度注意力引导
    from diffusers import UNet2DConditionModel
    
    # 加载预训练U-Net
    unet = UNet2DConditionModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="unet"
    )
    
    # 创建引导模块
    guidance = MultiScaleAttentionGuidance(
        scales=[8, 16, 32, 64],
        guidance_weights=[0.5, 1.0, 1.5, 2.0]
    )

    guidance.register_attention_hooks(unet)

    batch_size = 2
    latents = torch.randn(batch_size, 4, 64, 64)
    timestep = torch.tensor([500])
    encoder_hidden_states = torch.randn(batch_size, 77, 768)
    target_layout = torch.randn(batch_size, 150, 512, 512)
    

    guided_step = GuidedDenoisingStep(guidance, guidance_scale=7.5)
    