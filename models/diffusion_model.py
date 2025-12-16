import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from .instance_aware_spade import InstanceAwareSPADE, InstanceProbabilisticSampler, SPADEResBlock
from .multi_scale_attention import MultiScaleAttentionGuidance, GuidedDenoisingStep, detect_small_objects


class InstanceAwareSemanticSynthesis(nn.Module):

    
    def __init__(self, config):
        super().__init__()
        self.config = config
        

        model_id = config['model']['pretrained_model']
        

        pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if config['training']['mixed_precision'] == 'fp16' else torch.float32
        )
        
        self.vae = pipeline.vae
        self.unet = pipeline.unet
        self.text_encoder = pipeline.text_encoder
        self.tokenizer = pipeline.tokenizer
        self.scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

        self.freeze_pretrained_models()

        self.instance_sampler = InstanceProbabilisticSampler(
            noise_level=config['instance_aware']['noise_level']
        )

        self.inject_instance_aware_spade()

        self.attention_guidance = MultiScaleAttentionGuidance(
            scales=config['attention_guidance']['scales'],
            guidance_weights=config['attention_guidance']['guidance_weights']
        )
        self.attention_guidance.register_attention_hooks(self.unet)

        self.guided_denoising = GuidedDenoisingStep(
            self.attention_guidance,
            guidance_scale=7.5
        )
    
    def freeze_pretrained_models(self):

        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.unet.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_for_finetuning(self):

        for param in self.unet.parameters():
            param.requires_grad = True
    
    def inject_instance_aware_spade(self):

        num_classes = self.config['model']['num_semantic_classes']
        instance_nc = self.config['instance_aware'].get('num_instances_per_class', 100)
        

        for name, module in self.unet.named_modules():

            if isinstance(module, nn.GroupNorm):

                num_channels = module.num_channels
                spade_layer = InstanceAwareSPADE(
                    norm_nc=num_channels,
                    label_nc=num_classes,
                    instance_nc=instance_nc
                )

                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                if parent_name:
                    parent = dict(self.unet.named_modules())[parent_name]
                    setattr(parent, child_name, spade_layer)
                

                for param in spade_layer.parameters():
                    param.requires_grad = True
    
    def encode_text(self, text):

        text_inputs = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_inputs.input_ids.to(self.text_encoder.device))[0]
        return text_embeddings
    
    def forward(self, batch, timesteps=None):
 
        images = batch['image']
        seg_maps = batch['seg_map']
        instance_maps = batch['instance_map']
        
        batch_size = images.shape[0]
        device = images.device
        
        # 1. 将图像编码到潜在空间
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
        
        # 2. 采样噪声和时间步
        noise = torch.randn_like(latents)
        if timesteps is None:
            timesteps = torch.randint(
                0, self.scheduler.config.num_train_timesteps,
                (batch_size,), device=device
            ).long()
        
        # 3. 添加噪声
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        
        # 4. 生成实例概率图
        instance_prob_maps = self.instance_sampler(instance_maps, training=self.training)
        
        # 5. 编码条件（这里使用空文本或类别描述）
        text = [""] * batch_size  # 或者根据语义图生成描述
        encoder_hidden_states = self.encode_text(text)
        
        # 6. 预测噪声
        # 注意：这里seg_maps和instance_prob_maps会通过SPADE层注入到U-Net中
        # 需要在forward过程中传递这些条件
        model_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
        ).sample
        
        # 7. 计算去噪损失
        if self.scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.scheduler.config.prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.scheduler.config.prediction_type}")
        
        loss_denoising = nn.functional.mse_loss(model_pred.float(), target.float(), reduction="mean")
        
        # 8. 计算注意力引导损失（如果在训练模式）
        loss_attention = torch.tensor(0.0, device=device)
        if self.training:
            small_object_mask = detect_small_objects(
                seg_maps, 
                threshold=self.config['attention_guidance']['small_object_threshold']
            )
            loss_attention = self.attention_guidance.compute_layout_attention_loss(
                seg_maps,
                small_object_mask
            )
        
        # 总损失
        total_loss = loss_denoising + 0.1 * loss_attention
        
        return {
            'loss': total_loss,
            'loss_denoising': loss_denoising,
            'loss_attention': loss_attention
        }
    
    @torch.no_grad()
    def generate(self, seg_map, instance_map, num_inference_steps=50, guidance_scale=7.5):
       
        device = seg_map.device
        batch_size = 1
        
        # 1. 准备潜在表示
        height = width = self.config['dataset']['image_size']
        latents = torch.randn(
            (batch_size, self.unet.config.in_channels, height // 8, width // 8),
            device=device
        )
        
        # 2. 生成实例概率图
        instance_prob_map = self.instance_sampler(instance_map, training=False)
        
        # 3. 编码文本
        text = [""]
        encoder_hidden_states = self.encode_text(text)
        
        # 4. 设置调度器
        self.scheduler.set_timesteps(num_inference_steps)
        
        # 5. 检测小物体
        small_object_mask = detect_small_objects(seg_map)
        
        # 6. 去噪循环
        for t in self.scheduler.timesteps:
            # 使用引导去噪
            noise_pred = self.guided_denoising(
                self.unet,
                latents,
                t,
                encoder_hidden_states,
                seg_map,
                small_object_mask
            )
            
            # 更新latents
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        
        # 7. 解码到图像空间
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        
        # 8. 后处理
        image = (image / 2 + 0.5).clamp(0, 1)
        
        return image


if __name__ == "__main__":
    # 测试模型
    config = {
        'model': {
            'pretrained_model': 'runwayml/stable-diffusion-v1-5',
            'num_semantic_classes': 150
        },
        'dataset': {
            'image_size': 512
        },
        'training': {
            'mixed_precision': 'fp16'
        },
        'instance_aware': {
            'noise_level': 0.1,
            'num_instances_per_class': 100
        },
        'attention_guidance': {
            'scales': [8, 16, 32, 64],
            'guidance_weights': [0.5, 1.0, 1.5, 2.0],
            'small_object_threshold': 1024
        }
    }
    
    model = InstanceAwareSemanticSynthesis(config)


    batch = {
        'image': torch.randn(2, 3, 512, 512),
        'seg_map': torch.randn(2, 150, 512, 512),
        'instance_map': torch.rand(2, 100, 512, 512)
    }
    
    output = model(batch)
    print(f"loss: {output['loss'].item():.4f}")
