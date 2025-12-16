import os
import sys
import yaml
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
import wandb

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data.ade20k_loader import get_dataloader
from models.diffusion_model import InstanceAwareSemanticSynthesis
from utils.metrics import FIDCalculator, LPIPSCalculator


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_stage1(model, train_loader, val_loader, config, accelerator):
    """
    阶段1训练：仅训练Instance-Aware SPADE模块和注意力引导模块
    """
    print("\n" + "="*50)
    print("="*50)
    
    model.freeze_pretrained_models()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config['training']['stage1_lr'],
        weight_decay=0.01
    )
    
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['stage1_epochs']
    )

    model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, lr_scheduler
    )
    
    # 训练循环
    global_step = 0
    for epoch in range(config['training']['stage1_epochs']):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['stage1_epochs']}")
        for step, batch in enumerate(pbar):
            with accelerator.accumulate(model):
                # 前向传播
                outputs = model(batch)
                loss = outputs['loss']
                
                # 反向传播
                accelerator.backward(loss)
                
                # 梯度裁剪
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                optimizer.zero_grad()
                
                # 记录
                epoch_loss += loss.item()
                global_step += 1
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'loss_denoising': f"{outputs['loss_denoising'].item():.4f}",
                    'loss_attention': f"{outputs['loss_attention'].item():.4f}",
                })
                
                # 记录到wandb
                if config['logging']['use_wandb'] and global_step % config['logging']['log_every'] == 0:
                    wandb.log({
                        'train/loss': loss.item(),
                        'train/loss_denoising': outputs['loss_denoising'].item(),
                        'train/loss_attention': outputs['loss_attention'].item(),
                        'train/lr': optimizer.param_groups[0]['lr'],
                        'epoch': epoch,
                        'step': global_step
                    })
                
                # 保存检查点
                if global_step % config['training']['save_every'] == 0:
                    save_checkpoint(model, optimizer, epoch, global_step, config, accelerator, stage=1)
        
        # 验证
        if (epoch + 1) % 5 == 0:
            val_loss = validate(model, val_loader, accelerator)
            print(f"\nValidation Loss: {val_loss:.4f}")
            
            if config['logging']['use_wandb']:
                wandb.log({
                    'val/loss': val_loss,
                    'epoch': epoch
                })
        
        lr_scheduler.step()
    
    return model


def train_stage2(model, train_loader, val_loader, config, accelerator):

    print("\n" + "="*50)
    print("="*50)
    
    # 解冻所有参数
    model.unfreeze_for_finetuning()
    
    # 优化器 - 使用小学习率
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['stage2_lr'],
        weight_decay=0.01
    )
    
    # 学习率调度器
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['stage2_epochs']
    )
    
    # 使用Accelerator包装
    model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, lr_scheduler
    )
    
    # 训练循环
    global_step = 0
    for epoch in range(config['training']['stage2_epochs']):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['stage2_epochs']}")
        for step, batch in enumerate(pbar):
            with accelerator.accumulate(model):
                # 前向传播
                outputs = model(batch)
                loss = outputs['loss']
                
                # 反向传播
                accelerator.backward(loss)
                
                # 梯度裁剪
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                optimizer.zero_grad()
                
                # 记录
                epoch_loss += loss.item()
                global_step += 1
                
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                
                # 记录到wandb
                if config['logging']['use_wandb'] and global_step % config['logging']['log_every'] == 0:
                    wandb.log({
                        'train_stage2/loss': loss.item(),
                        'train_stage2/lr': optimizer.param_groups[0]['lr'],
                        'step': global_step
                    })
                
                # 保存检查点
                if global_step % config['training']['save_every'] == 0:
                    save_checkpoint(model, optimizer, epoch, global_step, config, accelerator, stage=2)
        
        # 验证
        val_loss = validate(model, val_loader, accelerator)
        print(f"\nValidation Loss: {val_loss:.4f}")
        
        if config['logging']['use_wandb']:
            wandb.log({
                'val_stage2/loss': val_loss,
                'epoch': epoch
            })
        
        lr_scheduler.step()
    return model


def validate(model, val_loader, accelerator):
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            outputs = model(batch)
            total_loss += outputs['loss'].item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    return avg_loss


def save_checkpoint(model, optimizer, epoch, step, config, accelerator, stage):
    checkpoint_dir = os.path.join(config['logging'].get('checkpoint_dir', './checkpoints'), f'stage{stage}')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch{epoch}_step{step}.pt')
    
    # 使用accelerator保存
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    
    accelerator.save({
        'epoch': epoch,
        'step': step,
        'model_state_dict': unwrapped_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config
    }, checkpoint_path)
    
    print(f"\n{checkpoint_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 初始化Accelerator
    accelerator = Accelerator(
        mixed_precision=config['training']['mixed_precision'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps']
    )
    
    # 初始化wandb
    if config['logging']['use_wandb'] and accelerator.is_main_process:
        wandb.init(
            project=config['logging']['project_name'],
            config=config,
            name=f"instance_aware_synthesis_{config['model']['name']}"
        )
    
    train_loader = get_dataloader(config, split='training')
    val_loader = get_dataloader(config, split='validation')

    model = InstanceAwareSemanticSynthesis(config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  - 可训练参数: {trainable_params:,}")
    print(f"  - 总参数: {total_params:,}")
    print(f"  - 可训练比例: {100 * trainable_params / total_params:.2f}%")
    
    # 阶段1训练
    model = train_stage1(model, train_loader, val_loader, config, accelerator)
    
    # 阶段2训练
    model = train_stage2(model, train_loader, val_loader, config, accelerator)
    
    # 保存最终模型
    if accelerator.is_main_process:
        final_model_path = os.path.join(config['logging'].get('checkpoint_dir', './checkpoints'), 'final_model.pt')
        torch.save(model.state_dict(), final_model_path)
        print(f"\n最终模型已保存: {final_model_path}")
    
    if config['logging']['use_wandb']:
        wandb.finish()
    
    print("\n" + "="*50)
    print("="*50)


if __name__ == "__main__":
    main()
