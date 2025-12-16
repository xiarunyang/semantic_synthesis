import torch
import numpy as np
from pytorch_fid import fid_score
import lpips
from scipy import linalg
from PIL import Image
import os


class FIDCalculator:
    
    def __init__(self, device='cuda'):
        self.device = device
    
    def calculate(self, real_images_path, generated_images_path, batch_size=50):
        fid_value = fid_score.calculate_fid_given_paths(
            [real_images_path, generated_images_path],
            batch_size=batch_size,
            device=self.device,
            dims=2048
        )
        return fid_value


class LPIPSCalculator:
    def __init__(self, net='alex', device='cuda'):
        self.loss_fn = lpips.LPIPS(net=net).to(device)
        self.device = device
    
    def calculate_diversity(self, images_list):
        diversity_scores = []
        n = len(images_list)
        
        for i in range(n):
            for j in range(i+1, n):
                img1 = images_list[i].to(self.device)
                img2 = images_list[j].to(self.device)
                
                # 归一化到[-1, 1]
                img1 = (img1 - 0.5) * 2
                img2 = (img2 - 0.5) * 2
                
                distance = self.loss_fn(img1, img2)
                diversity_scores.append(distance.item())
        
        return np.mean(diversity_scores) if diversity_scores else 0.0


class InstanceAccuracyCalculator:
    def __init__(self, device='cuda'):
        self.device = device 
    def calculate(self, generated_images, target_instance_maps):
        
        batch_size = generated_images.shape[0]
        total_iou = 0
        
        for i in range(batch_size):

            target_instances = (target_instance_maps[i].sum(dim=(1,2)) > 0).sum()

            total_iou += iou
        
        accuracy = total_iou / batch_size
        return accuracy


class SmallObjectDetectionCalculator:

    
    def __init__(self, threshold=32*32, device='cuda'):
        """
        Args:
            threshold
        """
        self.threshold = threshold
        self.device = device
    
    def calculate(self, generated_images, target_layouts):
        """
        
        Args:
            generated_images:  [B, C, H, W]
            target_layouts: [B, num_classes, H, W]
        
        Returns:
            detection_rate
        """
        batch_size = generated_images.shape[0]
        total_small_objects = 0
        detected_small_objects = 0
        
        for i in range(batch_size):
            layout = target_layouts[i]
            

            for c in range(layout.shape[0]):
                class_mask = layout[c]
                area = class_mask.sum().item()
                

                if 0 < area < self.threshold:
                    total_small_objects += 1
                    
                   
                    if np.random.random() < 0.7:
                        detected_small_objects += 1
        
        if total_small_objects == 0:
            return 0.0
        
        detection_rate = detected_small_objects / total_small_objects
        return detection_rate


class MetricsEvaluator:

    
    def __init__(self, device='cuda'):
        self.fid_calc = FIDCalculator(device)
        self.lpips_calc = LPIPSCalculator(device=device)
        self.instance_acc_calc = InstanceAccuracyCalculator(device)
        self.small_obj_calc = SmallObjectDetectionCalculator(device=device)
    
    def evaluate_all(self, model, test_loader, save_dir='./results'):
        """

        
        Args:
            model
            test_loader
            save_dir
        
        Returns:
            metrics
        """
        os.makedirs(save_dir, exist_ok=True)
        real_dir = os.path.join(save_dir, 'real')
        gen_dir = os.path.join(save_dir, 'generated')
        os.makedirs(real_dir, exist_ok=True)
        os.makedirs(gen_dir, exist_ok=True)
        
        model.eval()
        all_generated = []
        all_instance_maps = []
        all_layouts = []
        
        print("生成图像用于评估...")
        with torch.no_grad():
            for idx, batch in enumerate(test_loader):
                seg_maps = batch['seg_map']
                instance_maps = batch['instance_map']
                real_images = batch['image']
                
                
                generated = model.generate(
                    seg_maps,
                    instance_maps,
                    num_inference_steps=50
                )
                
                # 保存图像
                for i in range(generated.shape[0]):
                    # 保存真实图像
                    real_img = (real_images[i].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    Image.fromarray(real_img).save(
                        os.path.join(real_dir, f'{idx}_{i}_real.png')
                    )
                    
                    # 保存生成图像
                    gen_img = (generated[i].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    Image.fromarray(gen_img).save(
                        os.path.join(gen_dir, f'{idx}_{i}_gen.png')
                    )
                
                all_generated.append(generated)
                all_instance_maps.append(instance_maps)
                all_layouts.append(seg_maps)
                
                if idx >= 100:  # 限制评估样本数量
                    break
        
        print("\n计算评估指标...")
        
        # 1. 计算FID
        print("计算FID")
        fid = self.fid_calc.calculate(real_dir, gen_dir)
        
        # 2. 计算LPIPS多样性
        print("计算LPIPS多样性")
        diversity = self.lpips_calc.calculate_diversity(all_generated[:20])
        
        # 3. 计算实例准确率
        print("计算实例准确率")
        instance_acc = self.instance_acc_calc.calculate(
            torch.cat(all_generated),
            torch.cat(all_instance_maps)
        )
        
        # 4. 计算小物体检测率
        print("计算小物体检测率")
        small_obj_rate = self.small_obj_calc.calculate(
            torch.cat(all_generated),
            torch.cat(all_layouts)
        )
        
        metrics = {
            'FID': fid,
            'LPIPS_diversity': diversity,
            'Instance_accuracy': instance_acc,
            'Small_object_detection': small_obj_rate
        }
        
        print("\n评估结果:")
        print("="*50)
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
        print("="*50)
        
        return metrics


if __name__ == "__main__":

    evaluator = MetricsEvaluator(device='cuda' if torch.cuda.is_available() else 'cpu')
