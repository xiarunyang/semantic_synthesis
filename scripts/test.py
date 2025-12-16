import os
import sys
import yaml
import argparse
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data.ade20k_loader import get_dataloader
from models.diffusion_model import InstanceAwareSemanticSynthesis
from utils.metrics import MetricsEvaluator


def load_model(checkpoint_path, config):

    model = InstanceAwareSemanticSynthesis(config)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    return model


def evaluate_model(args):

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    model = load_model(args.checkpoint, config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    test_loader = get_dataloader(config, split='validation')
    

    evaluator = MetricsEvaluator(device=device)
    
    metrics = evaluator.evaluate_all(
        model,
        test_loader,
        save_dir=args.output_dir
    )
    
    # 保存结果
    results_path = os.path.join(args.output_dir, 'metrics.yaml')
    with open(results_path, 'w') as f:
        yaml.dump(metrics, f)
    print(f"\n 评估结果已保存到: {results_path}")
    
    return metrics


def generate_samples(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    

    model = load_model(args.checkpoint, config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    

    test_loader = get_dataloader(config, split='validation')
    

    os.makedirs(args.output_dir, exist_ok=True)

    model.eval()
    
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_loader)):
            if idx >= args.num_samples:
                break
            
            seg_maps = batch['seg_map'].to(device)
            instance_maps = batch['instance_map'].to(device)
            

            generated = model.generate(
                seg_maps,
                instance_maps,
                num_inference_steps=args.num_steps
            )

            for i in range(generated.shape[0]):
                img = generated[i].cpu().permute(1, 2, 0).numpy()
                img = (img * 255).astype('uint8')
                
                from PIL import Image
                Image.fromarray(img).save(
                    os.path.join(args.output_dir, f'sample_{idx}_{i}.png')
                )
    
    print(f"\n✓ 样本已保存到: {args.output_dir}")

def main():
    parser = argparse.ArgumentParser(description='测试和评估语义图像合成模型')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['evaluate', 'generate', 'compare'],
                       help='运行模式')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='输出目录')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='生成样本数量')
    parser.add_argument('--num_steps', type=int, default=50,
                       help='扩散步数')
    
    args = parser.parse_args()
    
    if args.mode == 'evaluate':
        evaluate_model(args)
    elif args.mode == 'generate':
        generate_samples(args)
    elif args.mode == 'compare':
        compare_with_baselines(args)


if __name__ == "__main__":
    main()
