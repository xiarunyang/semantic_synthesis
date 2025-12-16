import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo


class ADE20KDataset(Dataset):
    
    def __init__(self, root_dir, split='training', image_size=512, load_instances=True):
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size
        self.load_instances = load_instances
        
        self.img_dir = os.path.join(root_dir, 'images', split)
        self.seg_dir = os.path.join(root_dir, 'annotations', split)
        
        self.image_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.jpg')])
        self.img_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        self.seg_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=Image.NEAREST),
        ])
        
        if load_instances:
            self.instance_predictor = self._init_instance_segmentation()
    
    def _init_instance_segmentation(self):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        ))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        return DefaultPredictor(cfg)
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        seg_name = img_name.replace('.jpg', '.png')
        seg_path = os.path.join(self.seg_dir, seg_name)
        seg_map = Image.open(seg_path)
        
        # 应用变换
        image_tensor = self.img_transform(image)
        seg_map = self.seg_transform(seg_map)
        seg_array = np.array(seg_map)
        
        # 转换为one-hot编码
        seg_tensor = self._to_onehot(seg_array, num_classes=150)
        
        result = {
            'image': image_tensor,
            'seg_map': seg_tensor,
            'filename': img_name
        }
        
        # 生成实例分割图（如果需要）
        if self.load_instances:
            instance_map = self._generate_instance_map(image, seg_array)
            result['instance_map'] = instance_map
        
        return result
    
    def _to_onehot(self, seg_array, num_classes=150):

        h, w = seg_array.shape
        onehot = np.zeros((num_classes, h, w), dtype=np.float32)
        for i in range(num_classes):
            onehot[i] = (seg_array == i).astype(np.float32)
        return torch.from_numpy(onehot)
    
    def _generate_instance_map(self, image, seg_map):
        img_array = np.array(image)

        outputs = self.instance_predictor(img_array)
        instances = outputs["instances"].to("cpu")

        h, w = seg_map.shape
        instance_map = np.zeros((100, h, w), dtype=np.float32)  
        
        if len(instances) > 0:
            masks = instances.pred_masks.numpy()
            for i, mask in enumerate(masks[:100]):  
                mask_resized = Image.fromarray(mask).resize((w, h), Image.NEAREST)
                instance_map[i] = np.array(mask_resized).astype(np.float32)
        
        return torch.from_numpy(instance_map)


def get_dataloader(config, split='training'):
    dataset = ADE20KDataset(
        root_dir=config['dataset']['root_dir'],
        split=split,
        image_size=config['dataset']['image_size'],
        load_instances=True
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['dataset']['batch_size'],
        shuffle=(split == 'training'),
        num_workers=config['dataset']['num_workers'],
        pin_memory=True
    )
    
    return dataloader


if __name__ == "__main__":
    config = {
        'dataset': {
            'root_dir': './data/ADEChallengeData2016',
            'image_size': 512,
            'batch_size': 2,
            'num_workers': 2
        }
    }
    
    loader = get_dataloader(config, split='validation')
    batch = next(iter(loader))
    
    print(f"Image shape: {batch['image'].shape}")
    print(f"Segmentation map shape: {batch['seg_map'].shape}")
    print(f"Instance map shape: {batch['instance_map'].shape}")
