
mkdir -p data
mkdir -p checkpoints
mkdir -p results
cd data

if [ ! -d "ADEChallengeData2016" ]; then
    wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
    unzip -q ADEChallengeData2016.zip
    rm ADEChallengeData2016.zip


cd ..
python3 << EOF
from huggingface_hub import snapshot_download
import os

model_id = "runwayml/stable-diffusion-v1-5"
cache_dir = "./checkpoints/stable-diffusion-v1-5"

if not os.path.exists(cache_dir):
    snapshot_download(
        repo_id=model_id,
        cache_dir=cache_dir,
        resume_download=True
    )
    print("  ✓ Stable Diffusion模型下载完成!")
else:
    print("  ✓ Stable Diffusion模型已存在")
EOF
python3 << EOF
from detectron2 import model_zoo
from detectron2.config import get_cfg

cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
)
EOF
if [ -d "data/ADEChallengeData2016" ]; then
    train_count=$(ls data/ADEChallengeData2016/images/training/*.jpg 2>/dev/null | wc -l)
    val_count=$(ls data/ADEChallengeData2016/images/validation/*.jpg 2>/dev/null | wc -l)
fi
