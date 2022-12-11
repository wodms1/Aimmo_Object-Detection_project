# 1. Prerequisites

## Step 1. Create conda env
```
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

## Step 2. install pytorch
```
# on GPU
conda install pytorch torchvision -c pytorch

# on CPU
conda install pytorch torchvision cpuonly -c pytorch
```

# 2. Install

## Step 0. install mmcv & mim
```
pip install -U openmim
mim install mmcv-full
```
## step 1. install mmdetection
```
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
	# -v : is verbose
	# -e : editable mode install
```
## Step 2. Verify
```
mim download mmdet --config yolov3_mobilenetv2_320_300e_coco --dest .
python demo/image_demo.py demo/demo.jpg yolov3_mobilenetv2_320_300e_coco.py yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth --device cpu --out-file result.jpg
```

