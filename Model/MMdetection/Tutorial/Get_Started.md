# 0. GCP

- GCP 환경 setting
    - GCP

# 1. Prerequisites

- Create VM
    - Step 1. Create conda env
        
        ```python
        conda create --name openmmlab python=3.8 -y
        conda activate openmmlab
        ```
        
    - Step 2. install pytorch
        
        ```python
        # on GPU
        conda install pytorch torchvision -c pytorch
        
        # on CPU
        conda install pytorch torchvision cpuonly -c pytorch
        ```
        

# 2.Installation

- Shell
    
    ### 1. Install
    
    - Step 0. install mmcv & mim
        
        ```
        pip install -U openmim
        mim install mmcv-full
        ```
        
    - step 1. install mmdetection
        
        ```python
        git clone https://github.com/open-mmlab/mmdetection.git
        cd mmdetection
        pip install -v -e .
        	# -v : is verbose
        	# -e : editable mode install
        ```
        
    
    ### 2. Verify
    
    - Step 0. download config & checkpoint
        
        ```python
        mim download mmdet --config yolov3_mobilenetv2_320_300e_coco --dest .
        ```
        
    - Step 1. verify the inference demo
        
        ```python
        python demo/image_demo.py demo/demo.jpg yolov3_mobilenetv2_320_300e_coco.py yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth --device cpu --out-file result.jpg
        ```
        
- Colab
    - Step 1. install mmcv & mim
        
        ```python
        !pip3 install openmim
        !mim install mmcv-full
        ```
        
    - Step 2. Install MMDetection from the source.
        
        ```python
        !git clone https://github.com/open-mmlab/mmdetection.git
        %cd mmdetection
        !pip install -e .
        ```
        
    - Step 3. Verification.
        
        ```python
        import mmdet
        print(mmdet.__version__)
        # Example output: 2.23.0
        ```
        

# 3. Benchmark and Model Zoo

- Common Setting
    - 모든 model은 coco_2017_train으로 훈련되었다.
    - 모든 model은 coco_2017_val으로 평가되었다.
    - 모드 backbone은 pytorch-style ImageNet 기반 사전훈련되었다.
