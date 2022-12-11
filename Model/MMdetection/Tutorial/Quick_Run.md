- MMDetectiondms Model Zoo안에 존재하는 수십여가지 model을 지원하고 다수의 표준 dataset(**Pascal VOC, COCO, CityScapes, LVIS, etc.**)또한 지원합니다. 해당 page는 해당 데이터와 모델의 구동 과정을 아래를 포함하여 보입니다.
    1. Zoo Model 사용 inference with given images
    2. Zoo Model 사용 Test with standard datasets
    3. Zoo Model (Pre-defined) with standard datasets
- Faster R-CNN 사용 & config,checkpoint 준비
    - Config: Model 구성
    - Checkpoint: pre-trained weights bias

# 1. Inference with existing models

- High-level APIs for inference
    - must **download checkpoint_file**
    
    ```python
    !wget -c https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
          -O checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
    ```
    
    ```python
    from mmdet.apis import init_detector, inference_detector
    import mmcv
    
    # config & checkpoint file 경로 지정 -> Faster R-CNN 사용
    config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    
    # mmdet.apis의 init_detector 사용하여 config+checkpoint로 model build
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    
    # random image 1장을 사용하여 test
    img = 'test.jpg'  # or img = mmcv.imread(img), which will only load it once
    result = inference_detector(model, img)
    # visualize the results in a new window
    model.show_result(img, result)
    # or save the visualization results to image files
    model.show_result(img, result, out_file='result.jpg')
    
    # 아래의 경우 비디오 존재시 비디오 구현
    # test a video and show the results
    video = mmcv.VideoReader('video.mp4')
    for frame in video:
        result = inference_detector(model, frame)
        model.show_result(frame, result, wait_time=1)
    ```
    
    ```python
    # show the results
    show_result_pyplot(model, img, result)
    ```
    
- Test existing models on standard datasets
    - show **how to test**
    - folder structure는 아래와 같아야한다.
    
    ```python
    mmdetection
    ├── mmdet
    ├── tools
    ├── configs
    ├── data
    │   ├── coco
    │   │   ├── annotations
    │   │   ├── train2017
    │   │   ├── val2017
    │   │   ├── test2017
    ```
    
    - 사용 standard datasets ⇒ cityscapes
        - cityscapes annotations은 coco format으로 변환해 주어야한다.
- Train predefined models on standard datasets
    - config를 사용하여 model 정의
    - batch_size = 8(GPU 수) X 2(sample per GPU) 로 기본 설정되어있다.
        
        따라서 수정이 필요하다.
        
        - **Please remember to check the bottom of the specific config file you want to use, it will have `auto_scale_lr.base_batch_size` if the batch size is not `16`**
        - batch_size = 8 → lr = 0.01
        - batch_size = 1 → lr =0.00125
        - config → optimizer **→ lr 수정!**

# 2. Train with customized datasets

- Prepare the customized dataset
    - MMDetection 가동 위해서는 3가지 format의 data로 변환해야한다.
        1. **reorganize the dataset into COCO format.**
            
            ```python
            {
                "images": [image],
                "annotations": [annotation],
                "categories": [category]
            }
            
            image = {
                "id": int,
                "width": int,
                "height": int,
                "file_name": str,
            }
            
            annotation = {
                "id": int,
                "image_id": int,
                "category_id": int,
                "segmentation": RLE or [polygon],
                "area": float,
                "bbox": [x,y,width,height],
                "iscrowd": 0 or 1,
            }
            
            categories = [{
                "id": int,
                "name": str,
                "supercategory": str,
            }]
            ```
            
            ```python
            images = []
            annotations = []
            
            for _ in range(2):
                images.append(dict(\
                                  id =1,
                                  file_name = '이름',
                                  height = 1024,
                                  width = 1920))
                data_anno = dict(\
                                image_id=1,
                                id=0,
                                categori_id=0,
                                bbox=[1,2,3,4],
                                ares=1000,
                                iscrowed=0)
                annotations.append(data_anno)
                
            coco_format_json = dict(
                    images=images,
                    annotations=annotations,
                    categories=[{'id':0, 'name': 'balloon'}])
            coco_format_json
            
            #저장
            #mmcv.dump(coco_format_json, out_file)
            ```
            
        2. **reorganize the dataset into a middle format.**
        3. **implement a new dataset.**
- Prepare a config
    - Mask R-CNN with FPN 구현 → 풍성 dataset을 활용한 학습  &구현
    
    ```python
    _base_ = '해당 model config file 경로 .py로 끝나는'
    
    # class의 개수와 재설정
    model = dict(
        roi_head=dict(
            bbox_head=dict(num_classes=1),
            mask_head=dict(num_classes=1)))
    
    # dataset 관련 경로,type,class 재설정
    dataset_type = 'COCODataset'
    classes = ('balloon',)
    data = dict(
        train=dict(
            img_prefix='balloon/train/',
            classes=classes,
            ann_file='balloon/train/annotation_coco.json'),
        val=dict(
            img_prefix='balloon/val/',
            classes=classes,
            ann_file='balloon/val/annotation_coco.json'),
        test=dict(
            img_prefix='balloon/val/',
            classes=classes,
            ann_file='balloon/val/annotation_coco.json'))
    
    # checkpoint를 활용한 사전학습 weights 가져오기
    load_from = 'checkpoint 경로 .pth로 끝나는'
    ```
    
- Train, test, inference models on the customized dataset.
    
    ```python
    # train
    python tools/train.py configs/balloon/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_balloon.py
    
    # test
    python tools/test.py configs/balloon/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_balloon.py work_dirs/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_balloon/latest.pth --eval bbox segm
    ```
    

# 3. Train with customized model

- **Prepare the standard dataset**
    - 경로 설정 해라
    
    ```python
    mmdetection
    ├── mmdet
    ├── tools
    ├── configs
    ├── data
    │   ├── coco
    │   │   ├── annotations
    │   │   ├── train2017
    │   │   ├── val2017
    │   │   ├── test2017
    │   ├── cityscapes
    │   │   ├── annotations
    │   │   ├── leftImg8bit
    │   │   │   ├── train
    │   │   │   ├── val
    ```
    
    - dataset 변환안되어 있다면 변환해라
- **Prepare your own customized model**
    - custom module  or training setting 해라
        - 수정된 FPN으로 구성된 Mask R-CNN_R50 구성해보자
            1. Define new neck
            
            ```python
            from ..builder import NECKS
            
            @NECKS.register_module()
            class AugFPN(nn.Module):
            
                def __init__(self,
                            in_channels,
                            out_channels,
                            num_outs,
                            start_level=0,
                            end_level=-1,
                            add_extra_convs=False):
                    pass
            
                def forward(self, inputs):
                    # implementation is ignored
                    pass
            ```
            
            1. Import the module
            
            ```python
            # 1번
            from .augfpn import AugFPN
            
            # 2번
            custom_imports = dict(
                imports=['mmdet.models.necks.augfpn.py'],
                allow_failed_imports=False)
            ```
            
            1. Modify the config file
            
            ```python
            neck=dict(
                type='AugFPN',
                in_channels=[256, 512, 1024, 2048],
                out_channels=256,
                num_outs=5)
            ```
            
            - 더 자세한 customize는 듀토리얼4,5를 참고해라
- **Prepare a config**
    - 현재 custom FPN을 만들었다. config 수정하자
    
    ```python
    _base_ = [
        '../_base_/models/cascade_mask_rcnn_r50_fpn.py', # 기존 config file
        '../_base_/datasets/cityscapes_instance.py',     # dataset
    		'../_base_/default_runtime.py'                   # runtime
    ]
    model = dict(
    		# backbone = dict(init_cfg = True??)인 경우 ImageNet pre-train 가져온다
    		# init_cfg = load_from 가져오면 coco로 pre-train 가져온다.
        backbone=dict(init_cfg=None),
    
    		# 수정 FPN 삽입
    		neck=dict(
    		        type='AugFPN',
    		        in_channels=[256, 512, 1024, 2048],
    		        out_channels=256,
    		        num_outs=5),
    
    		# bbox_head 와 mask head 부분에 num_class 수 변경-> dataset에 맞게
    		roi_head=dict(
    		        bbox_head=[
    		            dict(
    		                type='Shared2FCBBoxHead',
    		                in_channels=256,
    		                fc_out_channels=1024,
    		                roi_feat_size=7,
    								.....
    										num_classes=8
    						mask_head=dict(
    				            type='FCNMaskHead',
    				            num_convs=4,
    				            in_channels=256,
    				            conv_out_channels=256,
    				            # change the number of classes from defaultly COCO to cityscapes
    				            num_classes=8,
    				            loss_mask=dict(
    				                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)))
    ```
    
- **Train, test, and inference models on the standard dataset.**
    - 실행해보자
    
    ```python
    # train
    python tools/train.py configs/cityscapes/cascade_mask_rcnn_r50_augfpn_autoaug_10e_cityscapes.py
    
    # test & inference
    python tools/test.py configs/cityscapes/cascade_mask_rcnn_r50_augfpn_autoaug_10e_cityscapes.py work_dirs/cascade_mask_rcnn_r50_augfpn_autoaug_10e_cityscapes.py/latest.pth --eval bbox segm
    
    ```
