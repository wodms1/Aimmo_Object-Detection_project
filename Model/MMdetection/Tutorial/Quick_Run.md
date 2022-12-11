- MMDetectiondms Model Zoo안에 존재하는 수십여가지 model을 지원하고 다수의 표준 dataset(**Pascal VOC, COCO, CityScapes, LVIS, etc.**)또한 지원합니다. 해당 page는 해당 데이터와 모델의 구동 과정을 아래를 포함하여 보입니다.
    1. Zoo Model 사용 inference with given images
    2. Zoo Model 사용 Test with standard datasets
    3. Zoo Model (Pre-defined) with standard datasets
- Faster R-CNN 사용 & config,checkpoint 준비
    - Config: Model 구성
    - Checkpoint: pre-trained weights bias
    
# 1. Inference with existing models
- High-level APIs for inference
  - must download checkpoint_file
  ```
  !wget -c https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
      -O checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
  ```
  ```
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
  
  # show the results
  show_result_pyplot(model, img, result)
  ```
- Test existing models on standard datasets
  - show **how to test**
  - folder structure는 아래와 같아야한다.
  ```
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
      2. **reorganize the dataset into a middle format.**
      3. **implement a new dataset.**
  - Prepare a config
    - Mask R-CNN with FPN 구현 → 풍성 dataset을 활용한 학습  &구현
```
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


# 3. Train with customized model
- **Prepare the standard dataset**
    - 경로 설정 
- Prepare your own customized model
  - custom module  or training setting 해라



