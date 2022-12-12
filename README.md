# AIFFEL x AIMMO Project
- DL 기반 데이터 라벨링 서비스를 제공하는 Aimmo의 자율주행 학습용 Dataset을 활용한 Model-centric Object Detection.

# 1. 프로젝트 소개

## 1.1 개요
- 주제 : Model-Centric Object Detection AI
- 목적 : 제한적인 조건(주간/맑음 & 특정 class dataset)에서 추론 성능이 잘 나오는 모델을 합리적으로 찾아서 학습 및 테스트
- 기간 : 2022.11.01 ~ 2022.12.13
- 방식 : 팀 프로젝트

## 1.2 프로젝트 목표
- 다양한 object detection model을 활용한 객체 탐지성능 최대화 및 추론 결과 논의

## 1.3 구성원
|이름|직책|역할|
|----|-----|-----|
|이재은|팀장|Data EDA & Preprocessing, modeling(faster-rcnn,yolo_v3,yolox,fcos)|
|박철영|팀원|EDA, RCNN 및 YOLO 논문 리뷰|
|최훈성|팀원|modeling(yolox)|
|김명찬|팀원|modeling(yolo_v5)|
|윤동환|팀원|data format transform, modeling(yolo_v3)|

## 1.4 기술 스택
- jupyter notebook , mmdetection , WandB 

# 2. Data
## 2.1 Aimmo Dataset


## 2.2 EDA
### 2.2.1 Sunny & Day Data 

### 2.2.2 meta data EDA


### 2.2.3 annotations EDA




- Objcet의 Class(정답)에 해당하는 label과 attribute(상세 label) 모두 class unbalance가 존재한다.




## 2.3 Data Preprocessing
### 2.3.1 Sunny & Day dataset
- 전체 dataset이 아닌 한정된(주간/맑음) dataset 사용
    - 78,463 -> 23,342 
### 2.3.2 Class
- 전체 label & attribute가 아닌 한정된 class 사용
- car,bus,truck,pedestrian class 사용


### 2.3.2 Data Format
- MMDetection training & inference 를 위한 Data format converting
    1. COCO format
        ![image](https://user-images.githubusercontent.com/91417254/206923719-25fe3245-e6ac-45ef-91e6-30e367783a2d.png)
    2. Middle format
        ![image](https://user-images.githubusercontent.com/91417254/206923745-910cc39c-5fe9-43d7-a802-43246b460c78.png)


## 3. Object Detection Paper
### 3.1 R-CNN
- R-CNN 원문을 읽고 서로 논문 리뷰를 통하여 Object Detection에 대한 기본 개념 학습
- 학습한 개념
    - R-CNN 구조
    - IOU
    - mAP
    - 1-stage, 2-stage
    - Selective Search
    - Warp & Crop 차이점
    - transfer learning
        - pre-training
        - fine-tuning
    - Negative Mining 

## 4. Model
### 4.1 MMDetection
[mmdetection tutorial 학습](https://github.com/wodms1/Aimmo_Object-Detection_project/tree/master/Model/MMdetection/Tutorial)

# Faster R-CNN
|file_name|fulldata|backbone|epoch|img_size |anchor_box_scale|anchor_box_stride|best_val_mAP|best_test_mAP|
|----|-----|-----|-----|-----|-----|-----|-----|-----|
|default|x|resnet 50|50|(960,512)|default|default|0.877|0.682|
|fulldata|o|resnet 50|15|(960,512)|default|default|0.877|0.748|
|fulldata_fullsize|o|resnet 50|15|(960,960)|default|default|0.877|0.740|
|anchor_box_scale|o|resnet 50|15|(960,512)|1/2|default|X|X|
|anchor_box_scale_stride|o|resnet 50|15|(960,512)|1/2|1/2|0.873|0.327|
|fulldata_r101|o|resnet 101|15|(960,512)|default|default|0.909|0.793|
- default: faster-rcnn은 10,000개의 data를 사용하여 학습,검증,훈련을 진행하였다. 적당한 val mAP가 나왔지만 test mAP가 크게 떨어진다.
- fulldata : 약21,000의 train data와 2,300의 val data를 사용하여 학습을 진행하였다. data의 증가로 인하여 test mAP가 6% 상승하였다.
- fulldata_fullsize: 전체 dataset을 활용하고 image resize를 960,960으로 진행한다. 이는 keep_ratio로 짧은 image 부분이 padding으로 채워지며 detection을 직사각형으로 진행할 것으로 예측하였으나 test mAP는 더 떨어졌다.
- anchor_box_scale_stride: anchor box의 scale과 stride를 각각 절반으로 줄인뒤 학습을 하였다. 예상 결과는 더욱 작은 물체를 잘 탐지 할 것이라 예측하였지만 bus와 truck를 전혀 탐지하지 못하는 결과를 초래하였다.
- fulldata_r101 : backbone network를 resnet 101로 변경한 뒤 전체 데이터로 학습을 진행하였다. 결과는 가장 좋은 test mAP가 나오게 되었다.


# YOLO v3
|file_name|fulldata|backbone|epoch|img_size |anchor_box_scale|anchor_box_stride|best_val_mAP|best_test_mAP|
|----|-----|-----|-----|-----|-----|-----|-----|-----|
|default|x|darknet|15|(960,960)|default|default|0.845|X|
|mixed_precision_training|o|darknet|15|(960,960)|default|default|0.828|X|
- yolo v3의 경우 val에 대한 mAP가 나오지만 test에 대한 mAP가 0으로 나온다. 이는 코드상의 에러로 보인다. 추후 수정

# YOLO v5
|file_name|fulldata|fl_gamma(focal loss gamma)|epoch|img_size|test data(clear day)|best_val_mAP|best_test_mAP|
|-----|-----|-----|-----|-----|-----|-----|-----|
|default|x|0.0|30|(1920,1024)|x|0.403|x|
|default|o|0.0|28|(960,512)|x|0.717|0.717|
|default|o|0.0|28|(960,512)|o|0.717|0.742|
|Focal_loss_1.0|o|1.0|20|(960,512)|o|0.709|0.733|
|Focal_loss_2.0|o|2.0|20|(960,512)|o|0.699|0.683|
|Augmentation|o|1.0|43|(960,512)|o|0.0005311|0.64|

- default 값은 pretrain 된 yolov5l 모델을 사용했다. 성능 개선을 위해 상위버전인 yolov5xl를 써보려고 했으나, 메모리 문제로 실패했다.
우선, sample 데이터로 모델 구동이 가능한지 실험을 했다. 성공 후, 전체 데이터로 훈련시켰다. 그 후, 전체 Test 파일과, 주간 맑음 파일로 테스트를 진행했다.

- 성능개선을 위하여, Loss 모델을 변경했다. Yolov5 기본 loss 모델은 Cross_entropy이다. 하지만, 하이퍼 파라미터 파일에서 fl_gamma 값을 조정하면, focal_loss 모델을 이용할 수 있었다. 실험은 fl_gamma값을 각각 1.0, 2.0으로 실험을 진행했다. 결과는 동일한 epoch 기준 1.0값이 제일 좋은 성능을 보였다.

- mAP 값을 고득점 받기 위한 생각을 했을 때, 가장 먼저 떠오르는 생각은 데이터 양을 늘리는 것이다. 이 가설을 위해 augmentation 기법을 이용하여 실험했다. 또한, 더 좋은 성능을 보였던 focalloss을 이용했다. 

# YOLO X
|file_name|fulldata|backbone|epoch|img_size |best_val_mAP|best_test_mAP|
|----|-----|-----|-----|-----|-----|-----|
|yolox-L|o|CSPDarknet|12|(640,640)|0.768|0.626|
- yolox-l은 상당히 무거운 모델로 1epoch 훈련에 4시간이 소요되었다. 2020년 논문에 대량의 파라미터로 조금 더 개선된 mAP를 기대하였지만 좋지 못한 성능이다.

# FCOS
|file_name|fulldata|backbone|epoch|img_size |best_val_mAP|best_test_mAP|
|----|-----|-----|-----|-----|-----|-----|
|fcos|o|resnet 50|12|(960,512)|0.851|0.378|
- fcos는 anchor-free방식으로 object detection의 새로운 트렌드이다. train에서는 적당한 mAP 성능을 보이지만 test에 대해서는 상당히 낮은 mAP를 보인다. 이는 모델이 과적합 되었다는 생각이든다. 이를 해결하기 위해 모델의 복잡성을 낮추거나 데이터를 증량하는 방식으로 실험을 진행한다. 

