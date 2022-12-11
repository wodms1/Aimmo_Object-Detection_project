# AIFFEL x AIMMO Project
- DL 기반 데이터 라벨링 서비스를 제공하는 Aimmo의 자율주행 학습용 Dataset을 활용한 Model-centric Object Detection.

# 1. 프로젝트 소개

## 1.1 개요
- 주제 : Model-Centric Object Detection AI
- 목적 : 제한적인 조건(주간/맑음 & 특정 class dataset)에서 추론 성능이 잘 나오는 모델을 합리적으로 찾아서 학습 및 테스트
- 기간 : 2022.11.01 ~ 2022.12.13
- 방식 : 팀프로젝트

## 1.2 프로젝트 목표
- 다양한 object detection model을 활용한 객체 탐지성능 최대화 및 추론 결과 논의

## 1.3 구성원
|이름|직책|역할|
|----|-----|-----|
|이재은|팀장|Data EDA & Preprocessing, modeling(faster-rcnn,yolox,fcos)|
|박철영|팀원|EDA, RCNN 및 YOLO 논문 리뷰|
|최훈성|팀원|modeling(yolox)|
|김명찬|팀원|modeling(yolo_v5)|
|윤동환|팀원|data format transform, modeling(yolo_v3)|

## 1.4 기술 스택
- jupyter notebook , mmdetection , WandB 

# 2. Data
## 2.1 Aimmo Dataset
![image](https://user-images.githubusercontent.com/91417254/206915913-96f3c3ee-5101-46b9-9c57-f850ede2f93c.png)
```
python
{
    "filename": "1660698683036_FR-View-CMR-Wide.png",
    "parent_path": "/batch_11/20220817/2022-08-17_10-11-24_ADCV1-ADS-LC1/FR-View-CMR-Wide",
    "unix_timestamp": 1660698683036,
    "file_format": "png",
    "capacity": 2.3,
    "vehicle_id": "AIMMO-ADCV1",
    "region_name": "인천 남동구",
    "location": "수산동 450-1",
    "length": 9,
    "framerate": 30,
    "size": "1920*1024",
    "data_purpose": "train",
    "weather": "sunny",
    "time": "day",
    "road_feature": "r_expressway",
    "road_type": "normal",
    "location_feature": "other",
    "driving_scenario": "lane_following",
    "ego_long_vel_level": "middle",
    "season": "summer",
    "illumination_status": "normal",
    "road_status": "dry",
    "crowd_level": "high",
    "scene_att": "frame",
    "sensor_name": "FR_View_CMR_Wide",
    "sensor_status": "normal",
    "sensor_hfov": 122,
    "sensor_vfov": 74,
    "gps_mode": "other",
    "gps_latitude": 37.433127531808736,
    "gps_longitude": 126.7339902817851,
    "long_velocity": 63.8,
    "lat_velocity": -0.04,
    "long_accel": -0.53,
    "lat_accel": 0.11,
    "yaw": 177.83,
    "roll": 0.28,
    "pitch": 2.96,
    "annotations": [
        {
            "id": "1-f21428d6-1cbe-4cc5-85e9-06d7013502fa",
            "type": "bbox",
            "label": "vehicle",
            "attribute": "truck_s",
            "points": [
                [
                    1578,
                    593
                ],
                [
                    1920,
                    593
                ],
                [
                    1920,
                    1024
                ],
                [
                    1578,
                    1024
                ]
            ],
            "trackId": -1,
            "occlusion": 0,
            "truncation": 3,
            "scenario": 0,
            "isfake": 0,
            "ismask": 0,
            "area": 147402
        },
        '''

```
- total dataset : 156,957
    - image : 78,494
    - annotation: 78,463
    - unpaired images: 31 

## 2.2 EDA
### 2.2.1 Sunny & Day Data 
- model_centric은 전체 dataset이 아닌 sunny/day datset만을 사용한다.
- 총 78,463의 annotation file 중 sunnay/day file은 23,342 

### 2.2.2 meta data EDA
![image](https://user-images.githubusercontent.com/91417254/206916500-2dc5bf04-42e3-41a6-bda2-91470cae8840.png)
![image](https://user-images.githubusercontent.com/91417254/206916507-90d67b37-f9b9-49f7-9549-80fa467f14af.png)

### 2.2.3 annotations EDA
![image](https://user-images.githubusercontent.com/91417254/206916630-6900541f-ed69-4e4e-aeab-ae1d85c58503.png)
- 23,342의 annotations file의 bbox 수량은 362,428

![image](https://user-images.githubusercontent.com/91417254/206916675-6ac4a141-e8d7-422d-8760-a3765fadc48e.png)
![image](https://user-images.githubusercontent.com/91417254/206916894-f8521fc5-ceb5-4aa3-9963-ab7b3049f7e0.png)
![image](https://user-images.githubusercontent.com/91417254/206916909-7882ccfd-f2d4-4dcb-a8be-d7f40195de17.png)

![image](https://user-images.githubusercontent.com/91417254/206916924-a543a434-73d6-425b-917f-87afeb7cef39.png)
![image](https://user-images.githubusercontent.com/91417254/206916932-2f900822-1a4a-421a-987f-e552e7a83fd1.png)
![image](https://user-images.githubusercontent.com/91417254/206916940-60336338-b625-40d8-b355-dfc3f8a9bcbc.png)

- Objcet의 Class(정답)에 해당하는 label과 attribute(상세 label) 모두 class unbalance가 존재한다.

![image](https://user-images.githubusercontent.com/91417254/206917009-37766f88-e8e7-4b21-a66c-e8977295146d.png)
![image](https://user-images.githubusercontent.com/91417254/206917066-8f1753ea-9c57-48d2-aee2-6916e0b9efba.png)
- 625 pixel size 이하의 bbox(번호판,차량 )가 약 30% 존재한다


## 2.3 Data Preprocessing
### 2.3.1 Sunny & Day dataset
- 전체 dataset이 아닌 한정된(주간/맑음) dataset 사용
    - 78,463 -> 23,342 
### 2.3.2 Class
- 전체 label & attribute가 아닌 한정된 class 사용
- car,bus,truck,pedestrian class 사용

![image](https://user-images.githubusercontent.com/91417254/206923640-64b8c168-978a-4f7b-a850-d736891a2bb8.png)

