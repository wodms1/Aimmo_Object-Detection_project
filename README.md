# AIFFEL x AIMMO Project
- DL 기반 데이터 라벨링 서비스를 제공하는 Aimmo의 자율주행 학습용 Dataset을 활용한 Model-centric Object Detection.

## 1. 프로젝트 소개

### 1.1 개요
- 주제 : Model-Centric Object Detection AI
- 기간 : 2022.11.01 ~ 2022.12.13
- 방식 : 팀프로젝트

### 1.2 프로젝트 목표
- mmdetection을 활용하여 여러 모델을 구현해보고 각각의 하이퍼 파라미터를 바꿔가면서 보다 좋은 성능을 가진 모델을 찾아낸다.

### 1.3 구성원
|이름|직책|역할|
|----|-----|-----|
|이재은|팀장|Data EDA & Preprocessing, modeling(faster-rcnn,yolox,fcos)|
|박철영|팀원|EDA, RCNN 및 YOLO 논문 리뷰|
|최훈성|팀원|modeling(yolox)|
|김명찬|팀원|modeling(yolo_v5)|
|윤동환|팀원|data format transform, modeling(yolo_v3)|

### 1.4 기술 스택
- jupyter notebook , mmdetection 

## 2. Data
### 2.1 Aimmo Dataset
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

### 2.2 EDA
#### 2.2.1 Sunny & Day Data 
- model_centric은 전체 dataset이 아닌 sunny/day datset만을 사용한다.
- 총 78,463의 annotation file 중 sunnay/day file은 23,342 

#### 2.2.2 meta data EDA
![image](https://user-images.githubusercontent.com/91417254/206916500-2dc5bf04-42e3-41a6-bda2-91470cae8840.png)
![image](https://user-images.githubusercontent.com/91417254/206916507-90d67b37-f9b9-49f7-9549-80fa467f14af.png)


#### 2.2.3 annotations label
<img width="428" alt="image" src="https://user-images.githubusercontent.com/101169092/206896042-4d064589-7cc8-4eb5-b86f-9be99ff34c29.png"> <img width="498" alt="image" src="https://user-images.githubusercontent.com/101169092/206896054-301536b4-1fe1-47f9-b898-22a95c807060.png">


#### 2.5 annotations attribute
<img width="1088" alt="image" src="https://user-images.githubusercontent.com/101169092/206895958-8c3cf23d-c273-46f1-82f7-cfb3aee985df.png">

#### 2.6 points & area
- area의 크기가 625 이하의 경우 detection 성능에 도움이 되지 않기 때문에 제거하였음

points

<img width="158" alt="image" src="https://user-images.githubusercontent.com/101169092/206896138-9905873b-943f-4bcb-bd5b-ca0b8eec2a2f.png">

area

<img width="230" alt="image" src="https://user-images.githubusercontent.com/101169092/206896158-130016ba-aeca-4c09-a3fa-7c7a83296350.png">

### 3. Data Preprocessing
#### 3.1 주간 & 맑음 dataset 정의

#### 3.2 파일명 & 확장자명 변경
<img width="1006" alt="image" src="https://user-images.githubusercontent.com/101169092/206895108-c6f7899f-fd3f-4924-bffc-8f1f98e8c97c.png">

#### 3.3 특정 annotations 제거
- 362,428 개의 bbox가 212,623개로 줄었음
- <img width="324" alt="image" src="https://user-images.githubusercontent.com/101169092/206895153-3617f82c-83fc-45a4-8bd9-07c73be11bbc.png">

#### 3.4 Class 정의
<img width="161" alt="image" src="https://user-images.githubusercontent.com/101169092/206895159-72f30b8c-262a-462c-964f-cbe88fbf9944.png">
- car : 129,702 bus : 17,989 truck : 30,509 human : 31,405 other : 3,018

### 4. Data format convert
- mmdetection에서 dataset을 사용하기 위해서는 CoCo dataset format 또는 Middle dataset format으로 변경해야 함

