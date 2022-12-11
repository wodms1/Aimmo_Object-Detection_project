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

## 2. Aimmo Dataset
### 2.1 데이터 정의
<img width="449" alt="image" src="https://user-images.githubusercontent.com/101169092/206894223-f194b9ef-3eac-4a64-a37e-4ce561e8389a.png"> <img width="413" alt="image" src="https://user-images.githubusercontent.com/101169092/206894229-0764b988-f777-4f29-871a-91779b3d3a20.png">

- 메타 정보 : 주간, 맑음
- 레이블링 정보 : 차량, 버스, 트럭, 사람 등
- 데이터양 : 총 156,957(image : 78,484개, annotations : 78,463개)

### 2.2 EDA
#### 2.2 Sunny & Day Data EDA
- 총 78,464의 data 중에서 주간, 맑음에 해당하는 data는 23,342개 

#### 2.3 meta data EDA
- 주간 & 맑음에 해당하는 meta data는 총 362,428개

#### 2.4 annotations label
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

