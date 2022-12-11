# MMDetection
##- MMDetection이란? 
- MMDetection은 Pytorch 기반의 Object Detection 오픈소스 라이브러리이다. 전체 프레임워크를 모듈 단위로 분리해 관리할 수 있다는 것이 가장 큰 특징이다.

## What for?
- 이미지에서 객체 탐지 → Back bone(CNN 으로 feature map) —> Head(예:SVM으로 분류)를 해야하는 본 프로젝트의 구조적 특징에 부합한다.

![image](https://user-images.githubusercontent.com/91417254/206924831-aefe48bc-1053-420a-ba81-1d8d495b58da.png)

## Is it Best?
- Not sure, But No any other that better one(still checking though 😋)

## Any points we might check?
- MMDetection의 config파일 형태
- 새로운 데이터셋 사용방법(조건: annotation형식이 coco형식을 그대로 따르되, 클래스 이름과 개수만 바뀐 경우)
- augmentation 변경 방법
- backbone, neck, head, loss 변경 방법
- optimizer 변경방법

# MMDetection 정리

## 1. 가상환경 & lib 설치
```
conda create -n 이름 python= 버젼 -y
conda activate open 이름 # 활성화

conda install pytorch(==버젼) torchversion(==버젼) -c pytorch 
	#pytorch 홈페이지로 가서 필요version+cuda 설치

git clone (--branc)(v2.3.0) 주소
cd mmdetection

# 필요 lib 설치
pip install -r requirements/build.txt
pip install -v -e .
```

## 2. Dataset format 변환

### a. COCO format
```
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
### b. middle format
```
[
            {
                'filename': 'a.jpg',
                'width': 1280,
                'height': 720,
                'ann': {
                    'bboxes': <np.ndarray> (n, 4) in (x1, y1, x2, y2) order.
                    'labels': <np.ndarray> (n, ),
                    'bboxes_ignore': <np.ndarray> (k, 4), (optional field)
                    'labels_ignore': <np.ndarray> (k, 4) (optional field)
                }
            },
            ...
        ]
```

## 3. Training
- MMDetection github → Getting Started→getting_started.md 설명 읽기
  - dataset 정의 읽기(tutorial2 : adding new data)
  - configs 파일 다양한 모델 → base →default_runtime.py 열기(커스텀)
    - dataset 정의
      - class, 경로,전처리 pipeline, evaluation(interval,metric=bbox)
    - model 정의
      - 모델의 구조 backbone+neck+head
      - head부분에 anchor → scale 수정 → 줄일수록 작은물체 잘 탐지
        - 큰물체 탐지 위해서 stride 조정
      - load_from에서 사전학습 모델 가중치 가져오기
        - mmlab github→model→download 링크 복사
    - schedule 정의
      - optimizer, step, warmup, epochs
        - !!!!!! lr 수정 mmdetection→getting_started.md→important lr설명: GPU 1 → 0.0025
        - epoch 수정
 
## 4. inference
    
