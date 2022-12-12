# 1.EDA 

<details>
<summary>EDA 정의</summary>
<div markdown="1">       

## 1) 정의
  - 수집한 데이터가 들어왔을 때, 이를 다양한 각도에서 관찰하고 이해하는 과정입니다. 한마디로 데이터를 분석하기 전에 그래프나 통계적인 방법으로 자료를 직관적으로 바라보는 과정입니다.

## 2) 필요 이유
  1.  데이터의 분포 및 값을 검토함으로써 데이터가 표현하는 현상을 더 잘 이해하고, 데이터에 대한 **잠재적인 문제를 발견**할 수 있습니다. 이를 통해, 본격적인 분석에 들어가기에 앞서 데이터의 수집을 결정할 수 있습니다.
  2. 다양한 각도에서 살펴보는 과정을 통해 문제 정의 단계에서 미쳐 발생하지 못했을 다양한 패턴을 발견하고, 이를 바탕으로 기존의 가설을 수정하거나 **새로운 가설**을 세울 수 있습니다.

## 3) 과정
  - 기본적인 출발점은 문제 정의 단계에서 세웠던 연구 질문과 가설을 바탕으로 분석 계획을 세우는 것입니다. 분석 계획에는 어떤 속성 및 속성 간의 관계를 집중적으로 관찰해야 할지, 이를 위한 최적의 방법은 무엇인지가 포함되어야 합니다.
      1. 데이터를 전체적으로 살펴보기 : 데이터에 문제가 없는지 확인. head나 tail부분을 확인, 추가적으로 다양한 탐색(이상치, 결측치 등을 확인하는 과정)
      2. 데이터의 개별 속성값을 관찰 : 각 속성 값이 예측한 범위와 분포를 갖는지 확인. 만약 그렇지 않다면, 이유가 무엇인지를 확인.
      3. 속성 간의 관계에 초점을 맞추어, 개별 속성 관찰에서 찾아내지 못했던 패턴을 발견 (상관관계, 시각화 등)
      
</div>
</details>

# 2. EDA 개요
<details>
<summary>Aimmo Dataset sample</summary>
<div markdown="1">  

## 1) image file


## 2) annotation file



```
</div>
</details>

## Number of Dataset
- 총 file의 수는 156,957
  - image :78,494
  - annotation: 78,464
    - annotation이 없는 image 31개가 존재한다.
    
## Annotation Analysis
```
python

# pd.json_normalize -> json file을 datafream으로 생성하도록 돕는 함수
data = pd.json_normalize(annotations)

# 전체 feature가 아닌 특정feature만 사용


# sunny & day dataset만 사용

```

- 전체 annotation이 아닌 sunny & day annotation file 분석
  - meta data에서 불필요한 feature 제거
  - 전체 annotation file : 78,463
  - sunny & day annotation file: 23,342

### Meta Data Analysis
- 여타의 meta data feature에서 insight를 발견하지 못하였다.

### Bounding Box(Annotations) Anaysis
- sunny & day인 image-annotation 쌍은 총 362,428이며 annotation file 내부에 annotations는 13개의 feature를 가진다.
  
  - label은 bbox의 class(정답)이며 vehicle이 약 50%로 14개의 가짓수 중 가장 많은 비율을 차지한다. 
    - Model의 성능을 개선하기 위해서 vehicle을 제외한 label data를 augumentation이 하거나 vehicle을 일부분 제거하는것이 좋아보인다.



  - 25 X 25 pixel 이하의 차량은 자율주행 관점의 object detection에서 학습에 방해요소이디.
    - 625 pixel 이하의 bbox를 분석한다.


# EDA 요약
1. project에 사용될 image-annotation file pair는 23,342이다.
2. image의 file format은 png로 동일하며 image의 shape도 (1920,1024)로 유일하다.
3. meta data에서는 model 성능 개선을 위한 insight가 없다.
4. 총 bbox의 개수는 362,248이다.
5. class에 해당하는 label과 attribute의 class unbalance가 차량에 집중적으로 몰려있다.
  - 모델을 개선 방안
    - Oversampling: 차량을 제외한 class의 augmentation
    - Undersampling: 차량 bbox의 일부를 제거한다.
6. data 상에서 625 pixel 이하의 object는 약 90,000으로 약 30%이다.
