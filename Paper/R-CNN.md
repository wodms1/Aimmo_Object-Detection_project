# Bone
## 1. First: generates category-independent region proposals
## 2. Second: Large CNN that extracts a fixed-length feature vector from each region
## 3. Third: set of class-specific linear SVMs
![image](https://user-images.githubusercontent.com/91417254/206923902-755c0fdb-fc12-4ca9-9c61-63e9acc0d3c6.png)

# Muscle
## First
### Generates category-independent region proposals
- generate around 2000 region proposals for each image
- using selective search algorithm
  -selective search : is a region proposal algorithm for object detection tasks.
    1. 후보 region을 생성한다.
    2. 가장 유사도가 높은 region간의 통합을 반복한다.
    3. 최종적으로 region proposals가 생성된다.
    
    ![image](https://user-images.githubusercontent.com/91417254/206923964-b324a854-2f59-49b4-bdb1-e35662287af2.png)
    ![image](https://user-images.githubusercontent.com/91417254/206923972-0136caf8-8956-4932-9cae-f9b3cbad9d8b.png)

## Second 
### Large CNN that extracts a fixed-length feature vector from each region
1. Warp
  - we must first convert  region size into a form that is compatible with the CNN
    - Prior to warping, we dilate the tight bounding box so that at the warped size there are exactly 16 pixels 
  - reson for warp
    - its architecture requires inputs of a fixed 227 x 227 pixel size  
  ![image](https://user-images.githubusercontent.com/91417254/206924129-9e82cb31-10a3-4759-961f-104bf117b858.png)

2. transfer learning
  - pre-training: pre-trained the CNN(AlexNet/VGG16) on a large auxiliary dataset(ILSVRC 2012 classification)
  - fine-tuning:  using only warped region proposals update(SGD) CNN’s parameters
    - CNN’s replacing
      - classification layer(last Fully Connected Layer (fc7)
        - 1000 classes(number of ImageNet class) -> with a randomly initialized N+1 (number of PASCAL VOC classes + background)
    - training
      - if region proposals and ground-truth
        - IOU≥0.5 →  positive
        - IOU< → negative
      - 32 positive, 96 background windows → 128 batch size   
        - (positive windows are rare compare to background)
        - In each SGD iteration ( mini-batch of size 128)

```
Transfer learning(전이학습)
- 정의:한 분야의 문제를(downstream task) 해결하기 위해서 다른 분야(pre-train task)
에서얻은 지식을 활용하는 방식 
    -  Pre-trained task : solved task ex) ImageNet
    -  Downstream task :  target task ex) PASCAL VOC

pre-training(사전학습)
- 정의: parameters의 초기값을 정하는 방법으로 pre-trained task dataset으로 학습한다.

fine-tuning(미세 조정)
- 정의 : 사전학습으로 학습된 모델(가중치)를 새로운 분야(탐지) 및 영역에 맞게 변환하는 방법
````
3. Extract feature vectors of region proposals 
  - extract a 4096-dimensional feature vector from each region proposal using CNN(AlexNet)
    - feature extract
      - fine-tuning
        - 5 conv layer + 2 FC layer (N+1)
      - test
        - 5 conv. layers & 2 FC layer(4096)
    - output → (2000,4096) matrix


## Third
### Set of classes specific linear SVM
1. result
  - Positive : image region tightly enclosing a object
    - if region proposals and ground truth IOU>0.3  than positive 
  - Negative: background region which has nothing to do with object
    - if region proposals and ground truth IOU≤0.3  than negative
  - Less Clear: region that partially overlaps a object
    - Why IoU threshold 0.3?
      - The overlap threshold, 0.3,was selected by a grid search over {0, 0.1, . . . , 0.5} on a validation set. 
2. SVM’s processing
  - optimize one linear SVM per class.
  - using hard negative mining
    - why using hard negative mining?
      - training data is too large to fit in memory
    - IoU(if there is a lot of overlapping part with the ground truth → an area displaying an object, and in the opposite case → a background independent of the object)
```
💡 Negative Mining

hard negative mining는 hard negative 데이터(원래 negative 인데 positive라고 잘못 예측한 데이터)를 학습 데이터로 사용하기 위해 모으는(mining) 것이다.

hard negative mining으로 얻은 데이터를 원래의 데이터에 추가해서 재학습하면  false positive 오류에 강해진다.


```
![image](https://user-images.githubusercontent.com/91417254/206924395-84db7b30-72ce-4ea6-8dc7-ebe98e46a685.png)
![image](https://user-images.githubusercontent.com/91417254/206924399-1a547779-abfa-4ffb-88bd-2913f38ab74e.png)

## Appendix
### Positive vs negative examples and softmax
- pre-trained CNN (regoin proposal)
  - threshold ≥ 0.5 IoU → positive 
  - threshold < 0.5 IoU → negative 
- SVM 
  - threshold ≥ 0.3 IoU → positive(Only Ground Truth)  
  - threshold < 0.3 IoU → negative(Background)
  - Proposal that fall into the grey zone are ignored
    - gray zone
      -  more than 0.3 IoU overlap, but are not ground truth  
