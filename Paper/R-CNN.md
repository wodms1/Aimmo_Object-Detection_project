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
