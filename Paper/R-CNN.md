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
