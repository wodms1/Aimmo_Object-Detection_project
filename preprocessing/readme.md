# Data Preprocessing
---
## Class
- train & test에서 탐지할 class는 car,bus,truck,pedestrian으로 한정한다. 따라서 여타 class는 제거한다.
### caution
![image](https://user-images.githubusercontent.com/91417254/206917544-60ad22f6-7044-41a9-917a-41d7cdcfb6e4.png)
-  annotation file이 read-only로 구성-> code로 접근이 불가능하다. 따라서 폴더에서 read-only 기능을 해제
![image](https://user-images.githubusercontent.com/91417254/206917584-89652f43-c68e-4a02-a343-310ad7d1c337.png)
![image](https://user-images.githubusercontent.com/91417254/206917589-b9b2d0aa-5557-48dd-b763-953e0f5e99e5.png)
- sunny/day annotation file : 23,342
  - total bbox : 362,428
  - preprocessing bbox : 212,623
  
![image](https://user-images.githubusercontent.com/91417254/206917692-41f2cc18-a69c-491c-9ffa-6d490590336c.png)
