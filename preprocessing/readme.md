# Data Preprocessing
---
## 1. Class
- train & test에서 탐지할 class는 car,bus,truck,pedestrian으로 한정한다. 따라서 여타 class는 제거한다.
### 1.0 caution

-  annotation file이 read-only로 구성-> code로 접근이 불가능하다. 따라서 폴더에서 read-only 기능을 해제

### 1.1 Drop label & attribute

  
### 1.2 Class Define
```
def drop_feature(path):
    count = 0
    for file in os.listdir(path):
        if file.endswith('json'):
            id  = []
            with open(path+'/'+file, 'r', encoding="UTF-8") as annotations:
                anno = json.load(annotations)

                for index,annotation in enumerate(anno['annotations']):
                    if (annotation['label'] == 'face') or (annotation['label'] == 'pedestrian'):
                        annotation['class'] = 'pedestrian'
                    elif annotation['attribute'] == 'car':
                        annotation['class'] = 'car'
                    elif (annotation['attribute'] == 'bus_l') or(annotation['attribute'] == 'bus_s'):
                        annotation['class'] = 'bus'
                    elif (annotation['attribute'] == 'truck_l') or(annotation['attribute'] == 'truck_s'):
                        annotation['class'] = 'truck'
                    else:
                        annotation['class'] = 'other'
                   
            try:
                with open(path+'/'+file, 'w', encoding='utf-8') as ch_annotations:
                    json.dump(anno, ch_annotations, indent="\t")
            except:
                count+=1
    return count
```



## 2. Data format Convert
- MMDetection에서 standard dataset 이외의 dataset을 사용하기 위해서 정해진 COCO 또느 Middle data format으로 dataset을 변환해야 한다.
### 2.1 MMDetection Data Format
  1. COCO format
  ![image](https://user-images.githubusercontent.com/91417254/206921407-4e6b1935-f589-4be4-b7ff-a026bee18162.png)
  2. Middle
  ![image](https://user-images.githubusercontent.com/91417254/206921460-c06b22eb-7b0d-4e24-bccb-af088c85904b.png)
  
  - 해당 format으로 변환 방법
    1. offline : code를 직접 작성하여 custom dataset을 변환한다.
    2. online : @DATASETS.register_module(force=True)를 통해 pipeline에서 변환

