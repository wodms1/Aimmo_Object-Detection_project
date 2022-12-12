# Faster R-CNN
|file_name|fulldata|backbone|epoch|img_size |anchor_box_scale|anchor_box_stride|best_val_mAP|best_test_mAP|
|----|-----|-----|-----|-----|-----|-----|-----|-----|
|default|x|resnet 50|50|(960,512)|default|default|0.877|0.682|
|fulldata|o|resnet 50|15|(960,512)|default|default|0.877|0.748|
|fulldata_fullsize|o|resnet 50|15|(960,960)|default|default|0.877|0.740|
|anchor_box_scale|o|resnet 50|15|(960,512)|1/2|default|X|X|
|anchor_box_scale_stride|o|resnet 50|15|(960,512)|1/2|1/2|0.873|0.327|
|fulldata_r101|o|resnet 101|15|(960,512)|default|default|0.909|0.793|
- default: faster-rcnn은 10,000개의 data를 사용하여 학습,검증,훈련을 진행하였다. 적당한 val mAP가 나왔지만 test mAP가 크게 떨어진다.
- fulldata : 약21,000의 train data와 2,300의 val data를 사용하여 학습을 진행하였다. data의 증가로 인하여 test mAP가 6% 상승하였다.
- fulldata_fullsize: 전체 dataset을 활용하고 image resize를 960,960으로 진행한다. 이는 keep_ratio로 짧은 image 부분이 padding으로 채워지며 detection을 직사각형으로 진행할 것으로 예측하였으나 test mAP는 더 떨어졌다.
- anchor_box_scale_stride: anchor box의 scale과 stride를 각각 절반으로 줄인뒤 학습을 하였다. 예상 결과는 더욱 작은 물체를 잘 탐지 할 것이라 예측하였지만 bus와 truck를 전혀 탐지하지 못하는 결과를 초래하였다.
- fulldata_r101 : backbone network를 resnet 101로 변경한 뒤 전체 데이터로 학습을 진행하였다. 결과는 가장 좋은 test mAP가 나오게 되었다.


# YOLO v3
|file_name|fulldata|backbone|epoch|img_size |anchor_box_scale|anchor_box_stride|best_val_mAP|best_test_mAP|
|----|-----|-----|-----|-----|-----|-----|-----|-----|
|default|x|darknet|15|(960,960)|default|default|0.845|X|
|mixed_precision_training|o|darknet|15|(960,960)|default|default|0.828|X|
- yolo v3의 경우 val에 대한 mAP가 나오지만 test에 대한 mAP가 0으로 나온다. 이는 코드상의 에러로 보인다. 추후 수정

# YOLO v5

# YOLO X
|file_name|fulldata|backbone|epoch|img_size |best_val_mAP|best_test_mAP|
|----|-----|-----|-----|-----|-----|-----|
|yolox-L|o|CSPDarknet|12|(640,640)|0.768|0.626|
- yolox-l은 상당히 무거운 모델로 1epoch 훈련에 4시간이 소요되었다. 2020년 논문에 대량의 파라미터로 조금 더 개선된 mAP를 기대하였지만 좋지 못한 성능이다.

# FCOS
|file_name|fulldata|backbone|epoch|img_size |best_val_mAP|best_test_mAP|
|----|-----|-----|-----|-----|-----|-----|
|fcos|o|resnet 50|12|(960,512)|0.851|0.378|
- fcos는 anchor-free방식으로 object detection의 새로운 트렌드이다. train에서는 적당한 mAP 성능을 보이지만 test에 대해서는 상당히 낮은 mAP를 보인다. 이는 모델이 과적합 되었다는 생각이든다. 이를 해결하기 위해 모델의 복잡성을 낮추거나 데이터를 증량하는 방식으로 실험을 진행한다. 
