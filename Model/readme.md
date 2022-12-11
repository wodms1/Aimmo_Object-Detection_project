# Faster R-CNN
|file_name|fulldata|backbone|epoch|img_size |anchor_box_scale|anchor_box_stride|best_val_mAP|best_test_mAP|
|----|-----|-----|-----|-----|-----|-----|-----|-----|
|default|x|resnet 50|50|(960,512)|default|default|0.877|0.682|
|fulldata|o|resnet 50|15|(960,512)|default|default|0.877|0.748|
|fulldata_fullsize|o|resnet 50|15|(960,960)|default|default|0.877|0.740|
|anchor_box_scale|o|resnet 50|15|(960,512)|1/2|default|X|X|
|anchor_box_scale_stride|o|resnet 50|15|(960,512)|1/2|1/2|0.873|0.327|
|fulldata_r101|o|resnet 101|15|(960,512)|default|default|0.909|0.643|

# YOLO v3
|file_name|fulldata|backbone|epoch|img_size |anchor_box_scale|anchor_box_stride|best_val_mAP|best_test_mAP|
|----|-----|-----|-----|-----|-----|-----|-----|-----|
|default|x|darknet|15|(960,960)|default|default|0.845||
|mixed_precision_training|o|darknet|15|(960,960)|default|default|0.828||

# YOLO v5

# YOLO X
|file_name|fulldata|backbone|epoch|img_size |best_val_mAP|best_test_mAP|
|----|-----|-----|-----|-----|-----|-----|
|yolox-L|o|CSPDarknet|12|(640,640)|0.768||

# FCOS
|file_name|fulldata|backbone|epoch|img_size |best_val_mAP|best_test_mAP|
|----|-----|-----|-----|-----|-----|-----|
|fcos|o|resnet 50|12|(960,512)|0.851||
