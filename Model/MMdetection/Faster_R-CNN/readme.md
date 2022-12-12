|file_name|fulldata|backbone|epoch|img_size |anchor_box_scale|anchor_box_stride|best_val_mAP|best_test_mAP|
|----|-----|-----|-----|-----|-----|-----|-----|-----|
|default|x|resnet 50|50|(960,512)|default|default|0.877|0.682|
|fulldata|o|resnet 50|15|(960,512)|default|default|0.877|0.748|
|fulldata_fullsize|o|resnet 50|15|(960,960)|default|default|0.877|0.740|
|anchor_box_scale|o|resnet 50|15|(960,512)|1/2|default|X|X|
|anchor_box_scale_stride|o|resnet 50|15|(960,512)|1/2|1/2|0.873|0.327|
|fulldata_r101|o|resnet 101|15|(960,512)|default|default|0.909|0.793|


```
---------------iou_thr: 0.5---------------

+------------+------+------+--------+-------+
| class      | gts  | dets | recall | ap    |
+------------+------+------+--------+-------+
| car        | 2057 | 2968 | 0.906  | 0.888 |
| truck      | 911  | 1654 | 0.867  | 0.801 |
| bus        | 319  | 645  | 0.893  | 0.819 |
| pedestrian | 549  | 1180 | 0.769  | 0.663 |
+------------+------+------+--------+-------+
| mAP        |      |      |        | 0.793 |
+------------+------+------+--------+-------+
OrderedDict([('AP50', 0.793), ('mAP', 0.7929246425628662)])
```
