- pre-trained model이 다른 dataset에 좋은 역할한다. 이번 듀토리얼은 Model Zoo에 존재하는 model을 활용하여 다른 dataset에 더 좋은 성능을 활용하는 지침을 알려준다.
- new dataset finetune 2 step
    1. Turorial2를 따라 dataset 변환
    2. config file modify

- 파인 튜닝 과정을 위한 cityscape 예시 5단계 config 수정
    - **Inherit base configs**
        - config 상속 중요!
            - 전체 config 작성에 대한 부담과 bug를 줄이기 위해서 mmd v 2.0은 상속을 지원한다.
            - new config file로 fine-tuning 하기 위해서 기존 model의 config file 상속이 필요하다.
            - dataset 또한 내장되어 있을경우 상속한다.
            - runtime setting(ex training schedule) 또한 상속
            - 이러한 config file들은 **configs** 디렉토리안에 존재한다. 따라서 상속 말고도 직접 작성 및 사용 가능하다.
            
            ```python
             _base_ = [
                '../_base_/models/mask_rcnn_r50_fpn.py',
                '../_base_/datasets/cityscapes_instance.py', 
            		'../_base_/default_runtime.py'
            ]
            ```
            
    - **Modify head**
        - 새로운 config 만들었으면 head(num_classes)를 수정해야한다.
            - roi-head는 클래스의 숫자만 바뀌기에 마지막 prediction head 제외하고는 weight 그대로 사용한다.
            
            ```python
            model = dict(
                pretrained=None,
                roi_head=dict(
                    bbox_head=dict(
                        type='Shared2FCBBoxHead',
                        in_channels=256,
                        fc_out_channels=1024,
                        roi_feat_size=7,
                        num_classes=8,
                        bbox_coder=dict(
                            type='DeltaXYWHBBoxCoder',
                            target_means=[0., 0., 0., 0.],
                            target_stds=[0.1, 0.1, 0.2, 0.2]),
                        reg_class_agnostic=False,
                        loss_cls=dict(
                            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
                    mask_head=dict(
                        type='FCNMaskHead',
                        num_convs=4,
                        in_channels=256,
                        conv_out_channels=256,
                        num_classes=8,
                        loss_mask=dict(
                            type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))))
            ```
            
    - **Modify dataset**
        - dataset 준비하고 해당 dataset에 대한 config 작성해야한다.
    - **Modify training schedule**
        - finetuning hyperparameters은 default schedule과 다르다. 작은 lr,epoch 를 원한다.
        
        ```python
        # optimizer
        # lr is set for a batch size of 8
        optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
        optimizer_config = dict(grad_clip=None)
        # learning policy
        lr_config = dict(
            policy='step',
            warmup='linear',
            warmup_iters=500,
            warmup_ratio=0.001,
            step=[7])
        # the max_epochs and step in lr_config need specifically tuned for the customized dataset
        runner = dict(max_epochs=8)
        log_config = dict(interval=100)
        ```
        
    - **Use pre-trained model**
        - **load_from**을 통해 해당 link와 새로운 config를 통해 pre-train model을 사용한다.
        - user는 model의 weight를 학습전에 다운로드를 추천한다 → 시간때문
