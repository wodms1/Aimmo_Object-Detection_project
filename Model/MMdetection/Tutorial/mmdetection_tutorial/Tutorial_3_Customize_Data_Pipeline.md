- **Design of Data pipelines**
    - **Dataset** , **DataLoader** 사용 data loading, data 같은 크기 아닐 수 있다.
    - dataset & data pipeline 분해된다.
    - pipelie은 dict이 input → dict가 output 된다.
    - pipeline color
        - Blue Block: pipeline block
        - green : add new keys
        - orange : result dict or update key
        - dark : no change
    
    ![image](https://user-images.githubusercontent.com/91417254/206927084-e92b4deb-4d29-4053-ac98-61ced3866530.png)
    
    - whole pipeline
        - data loading → pre-processing → (test time augmentation)→formatting
        1. Data Loading: loadimagefromfile,loadannotations,loadproposals
        2. Pre-Processing: Resize, RandomFilp, Pad, RandomCrop,Normalize,SegRescals,Expand, MinIOURandomCrop,Corrupt
        3. Formatting: ToTensor, ImageToTensor, TransPose, ToDataContainer, DefaultFormetaBundle,Collect
        4. Test time augmentation: MultiScaleFlipAug
        
        - whole pipeline example
            - LoadImageFromFile: 이미지 로드
                - green: img , img_shape, ori_shape
            - LoadAnnotations: 어노테이션 로드
                - green: gt_bboxes, gt_labels, bbox_fields
                - dark: img , img_shape, ori_shape
            - Resize: 크기 변환
                - green: pad_shape, scale, scale_idx ,scale_factor, keep_ratio
                - orange: img , img_shape,**gt_bboxs → 여기서 좌표값도 변환됨**
                - dark: ori_shape, gt_labels, bbox_fields
            - RandomFilp: 뒤집기
                - green: filp
                - orange: img ,gt_bboxes
                - dark: img_shape, ori_shape, pad_shape, gt_labels, bbox_fields, scale, scale_idx, scale_factor, keep_ratio
            - Normalize: 정규화
                - green: img_norm_cfg
                - orange: img
                - dark : img_shape, ori_shape, pad_shape,gt_boxes,  gt_labels, bbox_fields, scale, scale_idx, scale_factor, keep_ratio, flip
            - Pad: 둘러싸기
                - green: pad_fixed_size, pad_size_divisor
                - orange: img, pad_shape
                - dark : img_shape, ori_shape,gt_boxes,  gt_labels, bbox_fields, scale, scale_idx, scale_factor, keep_ratio, flip,img_norm_cfg
            - DefaultFormatBundle: 묶기
                - orange: img, gt_bboxes, gt_labels
                - dark: img_shape, ori_shape, bbox_fields, scale, scale_idx, scale_factor, keep_ratio, flip,img_norm_cfg, pad_fixed_size, pad_size_divisor
            - Collect: 수집
                - green: img_meta
                - dark: img, ori_shape, img_shape, pad_shape, scale_factor, flip, img_norm_cfg, gt_bboxes, gt_labels
    - example
        
        ```python
        img_norm_cfg = dict(
            mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
        train_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
        ]
        test_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(type='Normalize', **img_norm_cfg),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img']),
                ])
        ]
        ```
- **Extend and use custom pipelines**
    1. 새로운 pipelin file 정의 후 저장 **my_pipeline.py**
    
    ```python
    import random
    from mmdet.datasets import PIPELINES
    
    @PIPELINES.register_module()
    class MyTransform:
        """Add your transform
    
        Args:
            p (float): Probability of shifts. Default 0.5.
        """
    
        def __init__(self, p=0.5):
            self.p = p
    
        def __call__(self, results):
            if random.random() > self.p:
                results['dummy'] = True
            return results
    ```
    
    1. import & use my_pipelin
    
    ```python
    custom_imports = dict(imports=['path.to.my_pipeline'], allow_failed_imports=False)
    
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='Pad', size_divisor=32),
        dict(type='MyTransform', p=0.2),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
    ]
    ```
    
    1. pipeline 시각화
    
    ```python
    tools/misc/browse_dataset.py
    ```
