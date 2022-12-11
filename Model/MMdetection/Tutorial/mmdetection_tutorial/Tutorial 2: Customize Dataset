- new data format 사용을 위해서는 online or offline에서 dataset을 변환해야한다.
    - **Support new data format**
        - **Reorganize new data formats to COCO format**
            - 가장 간단한 방법은 new dataset을 기존 data format(coco,voc)으로 변환이다.
                - The annotation json files in COCO
                    - images: image의 정보 → file_name , height, width , id
                    - annotations: 각 객체별 annotations 정보
                    - categories: class name과 id
                - COCO format example
                    
                    ```python
                    'images': [
                        {
                            'file_name': 'COCO_val2014_000000001268.jpg',
                            'height': 427,
                            'width': 640,
                            'id': 1268
                        },
                        ...
                    ],
                    
                    'annotations': [
                        {
                            'area': 1035.749,
                            'iscrowd': 0,
                            'image_id': 1268,
                            'bbox': [192.81, 224.8, 74.73, 33.43],
                            'category_id': 16,
                            'id': 42986
                        },
                        ...
                    ],
                    
                    'categories': [
                        {'id': 0, 'name': 'car'},
                     ] 
                    ```
                    
            - dataset format 변환 후 2 step의 작업 필요하다
                - 1. **Modify the config file for using the customized dataset.**
                    - modify config file two aspects
                        1. **The `classes` field in data →**수정(in data.train,data.val,data.test) 
                        2. **The `num_classes` field in model →** 수정 all the num_classes
                        - example
                            
                            ```python
                            # the new config inherits the base configs to highlight the necessary modification
                            _base_ = './cascade_mask_rcnn_r50_fpn_1x_coco.py'
                            
                            # 1. dataset settings
                            dataset_type = **'CocoDataset'**
                            classes = **('a', 'b', 'c', 'd', 'e')**
                            data = dict(
                                samples_per_gpu=2,
                                workers_per_gpu=2,
                                train=dict(
                                    type=dataset_type,
                                    # explicitly add your class names to the field `classes`
                                    **classes=classes,**
                                    ann_file=**'path/to/your/train/annotation_data',**
                                    img_prefix=**'path/to/your/train/image_data'),**
                                val=dict(
                                    type=dataset_type,
                                    # explicitly add your class names to the field `classes`
                                    **classes=classes,**
                                    ann_file=**'path/to/your/val/annotation_data',**
                                    img_prefix=**'path/to/your/val/image_data'),**
                                test=dict(
                                    type=dataset_type,
                                    # explicitly add your class names to the field `classes`
                                    **classes=classes,**
                                    ann_file=**'path/to/your/test/annotation_data',**
                                    img_prefix**='path/to/your/test/image_data')**)
                            
                            # 2. model settings
                            
                            # explicitly over-write all the `num_classes` field from default 80 to 5.
                            model = dict(
                                roi_head=dict(
                                    bbox_head=[
                                        dict(
                                            type='Shared2FCBBoxHead',
                                            # explicitly over-write all the `num_classes` field from default 80 to 5.
                                            **num_classes=5**),
                                        dict(
                                            type='Shared2FCBBoxHead',
                                            # explicitly over-write all the `num_classes` field from default 80 to 5.
                                            **num_classes=5**),
                                        dict(
                                            type='Shared2FCBBoxHead',
                                            # explicitly over-write all the `num_classes` field from default 80 to 5.
                                            **num_classes=5**)],
                                # explicitly over-write all the `num_classes` field from default 80 to 5.
                                mask_head=dict(**num_classes=5**)))
                            ```
                            
                - 2.**Check the annotations of the customized dataset.**
                    - check annotation 3 aspect
                        1. categories field length == classes field length
                        2. classes & categories field element must **same** **order,name**
                        3. category_id in annotations 유효해야한다→ 값이 categori에 속해야한다.
                        - example
                            
                            ```python
                            'annotations': [
                                {
                                    'area': 1035.749,
                                    'iscrowd': 0,
                                    'image_id': 1268,
                                    'bbox': [192.81, 224.8, 74.73, 33.43],
                                    'category_id': 16,
                                    'id': 42986
                                },
                                ...
                            ],
                            
                            # MMDetection automatically maps the uncontinuous `id` to the continuous label indices.
                            'categories': [
                                {'id': 1, 'name': 'a'}, {'id': 3, 'name': 'b'}, {'id': 4, 'name': 'c'}, {'id': 16, 'name': 'd'}, {'id': 17, 'name': 'e'},
                             ]
                            ```
                            
        - **Reorganize new data format to Middle format**
            - COCO or PASCAL 아니어도 괜찮다.
            - Middle format
                - for testing: filename, width,height
                - for training: ann
                    - ann contain at least 2 fields
                        - **bboxs: numpy array**
                        - **labels: numpy array**
            - example
                
                ```python
                [
                    {
                        'filename': 'a.jpg',
                        'width': 1280,
                        'height': 720,
                        'ann': {
                            'bboxes': <np.ndarray, float32> (n, 4),
                            'labels': <np.ndarray, int64> (n, ),
                            'bboxes_ignore': <np.ndarray, float32> (k, 4),
                            'labels_ignore': <np.ndarray, int64> (k, ) (optional field)
                        }
                    },
                    ...
                ]
                ```
                
        - Convert method
            1. online
                1. **`load_annotations(self, ann_file)` and `get_ann_info(self, idx)`**
            2. offline
                1. 직접 코드
        - **An example of customized dataset**
            - new dataset
                
                ```python
                #
                000001.jpg     # filename
                1280 720       # width , height
                2              # bbox num
                10 20 40 60 1  # bbox coordi & class
                20 40 50 60 2  # bbox coordi & class
                #
                000002.jpg
                1280 720
                3
                50 20 40 60 2
                20 40 30 45 2
                30 40 50 60 3
                ```
                
            - create new dataset format in **mmdet/datasets/my_dataset.py**
                
                ```python
                import mmcv
                import numpy as np
                
                from .builder import DATASETS
                from .custom import CustomDataset
                
                @DATASETS.register_module()
                class MyDataset(CustomDataset):
                
                    CLASSES = ('person', 'bicycle', 'car', 'motorcycle')
                
                    def load_annotations(self, ann_file):
                        ann_list = mmcv.list_from_file(ann_file)
                
                        data_infos = []
                        for i, ann_line in enumerate(ann_list):
                            if ann_line != '#':
                                continue
                
                            img_shape = ann_list[i + 2].split(' ')
                            width = int(img_shape[0])
                            height = int(img_shape[1])
                            bbox_number = int(ann_list[i + 3])
                
                            anns = ann_line.split(' ')
                            bboxes = []
                            labels = []
                            for anns in ann_list[i + 4:i + 4 + bbox_number]:
                                bboxes.append([float(ann) for ann in anns[:4]])
                                labels.append(int(anns[4]))
                
                            data_infos.append(
                                dict(
                                    filename=ann_list[i + 1],
                                    width=width,
                                    height=height,
                                    ann=dict(
                                        bboxes=np.array(bboxes).astype(np.float32),
                                        labels=np.array(labels).astype(np.int64))
                                ))
                
                        return data_infos
                
                    def get_ann_info(self, idx):
                        return self.data_infos[idx]['ann']
                ```
                
                ```python
                # 결과물
                [
                	{
                	filename = str
                	width = int
                	height =int
                	ann = {
                	bboxes: np.array(float[4개]).astype(np.float32),
                	labels : np.array(int).astype(np.int64)}
                	}
                	{...}
                	{...}
                ]
                ```
                
            - use mydataset and modify config file
                
                ```python
                dataset_A_train = dict(
                    type='MyDataset',
                    ann_file = 'image_list.txt',
                    pipeline=train_pipeline
                )
                ```
                
    - **Customize datasets by dataset wrappers**
        - many dataset wrapper→ to mix or modify dataset
        - **`RepeatDataset`: simply repeat the whole dataset.**
            
            ```python
            dataset_A_train = dict(
                    type='RepeatDataset',
                    times=N,
                    dataset=dict(  # This is the original config of Dataset_A
                        type='Dataset_A',
                        ...
                        pipeline=train_pipeline
                    )
                )
            ```
            
        - **`ClassBalancedDataset`: repeat dataset in a class balanced manner. category frequency base**
            
            ```python
            dataset_A_train = dict(
                    type='ClassBalancedDataset',
                    oversample_thr=1e-3,
                    dataset=dict(  # This is the original config of Dataset_A
                        type='Dataset_A',
                        ...
                        pipeline=train_pipeline
                    )
                )
            ```
            
        - **`ConcatDataset`: concatenate datasets.**
            1. concatenate same type with different annotation file
            
            ```python
            dataset_A_train = dict(
                type='Dataset_A',
                ann_file = ['anno_file_1', 'anno_file_2'],
                pipeline=train_pipeline
            )
            
            # test or evaluation에 사용될 경우
            dataset_A_train = dict(
                type='Dataset_A',
                ann_file = ['anno_file_1', 'anno_file_2'],
                separate_eval=False,
                pipeline=train_pipeline
            )
            ```
            
            1. concatenate different
            
            ```python
            dataset_A_train = dict()
            dataset_B_train = dict()
            
            data = dict(
                imgs_per_gpu=2,
                workers_per_gpu=2,
                train = [
                    dataset_A_train,
                    dataset_B_train
                ],
                val = dataset_A_val,
                test = dataset_A_test
                )
            ```
            
            1. **define `ConcatDataset` explicitly**
            
            ```python
            dataset_A_val = dict()
            dataset_B_val = dict()
            
            data = dict(
                imgs_per_gpu=2,
                workers_per_gpu=2,
                train=dataset_A_train,
                val=dict(
                    type='ConcatDataset',
                    datasets=[dataset_A_val, dataset_B_val],
                    separate_eval=False))
            ```
            
    - **Modify Dataset Classes**
        - 현재 dataset에서 특정 class만 train/test 하고 싶을 경우 자동으로 필터링
        
        ```python
        # 코드로 지정
        classes = ('person', 'bicycle', 'car')
        data = dict(
            train=dict(classes=classes),
            val=dict(classes=classes),
            test=dict(classes=classes))
        
        # file로 지정 가능
        classes = 'path/to/classes.txt'
        data = dict(
            train=dict(classes=classes),
            val=dict(classes=classes),
            test=dict(classes=classes))
        ```
        
        - 빈 GT는 자동 필터링 → middle는 아니다.
    - **COCO Panoptic Dataset**
        - COCO format과는 다르다.
        
        ```python
        'images': [
            {
                'file_name': '000000001268.jpg',
                'height': 427,
                'width': 640,
                'id': 1268
            },
            ...
        ]
        
        'annotations': [
            {
                'filename': '000000001268.jpg',
                'image_id': 1268,
                'segments_info': [
                    {
                        'id':8345037,  # One-to-one correspondence with the id in the annotation map.
                        'category_id': 51,
                        'iscrowd': 0,
                        'bbox': (x1, y1, w, h),  # The bbox of the background is the outer rectangle of its mask.
                        'area': 24315
                    },
                    ...
                ]
            },
            ...
        ]
        
        'categories': [  # including both foreground categories and background categories
            {'id': 0, 'name': 'person'},
            ...
         ]
        ```
