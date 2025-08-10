# RTMDet-tiny configuration for bird detection
# Much smaller and faster than RTMDet-S

num_classes = 1
meta_info = {
    "classes": ["bird"],
    "palette": [(255, 0, 0)],
}
data_root = "/hdd/side_projects/data/datasets/bird_datasets/birds_data"

# Model configuration - RTMDet-tiny (smaller than RTMDet-S)
checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth'

model = dict(
    type='RTMDet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False,
        batch_augments=None),
    backbone=dict(
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.167,  # Much smaller than S (0.33)
        widen_factor=0.375,   # Much smaller than S (0.5)
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True),
        init_cfg=dict(
            type='Pretrained', 
            prefix='backbone.', 
            checkpoint=checkpoint)),
    neck=dict(
        type='CSPNeXtPAFPN',
        in_channels=[96, 192, 384],  # Smaller channels than S [128, 256, 512]
        out_channels=96,             # Smaller than S (128)
        num_csp_blocks=1,
        expand_ratio=0.5,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        type='RTMDetSepBNHead',
        num_classes=num_classes,
        in_channels=96,              # Smaller than S (128)
        stacked_convs=2,
        feat_channels=96,            # Smaller than S (128)
        anchor_generator=dict(
            type='MlvlPointGenerator', offset=0, strides=[8, 16, 32]),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        with_objectness=False,
        exp_on_reg=False,            # Disabled for tiny version
        share_conv=True,
        pred_kernel_size=1,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True)),
    train_cfg=dict(
        assigner=dict(type='DynamicSoftLabelAssigner', topk=13),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=30000,
        min_bbox_size=0,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.65),
        max_per_img=300),
)

# Training pipeline with reduced augmentation complexity for faster training
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='CachedMosaic',
        img_scale=(640, 640),
        pad_val=114.0,
        max_cached_images=10,  # Reduced from 20 for faster processing
        random_pop=False),
    dict(
        type='RandomResize',
        scale=(1280, 1280),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(640, 640)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(
        type='CachedMixUp',
        img_scale=(640, 640),
        ratio_range=(1.0, 1.0),
        max_cached_images=5,   # Reduced from 10 for faster processing
        random_pop=False,
        pad_val=(114, 114, 114),
        prob=0.5),
    dict(type='PackDetInputs')
]

# Stage 2 pipeline (simpler, no mosaic/mixup)
train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomResize',
        scale=(640, 640),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(640, 640)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs')
]

# Test pipeline
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

# Dataset configuration (using your bird dataset paths)
train_dataloader = dict(
    batch_size=8,  # Increased batch size since model is smaller
    num_workers=4,
    batch_sampler=None,
    pin_memory=True,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=meta_info,
        ann_file="/hdd/side_projects/data/datasets/bird_datasets/birds_data/mva2023_sod4bird_train-20250313T193936Z-002/mva2023_sod4bird_train/annotations/split_train_coco.json",
        data_prefix=dict(img="images-001/images/"),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=None)
)

val_dataloader = dict(
    batch_size=8,  # Increased batch size since model is smaller
    num_workers=4,
    drop_last=False,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=meta_info,
        ann_file="/hdd/side_projects/data/datasets/bird_datasets/birds_data/mva2023_sod4bird_train-20250313T193936Z-002/mva2023_sod4bird_train/annotations/split_val_coco.json",
        data_prefix=dict(img="images-001/images/"),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=None)
)

test_dataloader = val_dataloader

# Evaluation
val_evaluator = dict(
    type='CocoMetric',
    ann_file="/hdd/side_projects/data/datasets/bird_datasets/birds_data/mva2023_sod4bird_train-20250313T193936Z-002/mva2023_sod4bird_train/annotations/split_val_coco.json",
    metric='bbox',
    format_only=False,
    backend_args=None,
    proposal_nums=(100, 1, 10)
)
test_evaluator = val_evaluator

# Training configuration
max_epochs = 300
stage2_num_epochs = 20
base_lr = 0.004
interval = 10
auto_scale_lr = dict(base_batch_size=16, enable=False)

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=interval,
    dynamic_intervals=[(max_epochs - stage2_num_epochs, 1)]
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, 
        bias_decay_mult=0, 
        bypass_duplicate=True)
)

# Learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

# Hooks
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', 
        interval=interval, 
        max_keep_ckpts=3
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook')
)

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2)
]

# Environment
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# Visualization
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='MLflowVisBackend',
             save_dir='/hdd/side_projects/data/ml_experiments/mlruns',
             exp_name='bird_detection',
             run_name='rtmdet_tiny_8xb32-300e_bird_detection',
             tracking_uri=None)
    ],
    name='visualizer'
)

# Logging
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False

# Pretrained model loading
load_from = "/hdd/side_projects/data/ml_models/pretrained/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth"