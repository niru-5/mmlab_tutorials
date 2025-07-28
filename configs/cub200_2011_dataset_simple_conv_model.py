# define scope here.
import sys
# sys.path.append('..')
sys.path.append('/hdd/side_projects/mmlab_tutorials/')
default_scope = 'mmengine_custom'



default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=4, 
                    save_best='accuracy/top1', rule='greater',
                    save_begin=20),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='CustomVisualizationHook', val_interval=8, test_interval=10))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend'),
                dict(type='MLflowVisBackend',
                     save_dir='/hdd/side_projects/data/ml_experiments/mlruns',
                     exp_name='cub200_2011',
                     run_name='cub200_2011_mobilenet_v2_model',
                     tracking_uri=None
                     )]
# Uncomment this line to use the custom .
visualizer = dict(
    type='CustomVisualizer', vis_backends=vis_backends, name='visualizer')

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False

model = dict(
    type='mmengine_custom.MobileNetV2Model', # mmengine_custom.SimpleConvModel
    num_classes=201,
    data_preprocessor=dict(type='ImgDataPreprocessor',
                           mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225],
                           pad_size_divisor=32,
                           pad_value=0,
                           bgr_to_rgb=False,
                           rgb_to_bgr=False,
                           non_blocking=False),
    init_cfg=dict(type='Xavier', gain=1.0, distribution='normal', bias=0.0, bias_prob=None, layer=None),
)

optim_wrapper = dict(
    optimizer=dict(lr=0.1, momentum=0.9, type='SGD', weight_decay=0.0001),
    type='OptimWrapper')
param_schedulers = [
    dict(
        begin=0, by_epoch=False, end=20, start_factor=0.001, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=100,
        gamma=0.1,
        milestones=[
            2,
            20,
            50,
            80,
        ],
        type='MultiStepLR'),
]

train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    dataset=dict(type='mmengine_custom.CUB200_2011Dataset',
                 ann_file='/hdd/side_projects/data/datasets/bird_datasets/Caltech-UCSD-Birds-200-2011/CUB_200_2011/CUB_200_2011/train_df.csv',
                 data_root='/hdd/side_projects/data/datasets/bird_datasets/Caltech-UCSD-Birds-200-2011/CUB_200_2011/CUB_200_2011',
                 pipeline=[dict(type='LoadImageFromFile'),
                           dict(type='Resize', scale=(256, 256), keep_ratio=False),
                        #    dict(type='RandomFlip', prob=0.50),
                        #    dict(type='RandomRotate', degrees=10, prob=0.25),
                           dict(type='ToTensor', keys=['img']),
                           dict(type='mmengine_custom.CustomPackClsInputs')]
                 ),
    sampler=dict(type='DefaultSampler', shuffle=True),
    persistent_workers=True,
    pin_memory=True,
    collate_fn=dict(type='default_collate')
)
train_cfg=dict(by_epoch=True, max_epochs=100, val_interval=4, val_begin=10)
val_cfg=dict()

val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    dataset=dict(type='mmengine_custom.CUB200_2011Dataset',
                 ann_file='/hdd/side_projects/data/datasets/bird_datasets/Caltech-UCSD-Birds-200-2011/CUB_200_2011/CUB_200_2011/validation_df.csv',
                 data_root='/hdd/side_projects/data/datasets/bird_datasets/Caltech-UCSD-Birds-200-2011/CUB_200_2011/CUB_200_2011',
                 pipeline=[dict(type='LoadImageFromFile'),
                           dict(type='Resize', scale=(256, 256), keep_ratio=False),
                           dict(type='ToTensor', keys=['img']),
                           dict(type='mmengine_custom.CustomPackClsInputs')]),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True,
    pin_memory=True,
    collate_fn=dict(type='default_collate')
)

val_evaluator = dict(type='mmengine_custom.Accuracy')





# model
# data preprocessor

# train dataset, dataloader, augmentation, train cfg
# val dataset, dataloader, val cfg, val evaluator
# test dataset, dataloader, test cfg, test evaluator

# optimizer, learning rate scheduler

# custom hooks
