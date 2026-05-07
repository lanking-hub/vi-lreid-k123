from yacs.config import CfgNode as CN
cfg = CN()

cfg.SEED = 0

# dataset
# Current working setup uses the three confirmed datasets first.
# VCM will be added back after the dataset is uploaded and preprocessed.
cfg.DATASETS = ['regdb', 'sysu', 'llcm']
cfg.DATA_PATH_RegDB = '/data/sheng_hao_xuan_2025/datasets/RegDB/'
cfg.DATA_PATH_SYSU = '/data/sheng_hao_xuan_2025/datasets/SYSU-MM01/'
cfg.DATA_PATH_LLCM = '/data/sheng_hao_xuan_2025/datasets/LLCM/'
cfg.DATA_PATH_VCM = '/data/sheng_hao_xuan_2025/datasets/HITSZ-VCM/'
cfg.PRETRAIN_PATH = '/data/sheng_hao_xuan_2025/pretrained/jx_vit_base_p16_224-80ecf9dd.pth'

cfg.START_EPOCH = 1
cfg.MAX_EPOCH = 50

cfg.H = 256
cfg.W = 128
cfg.BATCH_SIZE = 32  # num of images for each modality in a mini batch
cfg.NUM_POS = 4

# PMT
# cfg.METHOD ='PMT'
# cfg.PL_EPOCH = 6    # for PL strategy
# cfg.MSEL = 0.5      # weight for MSEL
# cfg.DCL = 0.5       # weight for DCL
cfg.MARGIN = 0.1    # margin for triplet
cfg.MARGINCROSS = 0.1
cfg.MARGINCENTER = 0.1


# model
cfg.STRIDE_SIZE =  [12, 12]
cfg.DROP_OUT = 0.03
cfg.ATT_DROP_RATE = 0.0
cfg.DROP_PATH = 0.1
cfg.K1_BLOCKS = '4-7'
cfg.K2_BLOCKS = '0-3'
cfg.K3_BLOCKS = '8-11'
cfg.K1_LORA_RANK = 4
cfg.K2_LORA_RANK = 4
cfg.K3_LORA_RANK = 4
cfg.K1_LORA_ALPHA = 8
cfg.K2_LORA_ALPHA = 8
cfg.K3_LORA_ALPHA = 8

# optimizer
cfg.OPTIMIZER_NAME = 'AdamW'  # AdamW or SGD
cfg.MOMENTUM = 0.9    # for SGD

cfg.BASE_LR = 3e-4
cfg.WEIGHT_DECAY = 1e-4
cfg.WEIGHT_DECAY_BIAS = 1e-4
cfg.BIAS_LR_FACTOR = 1

cfg.LR_PRETRAIN = 0.5
cfg.LR_MIN = 0.01
cfg.LR_INIT = 0.01
cfg.WARMUP_EPOCHS = 3








