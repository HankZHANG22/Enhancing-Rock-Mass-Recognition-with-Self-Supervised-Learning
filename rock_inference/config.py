from easydict import EasyDict as edict


config = edict()
config.backbone = 'resnet101'
config.model_version = "fujian_best"
config.embedding_size = 512
config.input_size = 448
config.detection_version = "detection_best"