import torch
import torchvision
from .mlgcn import MLGCN

model_dict = {'MLGCN':MLGCN}

def get_model(num_classes, args):
    if args.model_name == 'MLGCN':
        res101 = torchvision.models.resnet101(pretrained=True)
        model = model_dict[args.model_name](res101, num_classes, word_feature_path=args.word_feature_path)
        return model
    else:
        raise NotImplementedError('Only MLGCN can be chosen!')