from pathlib import Path
import os
from PIL import Image

import torch
from torch import nn
from torchvision import transforms
import timm

from utils import *

def infer(cfg):
    data_path = cfg['DATASET']['PARENT_PATH']
    data_path = Path(data_path)
    num_classes = len(cfg['DATASET']['CLASSES'])

    assert cfg['INFER']['CHECKPOINT_PATH'] is not None
    assert cfg['INFER']['CHECKPOINT_PATH'] != ""
    checkpoint_path = cfg['INFER']['CHECKPOINT_PATH']

    model = timm.create_model(
        cfg['MODEL']['NAME'], 
        pretrained=False, 
        num_classes=num_classes
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    softmax = nn.Softmax()

    assert cfg['INFER']['IMAGE_PATH'] is not None
    assert os.path.isfile(cfg['INFER']['IMAGE_PATH'])
    inputs_path = cfg['INFER']['IMAGE_PATH']
    input_pil = Image.open(inputs_path)
    transform = transforms.Compose([
        transforms.Resize(cfg['INFER']['IMAGE_SIZE']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(input_pil).unsqueeze(0)

    out = model(input_tensor)
    out_sm = softmax(out)

    for idx, conf in enumerate(out_sm[0]):
        print(cfg['DATASET']['CLASSES'][idx] + ":", f"{float(conf) * 100:.2f}%")


if __name__ == "__main__":
    args = args_parser()
    cfg = read_cfg(args)
    infer(cfg)