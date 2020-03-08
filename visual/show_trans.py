import torch
from PIL import Image
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import json
import os
import pandas as pd

data_transform = transforms.Compose(
    [transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
     ])


img_root = os.getcwd() + '/claude_monet_2752.png'
img = Image.open(img_root)
img = data_transform(img)
img = transforms.ToPILImage()(img)
img.show()
img.save(os.getcwd()+'op2.png')