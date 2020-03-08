import torch
from model import resnet34
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
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])#Correspond with pretrain setting

data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))
image_path = data_root + "/data_set/paints_data/test/"

try:
    json_file = open('./class_indices_exp2.json', 'r')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)


model = resnet34(num_classes=11)

model_weight_path = "./resNet34.pth"
model.load_state_dict(torch.load(model_weight_path))

model.eval()
test_result = {}

with torch.no_grad():

    root_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))
    for root,dirs,files in os.walk(root_path+'/data_set/paints_data/test'):

        for file in files:

            img = Image.open(os.path.join(root,file))
            img = data_transform(img)
            # expand batch dimension
            img = torch.unsqueeze(img, dim=0)

            output = torch.squeeze(model(img))
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()
            test_result[os.path.splitext(file)[0]] = class_indict[str(predict_cla)]

result_1 = []
for id,pre in zip(test_result.keys(), test_result.values()):
    result_1.append({'Photos_ID':id,'Pre_painter':pre})
result_DF = pd.DataFrame(result_1)
result_DF.to_excel('result.xls',index='Photos_ID')#Save predict result to an excel