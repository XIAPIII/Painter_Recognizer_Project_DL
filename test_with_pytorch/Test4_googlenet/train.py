import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torchvision.models.googlenet
import json
import matplotlib.pyplot as plt
import os
import torch.optim as optim
from model import GoogLeNet
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "val": transforms.Compose([transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))
image_path = data_root + "/data_set/paints_data/"

train_dataset = datasets.ImageFolder(root=image_path + "train",
                                     transform=data_transform["train"])
train_num = len(train_dataset)

flower_list = train_dataset.class_to_idx
cla_dict = dict((val, key) for key, val in flower_list.items())
# write dict into json file
json_str = json.dumps(cla_dict, indent=4)
with open('class_indices_exp2.json', 'w') as json_file:
    json_file.write(json_str)

batch_size = 100
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=0)

validate_dataset = datasets.ImageFolder(root=image_path + "val",
                                        transform=data_transform["val"])
val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=0)

net = torchvision.models.googlenet()
#net = GoogLeNet(aux_logits = True)#num_classes=11, aux_logits=True, init_weights=True)

model_weight_path = "./googlenet-pre.pth"
net.load_state_dict(torch.load(model_weight_path), strict=False)
net.fc = nn.Linear(1024, 11)

net.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0003)

best_acc = 0.0
save_path = './googleNet.pth'

loss_list = []
acc_list = []

for epoch in range(50):
    # train
    net.train()
    running_loss = 0.0
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        optimizer.zero_grad()
        logits, aux_logits2, aux_logits1 = net(images.to(device))
        loss0 = loss_function(logits, labels.to(device))
        loss1 = loss_function(aux_logits1, labels.to(device))
        loss2 = loss_function(aux_logits2, labels.to(device))
        loss = loss0 + loss1 * 0.3 + loss2 * 0.3
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        rate = (step + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
    print()

    net.eval()
    acc = 0.0
    with torch.no_grad():
        for data_test in validate_loader:
            test_images, test_labels = data_test
            outputs = net(test_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += (predict_y == test_labels.to(device)).sum().item()
        accurate_test = acc / val_num
        if accurate_test > best_acc:
            torch.save(net.state_dict(), save_path)
        print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
              (epoch + 1, running_loss / step, accurate_test))
    loss_list.append(running_loss / step)
    acc_list.append(acc / val_num)

print('Finished Training')

saving_list_l = np.array(loss_list)
np.save('losslist_ggl.npy', saving_list_l)
saving_list_a = np.array(acc_list)
np.save('acclist_ggl.npy', saving_list_a)

from matplotlib import pyplot as plt

plt.title("Loss Viration")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(loss_list)
plt.savefig(os.getcwd()+'ggl_loss.png')
plt.show()

plt.title("Accuracy Viration")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.plot(acc_list)
plt.savefig(os.getcwd()+'ggl_acc.png')
plt.show()