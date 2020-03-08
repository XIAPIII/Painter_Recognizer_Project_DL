import torch.nn as nn
from torchvision import transforms, datasets
import json
import os
import torch.optim as optim
from model import vgg
import torch
import numpy as np
import torchvision.models.vgg
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

data_transform = {
    "train": transforms.Compose([transforms.Resize(256),# equal ratio
                                 transforms.CenterCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "val": transforms.Compose([transforms.Resize(224),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}


data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))
image_path = data_root + "/data_set/paints_data/"
train_dataset = datasets.ImageFolder(root=image_path+"train",
                                     transform=data_transform["train"])
train_num = len(train_dataset)

flower_list = train_dataset.class_to_idx
cla_dict = dict((val, key) for key, val in flower_list.items())
json_str = json.dumps(cla_dict, indent=4)
with open('class_indices_exp2.json', 'w') as json_file:
    json_file.write(json_str)

batch_size = 50 #32
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=0)

validate_dataset = datasets.ImageFolder(root=image_path + "val",
                                        transform=data_transform["val"])
val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=0)

model_name = "vgg11"    #vgg16
net = vgg(model_name=model_name, class_num=11, init_weights=True)
net.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)

best_acc = 0.0
save_path = './{}Net.pth'.format(model_name)

loss_list = []
acc_list = []

for epoch in range(2):
    net.train()
    running_loss = 0.0
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        optimizer.zero_grad()
        outputs = net(images.to(device))
        loss = loss_function(outputs, labels.to(device))
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
            optimizer.zero_grad()
            outputs = net(test_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += (predict_y == test_labels.to(device)).sum().item()
        accurate_test = acc / val_num
        if accurate_test > best_acc:
            best_acc = accurate_test
            torch.save(net.state_dict(), save_path)
        print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
              (epoch + 1, running_loss / step, accurate_test))
    loss_list.append(running_loss / step)
    acc_list.append(acc / val_num)

print('Finished Training')

saving_list_l = np.array(loss_list)
np.save('losslist_vgg.npy', saving_list_l)
saving_list_a = np.array(acc_list)
np.save('acclist_vgg.npy', saving_list_a)

from matplotlib import pyplot as plt

plt.title("Loss Viration")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(loss_list)
plt.savefig(os.getcwd()+'vgg_loss.png')
plt.show()

plt.title("Accuracy Viration")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.plot(acc_list)
plt.savefig(os.getcwd()+'vgg_acc.png')
plt.show()