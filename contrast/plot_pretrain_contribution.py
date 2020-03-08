import numpy as np
import matplotlib.pyplot as plt
import os
#...手动筛选最大准确率
acclist_res_pre = np.load('./acclist_res.npy')
acclist_res = np.load('./acclist_resnopre.npy')
acclist_vgg_pre = np.load('./acclist_vgg_pre.npy')
acclist_vgg = np.load('./acclist_vgg.npy')
acclist_alex_pre = np.load('./acclist_alex_pre.npy')
acclist_alex = np.load('./acclist_alex.npy')
#print(acclist_alex)


x=np.arange(3)

y1=[0.714,0.571,0.618]
y2=[0.952,0.971,0.914]

bar_width=0.3
tick_label=['ResNet34','VGG16','AlexNet']


plt.bar(x,y1,bar_width,color='lightskyblue',label='Without Pretraining')
plt.bar(x+bar_width,y2,bar_width,color='teal',label='With Pretraining')
plt.title('Pretraining Contribution to Accuracy')
plt.legend()
plt.xticks(x+bar_width/2,tick_label)
plt.show()
plt.savefig(os.getcwd()+'2.png')