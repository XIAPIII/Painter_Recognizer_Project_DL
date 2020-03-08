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
acclist_ggl = np.load('./acclist_ggl.npy')
#print(acclist_ggl)

models_name = ['ResNet34','VGGNet16','GoogLeNet','AlexNet']
models_scores = [1.00, 0.79, 0.95, 0.86]
fig, ax = plt.subplots(figsize=(8, 8))
ax.barh(models_name, models_scores)
labels = ax.get_xticklabels()
plt.setp(labels)

# Add a vertical line, here we set the style in the function call
ax.axvline(np.mean(models_scores), ls='--', color='r')

# Now we'll move our title up since it's getting a little cramped
ax.title.set(y=1.03)

ax.set(xlim=[-0.1, 1.2], xlabel='Models Scores', ylabel='Models',
       title='Models Performance')

ax.set_xticks([0,0.33,0.66,1.00])
#fig.subplots_adjust(right=.1)

plt.show()
plt.savefig(os.getcwd()+'3.png')