import numpy as np
import matplotlib.pyplot as plt
import os

losslist_res = np.load('./losslist_res.npy')
acclist_res = np.load('./acclist_res.npy')

plt.subplot(2,2,1)
plt.title("Loss Viration")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(losslist_res)

plt.subplot(2,2,2)
plt.title("Accuracy Viration")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.axhline(y=0.8, c="r", ls="--", lw=0.5)
plt.plot(acclist_res)
plt.savefig(os.getcwd()+'1.png')
plt.subplots_adjust(right = 0.1)
plt.show()
