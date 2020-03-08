# Painter_Recognizer_Project_DL
深度学习解决一个画作的画家识别项目

运用深度学习知识解决画家与画作的匹配

数据集：
  数据集分为带标签部分与无标签部分，
  带标签部分包含370张图片的训练集与35张图片的测试集，
  不带标签部分含46张图片，
  图片大小均为3*256*256
  
模型：
  最终使用ResNet34解决问题，对不带标签的画作判断以excel形式保存在resnet相应的目录下，
  同时也运用了VGG16、AlexNet、GoogLeNet训练显示对比
  
 其他：
  visual展示了图片经过transform.Compose的过程，
  contrast运用matplotlib简单绘制了模型表现及对比
 
