# U_net_mnist

# 运行：

直接运行main.py，搭建U_net网络结构，对语义分割mnist数据集进行训练。

combined.npy：不同数字的灰度图

segmented.npy：相应图片的语义标注

# 实验效果

语义分割模型训练起来要比目标检测模型轻松很多，val loss能很轻易地降低至一个较低水平，测试集分割效果极佳。

语义分割效果可见demo文件夹。

0 - purplish red

1 - orange

2 - green

3 - pink

4 - white

5 - gray

6 - yellow

7 - violet

8 - dark blue

9 - black

10 -light blue
