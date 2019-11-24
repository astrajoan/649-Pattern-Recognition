机器学习-目标检测-Viola-Jones

From:  shechj  7146  2018-09-03


前言
Viola-Jones算法在2001年的CVPR上提出的应用于图片人脸检测的算法，虽然有点像冷兵器时代的算法，因为其高效而快速的检测即使到现在也依然被广泛使用和借鉴。
Viola和Jones发表了经典的《Rapid Object Detection using a Boosted Cascade of Simple Features》和《Robust Real-Time Face Detection》，所以该算法叫做Viola-Jones 算法。
流程
Viola-Jones的基本思想是在图像灰度积分图的基础上，通过特征原型从多维度计算图像的特征，并应用AdaBoost应用与这些特征上强分类器，最终将强分类器串联生成更强大的级联分类器，最终实现检测图片是否含有人脸的功能。
Viola-Jones算法的实现大概分为4部分
对图像进行灰度并实现积分图。
在积分图的基础上，通过特征原型构建图像的Haar-like特征。
使用Haar-like特征，并通过用AdaBoost方案构建强分类器。
将多个强分类器串联构建级联分类器。

 


1灰度积分图

图像灰度积分图  
![image](https://github.com/astrajoan/649-Pattern-Recognition/blob/master/img/WeChat%20Image_20191123183428.jpg)
对于一幅灰度的图像，积分图像中的任意一点(x,y)的值是指从图像的左上角到这个点的所构成的矩形区域内所有的点的灰度值之和。  
积分图的作用  
积分图可以快速找到灰度图像中某个矩形范围内的灰度值的和，在做Harr-like特征是非常重要，如下图所示。  
![image](https://github.com/astrajoan/649-Pattern-Recognition/blob/master/img/WeChat%20Image_20191123183844.jpg)


2Haar特征  
Haar特征又叫做Haar-like特征，是计算机视觉领域一种常用的特征描述算子(也称为Haar特征，这是因为Haar-like是受到一维haar小波的启示而发明的,所以称为类Haar特征)，后来又将Haar-like扩展到三维空间(称为3DHaar-Like)用来描述视频中的动态特征。
可以参考论文 《A general framework for object detection》、《Rapid object detection using a boosted cascade of simple features》、《An extended set of Haar-like features for rapid object detection》。


特征原型  
![image](https://github.com/astrajoan/649-Pattern-Recognition/blob/master/img/WeChat%20Image_20191123183849.jpg)

Haar-like特征就是在图像中以特征原型为基础并在图像上进行扩展最终生成的特征矩形，特征矩形内的所有白色矩形值的和减去所有黑色矩形值的和的值被称作该特征矩形的特征值，也是图像的一个特征值。所以图像最终有多少个特征矩形就会有多少个特征(值)。特征圆形的发展如上图所示。  
![image](https://github.com/astrajoan/649-Pattern-Recognition/blob/master/img/WeChat%20Image_20191123183852.jpg)


Viola-Jones 算法主要用了上图三种特征原型。特征原型是由黑色和白色矩形连接生成的。如上图所示，特征原型分为二邻接、三邻接、四邻接。特征原型是特征矩形的基本单位

特征矩形  
![image](https://github.com/astrajoan/649-Pattern-Recognition/blob/master/img/WeChat%20Image_20191123183855.jpg)
以特征原型为基本单位，在图像上扩展形成的即为特征矩形，最终生成的每一个特征矩形对应着图像的一个特征，特征矩形内的白色部分所有值的和减去黑色部分所有值的和即为该特征的特征值。  
![image](https://github.com/astrajoan/649-Pattern-Recognition/blob/master/img/WeChat%20Image_20191123183859.jpg)

如上图所示，a图中的矩形区域即为二邻接特征原型水平扩展生成的特征矩形，该矩形对应的图像灰度值区域为彩色框中的区域，白色区域的特征值之和减去黑色区域的特征值之和即为该特征矩形的特征值，上图分别用特征值矩阵和积分图进行了计算。可以看到当特征矩形很大时(比如20*20)时，用积分图计算会减少运算数量。

3AdaBoost
AdaBoost 同时进行特征选择与分类器训练，简单来说，AdaBoost 就是将一系列的”弱”分类器通过线性组合，构成一个”强”分类器
强弱分类器
![image](https://github.com/astrajoan/649-Pattern-Recognition/blob/master/img/WeChat%20Image_20191123183904.jpg)
弱分类器的能力弱，所以AdaBoost会训练多个弱分类器，同时设定每个弱分类的的权重值，最终的分类为所有弱分类器的组合判断结果。
AdaBoost训练过程
![image](https://github.com/astrajoan/649-Pattern-Recognition/blob/master/img/WeChat%20Image_20191123183907.jpg)
AdaBoost训练过程如上图所示，对上图简要说明:
初始化每个样本的权重。
每次循环(训练弱分类器)中：
归一化每个样本的权重。
在所有特征中找到可以使误分类最低的特征训练出的分类器作为该次的弱分类器（这里要训练θ）。
更新每个样本的权重，生成该弱分类器占所有弱分类器的权重。
生成强分类器
思考：强分类器中需要有多少弱分类器？

4级联分类器
只靠一个强分类器还不足以保证检测的正确率，需要一连串的强分类器联合在一起来提高检测正确率。但是如果让每个强分类器都很强的话，就需要让每个强分类器中包含更多的弱分类器，这样会增加训练难度的。
训练
级联分类器要处理两个平衡，一是强分类器中弱分类器的个数与训练时间的平衡（增加特征/弱分类器个数能提高检测率和降低误识率，但会增加计算时间），二是强分类器检测准确率和召回率的平衡。
基本策略是让前面的强分类器有少量的弱分类器(用少量的特征)，但是能够快速识别出不含人脸的头像，让含有人脸的图像经过后面检测严格(弱分类器个数多)的强分类器。
![image](https://github.com/astrajoan/649-Pattern-Recognition/blob/master/img/WeChat%20Image_20191123183912.jpg)
训练过程如上图所示，进行简要说明：
在训练之前设定每一层(强分类器的需要满足的检测失败率(FP)和召回率)。设定最终级联分类器需要满足的失败率的最大值。
只要级联分类器的失败率未降低到最大值以下，就需要生成一个新的强分类器
在每次生成的强分类器中，只要该强分类器满足该层预设的失败率和召回率即可停止。
预测过程
![image](https://github.com/astrajoan/649-Pattern-Recognition/blob/master/img/WeChat%20Image_20191123183917.jpg)
上图就是最终通过级联分类器进行分类的过程图。
