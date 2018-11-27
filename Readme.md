## 第 1 部分 - 开发 Notebook [Image Classifier Project-zh_v7.ipynb]
* 导入软件包	在 notebook 的第一个单元格中导入所有必要的软件包和模块
* 改善训练数据	使用 torchvision 变换以通过随机缩放、旋转、镜像和/或裁剪改善训练数据
* 数据标准化	恰当地裁剪和标准化训练、验证和测试数据
* 数据加载	使用 torchvision 的 ImageFolder 加载每个数据集（训练集、验证集、测试集）的数据
* 数据批处理	使用 torchvision 的 DataLoader 加载每个数据集的数据
* 预训练的网络	从 torchvision.models 加载 VGG16 等预训练的网络并冻结参数
* 前馈分类器	定义了一个新的前馈网络用作分类器，并使用特征作为输入
* 训练网络	适当地训练前馈分类器的参数，而特征网络的参数保持静态
* 验证损失和准确率	训练期间显示验证损失和准确率
* 测试准确率	衡量网络在测试数据上的准确率
* 保存模型	将训练的模型保存为检查点，并保存相关的超参数和 class_to_idx 字典
* 加载检查点	可以通过函数成功地加载检查点和重构模型
* 图像处理	process_image 函数成功地将 PIL 图像转换为张量，并用作已训练模型的输入
* 类别预测	predict 函数成功地获取图像路径和检查点，然后返回该图像的前 K 个可能的类别
* 通过 matplotlib 进行健全性检查	创建了 matplotlib 图表，并显示图像和相关的前 5 个可能类别及其实际花卉名称

## 第 2 部分 - 命令行应用 

### 使用前请先参阅 [README.ipynb]

[train.py]
* 训练网络：成功地用图像数据集训练一个新的网络
* 训练验证日志：在训练网络时输出训练损失、验证损失和验证准确率
* 模型架构：训练脚本使用户能够从 torchvision.models 中选择至少两个不同的可用架构
* 模型超参数：训练脚本使用户能够设置学习速率、隐藏单元数和训练周期超参数
* 在 GPU 上进行训练：训练脚本使用户能够选择在 GPU 上训练模型

[predict.py]
* 预测类别：成功地读取了图像和检查点，然后输出最可能的图像类别及其概率
* 前 K 个类别：使用户能够输出前 K 个类别及其相关的概率
* 显示类别名称：使用户能够加载将类别值映射到其他类别名称的 JSON 文件
* 在 GPU 上进行预测：使用户能够使用 GPU 计算预测值

## 项目总结

[P4 Tips.ipynb]


[Image Classifier Project-zh_v7.ipynb]:https://github.com/Zhengqi-Li/Create_Your_Own_Image_Classifier/blob/master/Image%20Classifier%20Project-zh_v7.ipynb
[train.py]:https://github.com/Zhengqi-Li/Create_Your_Own_Image_Classifier/blob/master/train.py
[predict.py]:https://github.com/Zhengqi-Li/Create_Your_Own_Image_Classifier/blob/master/predict.py
[README.ipynb]:https://github.com/Zhengqi-Li/Create_Your_Own_Image_Classifier/blob/master/README.ipynb
[P4 Tips.ipynb]:https://github.com/Zhengqi-Li/Create_Your_Own_Image_Classifier/blob/master/P4%20Tips.ipynb
