import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import os
import json 
import argparse

def main():
    #设置变量
    parser = argparse.ArgumentParser()

    #训练数据路径
    parser.add_argument('data_dir', 
                        type=str)
    #保存路径
    parser.add_argument('--save_dir',
                        type=str,
                        default='.')
    #保存文件名                  
    parser.add_argument('--save_name',
                        type=str,
                        default='checkpoint')
    #标签文件
    parser.add_argument('--categories_json',
                        type=str,
                        default="cat_to_name.json")
    #模型
    parser.add_argument('--arch',
                        type=str,    
                        default="densenet121")
    # GPU 模式
    parser.add_argument('--gpu',
                        default=False)
    #学习率
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.001)
    #隐藏层单元数
    parser.add_argument('--hidden_layer',
                        type=int,
                        default=512)
    #训练周期
    parser.add_argument('--epochs',
                        type=int,
                        default=5) 

    parser = parser.parse_args()

    with open(parser.categories_json, 'r') as f:
        cat_to_name = json.load(f)

    if not os.path.isdir(parser.save_dir):
        print(f'文件夹 {parser.save_dir} 不存在！创建中...')
        os.makedirs(parser.save_dir)
        print(f'模型将保存至 {parser.save_dir} 文件夹中')
    
    expect_mean = [0.485, 0.456, 0.406]
    expect_std = [0.229, 0.224, 0.225]
    resize = 256
    image_size = 224
    batch_size = 32

    #指定数据处理方式
    data_transforms = {                       
    'train': transforms.Compose([                 #训练集
        transforms.RandomResizedCrop(image_size), #随机裁剪
        transforms.RandomHorizontalFlip(), #随机翻转
        transforms.RandomRotation(30),     #随机旋转
        transforms.ToTensor(),             #将 PIL.Image/numpy.ndarray 数据进转化为torch.FloadTensor，并归一化到[0, 1.0]
        transforms.Normalize(expect_mean, expect_std)]), #将均值和标准差标准化到网络期望的结果
    
    'valid': transforms.Compose([                 #验证集
        transforms.Resize(resize),
        transforms.CenterCrop(image_size),        #裁剪至合适大小
        transforms.ToTensor(),
        transforms.Normalize(expect_mean, expect_std)]),
    
    'test': transforms.Compose([                  #测试集
        transforms.Resize(resize),
        transforms.CenterCrop(image_size), 
        transforms.ToTensor(),
        transforms.Normalize(expect_mean, expect_std)])
                }

    #指定数据集路径
    image_datasets = {x: datasets.ImageFolder(
                        os.path.join(parser.data_dir, x),
                        transform = data_transforms[x])
                        for x in list(data_transforms.keys())}


    #将数据集随机分批训练
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],     
                                                batch_size=batch_size, 
                                                shuffle=True) 
                                                for x in list(data_transforms.keys())}
    #数据集的大小
    dataset_sizes = {                 
            x: len(dataloaders[x].dataset) 
            for x in list(data_transforms.keys())} 

    # print(image_datasets)
    # print(dataloaders)
    # print(dataset_sizes)
    
    if parser.arch != 'vgg16' and parser.arch != 'densenet121':
        print('本程序只支持 vgg16 和 densenet121')
        exit(1)

    model = models.__dict__[parser.arch](pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    #输入单元大小
    input_size = {'densenet121': 1024,
                  'vgg16': 25088}
    #输出单元大小
    output_size = len(cat_to_name)

    #分类器
    classifier = nn.Sequential(OrderedDict([         
                          ('fc1', nn.Linear(input_size[parser.arch], parser.hidden_layer)),
                          ('relu1', nn.ReLU()),
                          ('fc2', nn.Linear(parser.hidden_layer, output_size))
                          ]))
    model.classifier = classifier

    # print(model.classifier)
###以下为训练主模块，测试时可被注释掉：
    if parser.gpu and torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('使用 GPU 模式训练')
    else:
        device = torch.device('cpu')
        print('使用 CPU 模式训练')

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=parser.learning_rate)
    print('当前使用 {} 模型进行训练\n隐藏层单元数：{}\n损失函数为交叉熵\n学习速率：{}'.format(parser.arch, parser.hidden_layer, parser.learning_rate))

    for epoch in range(parser.epochs):

        train_loss = 0.0     
        train_corrects = 0   
        train_sum = 0       
        valid_loss = 0.0
        valid_corrects = 0
        valid_sum = 0

        print('正在训练模型 {}/{}'.format(epoch + 1, parser.epochs))
        model.train()
        print('-' * 10)
        
        for ii, (images, labels) in enumerate(dataloaders['train']):
            
            images = images.to(device)   
            labels = labels.to(device)
            optimizer.zero_grad()              #初始化梯度为零  

            outputs = model(images)            #获取输出
            _, preds = torch.max(outputs, 1)   #获取概率最大的值及索引
            loss = criterion(outputs, labels)
            loss.backward()                    #反向传播
            optimizer.step()                   #进行优化

            train_loss += loss.item()          #累加损失
            train_sum += labels.size(0)        #累加训练数量,size(0)将 tensor 转换为 int
            train_corrects += (preds == labels).sum().item() ##累加正确预测的数量

            if (ii+1) % 20 == 0:               #每20次1输出

                avg_loss = train_loss / (ii+1)   #平均损失
                acc = (train_corrects / train_sum) * 100  #训练准确率
                print('训练损失: {:.4f} 准确率: {:.2f}% '.format(avg_loss, acc))
        print()             
        print('进入验证模式......')
        model.eval()
        print('-' * 10)
    
        with torch.no_grad():                 #进行验证，关闭梯度计算
            for ii, (images, labels) in enumerate(dataloaders['valid']):
                images = images.to(device) 
                labels = labels.to(device)
                outputs =  model(images)
                _, preds = torch.max(outputs, 1) 
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                valid_sum += labels.size(0)
                valid_corrects += (preds == labels).sum().item()

            acc = 0
            avg_valid_loss = valid_loss / dataset_sizes['valid']*batch_size
        if valid_corrects > 0:
            acc = (valid_corrects / valid_sum) * 100
        print('验证损失:{:.4f} 准确率:{:.2f}%'.format(avg_valid_loss, acc))
        print()
    print('模型训练完毕！')

    
    model.class_to_idx = image_datasets['train'].class_to_idx
    model.cpu()
    model_state = {
                'state_dict': model.state_dict(),
                'classifier': model.classifier,
                'class_to_idx': model.class_to_idx,
                'arch': parser.arch
                  }

    save_location = f'{parser.save_dir}/{parser.save_name}.pth'
    print(f'正在将模型保存至:{save_location}')
    torch.save(model_state, save_location)
    print('模型保存完毕！')

    print('正在测试模型......')
    model.eval()
    model.to(device)
    with torch.no_grad():
        running_corrects = 0
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(device)   
            labels = labels.to(device)
            outputs = model(inputs)               
            _, preds = torch.max(outputs, 1) 
            running_corrects += (preds == labels).sum().item()
        test_acc = running_corrects / dataset_sizes['test']
            
    print('测试的准确率为: %d %%' % (100 * test_acc))



if __name__ == '__main__':
    
    main()




    


