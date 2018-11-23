import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import os
import json
from PIL import Image
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path',
                        type=str)
    parser.add_argument('checkpoint',
                        type=str)
    parser.add_argument('--top_k',
                        type=int,
                        default=5)
    parser.add_argument('--category_names',
                        type=str,
                        default="cat_to_name.json")
    #代码改变部分请审阅者注意
    parser.add_argument('--gpu',
                        default=False)
    parser = parser.parse_args()


    with open(parser.category_names, 'r') as f:
        cat_to_name = json.load(f)

    if parser.gpu and torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('使用 GPU 模式预测')
    else:
        device = torch.device('cpu')
        print('使用 CPU 模式预测')
    
    model = load_model(parser.checkpoint)#加载已训练的模型
    #代码改变部分请审阅者注意
    top_probs, chosen_classes = predict(device, parser.image_path, model, parser.top_k)

    flower_names = []
    for idx in chosen_classes:
        flower_names.append(cat_to_name[idx])

    for itr in zip(flower_names, top_probs):
        print('预测为 {} 的概率是 {:.2f}%'.format(itr[0],itr[1]))

    title =  parser.image_path.split('/')[2]
    real_name = cat_to_name[title]
    print('花卉的真实种类是 {}'.format(real_name))

    if real_name == flower_names[0]:
        print('预测正确！')
    else:
        print('预测错误！')



def load_model(pth='checkpoint.pth'):
    
    print('正在重构模型......')
    model_state = torch.load(pth)
    print('正在使用 {} 模型......'.format(model_state['arch']))
    model = models.__dict__[model_state['arch']](pretrained=True)
    print('正在传递分类器状态......')
    model.classifier = model_state['classifier']
    print('正在加载已训练好的参数......')
    model.load_state_dict(model_state['state_dict'])
    print('正在传递标签索引......')
    model.class_to_idx = model_state['class_to_idx']
    print('模型重构完毕！')
    
    return model



def process_image(image_pth):

    pil_image = Image.open(image_pth)
    data_transforms = transforms.Compose([                  #测试集
        transforms.Resize(256),
        transforms.CenterCrop(224), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    image = data_transforms(pil_image)

    return image


#代码改变部分请审阅者注意
def predict(device, image_pth, model, topk=5):

    model.eval()
    #代码改变部分请审阅者注意
    # model.cpu()
    model.to(device)
    
    image = process_image(image_pth) # Tensor
    model_input = image.unsqueeze(0)
    #代码改变部分请审阅者注意
    # output = model(model_input)
    output = model(model_input.to(device))

    output_probs = F.softmax(output, dim=1)#将交叉熵输出归一化
    top_probs, top_labels = output_probs.topk(topk)
    #代码改变部分请审阅者注意
    # top_probs = top_probs.detach().numpy().tolist()[0] 
    # top_labels = top_labels.detach().numpy().tolist()[0]
    top_probs = top_probs.cpu().detach().numpy().tolist()[0] 
    top_labels = top_labels.cpu().detach().numpy().tolist()[0]
    
    index_to_class = {model.class_to_idx[k]: k for k in model.class_to_idx}
    chosen_classes = []
    for label in top_labels:
        chosen_classes.append(index_to_class[label])

    return top_probs, chosen_classes




if __name__ == '__main__':
    
    main()