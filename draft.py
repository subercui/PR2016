# -*- coding: utf-8 -*-
import keras
from scipy.misc import imread, imresize, imsave

test_img_path='/Users/subercui/Downloads/pr2016/example_01_0003.jpg'
img = imread(test_img_path)#750*1000,比如说我们就默认长边在前才是正方向，之后可以修正方向
if img.shape[0]<img.shape[1]:
    img=img.transpose(1,0,2)
for res in [100,200,300,500]:    
    resized_img=imresize(img, (res,res))
    imsave('/Users/subercui/Downloads/pr2016/example_res%d.jpg'%res,resized_img)