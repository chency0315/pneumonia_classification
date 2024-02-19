# pneumonia_classification
This dataset is from Kaggle https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data

# prepare the dataset
Download the dataset from Kaggle or use the API to call the dataset.

這個資料集裡，有4172訓練圖片，1044驗證圖，624測試圖。

In this dataset, there are 4172 images for training, 1044 for validation, 624 for testing

# Generate data and data augmentation 

## 利用擴增照片套件來增加更多訓練照片，例如:銳利化，水平翻轉，隨機推移錯切，裁切比例，圖像像素重新排序。

Using a few techniques to create more image data like sharpening, horizontal flipping, random shear, scale
the size of the image, and translate the pixels to get more training data.

```
aug = iaa.pillike.FilterSharpen() #sharpen images
        self.seq = iaa.Sequential([
            iaa.Fliplr(0.5), # 50% horizontal flip
            iaa.Affine(
                shear=(-16,16), # random shear -16 ~ +16 degree
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale x, y: 80%~120%
                translate_px={"x": (-20, 20), "y": (-20, 20)}, # Translate images by -20 to 20 pixels on x- and y-axis independently 
            ),
        ])
```
引用相關套件，例如: tensorflow.keras, imguag用來擴增圖片，matplotlib.pyplot作圖。

import the modules for training 

```
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob
from tqdm.auto import tqdm
import tensorflow as tf
from tensorflow.keras import layers, models, utils
from sklearn.model_selection import train_test_split
import pandas as pd
import imgaug.augmenters as iaa
import imgaug as ia
import seaborn as sns
```

## 總共要分3類，正常，細菌感染，病毒感染。

There are the classes for image classification, which are normal, bacteria, virus.

```
IMG_SIZE = 200
BATCH_SIZE = 64

all_class = ['normal', 'bacteria', 'virus']
class_map = {cls:i for i,cls in enumerate(all_class)} #  'normal':0, 'bacteria': 1, 'virus':2
class_map

```
<img width="464" alt="lung_xray" src="https://github.com/chency0315/pneumonia_classification/assets/100465252/a56df749-90ce-43d1-bbf9-a4234754ea40">

#  build the model

## 使用VGG19當作基底模型，接上GlobalAveragePooling2D，1024全連接層，激勵函數輸出為relu, 利用dropout砍去一半的nodes, 來避免過度擬合。最後利用softmax來分出3個類別。此外將VGG19層數凍結可以節省訓練時間。

For the model, I used VGG19 as the base model and then added a few layers for classifying using relu as an activation function, dropout 0.5 to avoid overfitting,
at last, used softmax as an activation function to classify into three classes. In addition, freezing the layers of VGG19 saves training time.
```

tf.keras.backend.clear_session()
base_model = tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(1024, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(100, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.BatchNormalization()(x)
pred = layers.Dense(3, activation='softmax')(x)

for l in base_model.layers: #freeze Vgg19 all layers
    l.trainable = False
model3 = models.Model(base_model.input,pred)
```

## 應用learning rate scheduler來調整learning rate, 訓練20 epoches 以後會以exp(-0.1)的學習率做梯度下降。

Apply learning rate scheduler, and then after 20 epochs learning rate will be calculated with this formula.

```
def scheduler(epoch, lr):
       if epoch < 20:
            return lr
       else:
            return lr * tf.math.exp(-0.1) # after 20 epoches learning rate will be calculated with this formula

model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(0.01),
              metrics=['accuracy'])
```

# plot the categorical accuracy and loss figure

![output](https://github.com/chency0315/pneumonia_classification/assets/100465252/6b4ae209-3f49-4dff-8d20-3d9a4df6f010)

驗證的圖片準確值為:0.79

The accuracy score is 0.79

<img width="328" alt="f1score" src="https://github.com/chency0315/pneumonia_classification/assets/100465252/74aada71-c947-46ec-9ba8-c5f8cf7aaaed">


