from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dropout,Flatten,Dense
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img
import h5py
from keras.models import load_model
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


#layer
model = Sequential()
model.add(Conv2D(input_shape=(256,256,3),filters=64,kernel_size=3,padding='same',activation='relu'))
model.add(Conv2D(filters=64,kernel_size=3,padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=2,strides=2))

model.add(Conv2D(filters=128,kernel_size=3,padding='same',activation='relu'))
model.add(Conv2D(filters=128,kernel_size=3,padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=2,strides=2))

model.add(Conv2D(filters=256,kernel_size=3,padding='same',activation='relu'))
model.add(Conv2D(filters=256,kernel_size=3,padding='same',activation='relu'))
model.add(Conv2D(filters=256,kernel_size=3,padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=2,strides=2))

model.add(Conv2D(filters=512,kernel_size=3,padding='same',activation='relu'))
model.add(Conv2D(filters=512,kernel_size=3,padding='same',activation='relu'))
model.add(Conv2D(filters=512,kernel_size=3,padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=2,strides=2))

model.add(Conv2D(filters=512,kernel_size=3,padding='same',activation='relu'))
model.add(Conv2D(filters=512,kernel_size=3,padding='same',activation='relu'))
model.add(Conv2D(filters=512,kernel_size=3,padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=2,strides=2))

model.add(Flatten())
model.add(Dense(4096,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4,activation='softmax'))

#Optimizer
adam = Adam(1e-4)

model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rotation_range=40,#旋转
    width_shift_range=0.2, #水平移动
    height_shift_range=0.3, #竖直移动
    rescale=1/255,#数据归一化
    shear_range=0.2,#随机裁剪
    zoom_range=0.2,#随机放大
    horizontal_flip=True,#水平翻转
    fill_mode='nearest'#填充方式

)
test_datagen = ImageDataGenerator(
    rescale=1/255
)


#生成训练图片
train_genetor = train_datagen.flow_from_directory(
    './image/train/',
    target_size=(256,256),
    batch_size=32
)
#测试图片
test_genetor = test_datagen.flow_from_directory(
    './image/test/',
    target_size=(256,256),
    batch_size=32
)
print(train_genetor.class_indices)

model.fit_generator(train_genetor,steps_per_epoch=32,epochs=1,validation_data=test_genetor,validation_steps=1)
model.save('./model/classification.h5')

