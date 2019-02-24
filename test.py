import numpy as np
from keras.preprocessing.image import load_img,img_to_array
from keras.models import load_model
import os
#Testing

label = np.array(['apple','cat','horse','orange'])
model = load_model('./model/classification.h5')
file_name = os.listdir('./image/test/orange/')
for name in enumerate(file_name):
    image = load_img('./image/test/orange/'+name[1])
    image = image.resize((256,256))
    image = img_to_array(image)
    image = image/255
    image = np.expand_dims(image,0)
    print(image.shape)

    print(label[model.predict_classes(image)])
