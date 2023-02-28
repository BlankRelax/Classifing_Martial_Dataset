import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

image_classes = np.loadtxt('C:\\Users\Hassaan\\PycharmProjects\\Classifing_Martial_Dataset\\labels-map-proj-v3.txt', dtype=str)
df = pd.DataFrame(image_classes,columns=['filename','class'])
idg = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.)
train_ds = idg.flow_from_dataframe(dataframe=df, directory='C:\\Users\Hassaan\\PycharmProjects\\Classifing_Martial_Dataset\\map-proj-v3',batch_size=32)


img1= train_ds[0][0][0][0]
img2=tf.reshape(img1,(32,24))

plt.imshow(img2, vmin=0,vmax=255)
plt.show()
