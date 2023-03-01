import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
def get_train_dat():
    image_classes = np.loadtxt('C:\\Users\Hassaan\\PycharmProjects\\Classifing_Martial_Dataset\\labels-map-proj-v3.txt',dtype=str)
    df = pd.DataFrame(image_classes, columns=['filename', 'class'])
    idg = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255.)
    train_ds = idg.flow_from_dataframe(dataframe=df,directory='C:\\Users\Hassaan\\PycharmProjects\\Classifing_Martial_Dataset\\map-proj-v3', batch_size=7495)
    return(train_ds)


train_ds = get_train_dat()

x_train=np.array(train_ds[0][0])
y_train=np.array(train_ds[0][0])
# d = {'Images': x_train, 'True Class Labels': y_train}
# df = pd.DataFrame(data=d)
# df.to_pickle("dat.pkl")
print(x_train.shape)
print(y_train.shape)
print(x_train[7496])

