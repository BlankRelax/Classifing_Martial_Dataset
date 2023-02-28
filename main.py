import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
def get_train_dat():
    image_classes = np.loadtxt('C:\\Users\Hassaan\\PycharmProjects\\Classifing_Martial_Dataset\\labels-map-proj-v3.txt',dtype=str)
    df = pd.DataFrame(image_classes, columns=['filename', 'class'])
    idg = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255.)
    train_ds = idg.flow_from_dataframe(dataframe=df,directory='C:\\Users\Hassaan\\PycharmProjects\\Classifing_Martial_Dataset\\map-proj-v3')
    return(train_ds)


train_ds = get_train_dat()
x_train= train_ds[0][0][0] # this prints the image, [0][0][32] is out of index because the batch size is 32
# access the different batches from the first index of train_ds
y_train= train_ds[0][1][0] # this tells us the class of the image

print(train_ds[0][1][0])

plt.imshow(train_ds[4][0][31])
plt.show()
