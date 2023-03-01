import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def get_train_dat():
    image_classes = np.loadtxt('C:\\Users\Hassaan\\PycharmProjects\\Classifing_Martial_Dataset\\labels-map-proj-v3.txt',dtype=str)
    df = pd.DataFrame(image_classes, columns=['filename', 'class'])
    idg = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255.)
    train_ds = idg.flow_from_dataframe(dataframe=df,directory='C:\\Users\Hassaan\\PycharmProjects\\Classifing_Martial_Dataset\\map-proj-v3', batch_size=7495)
    return(train_ds)


train_ds = get_train_dat()
x_train_df, x_test_df = train_test_split(train_ds[0][0], test_size=0.4) # this accesses the 0th(one and only batch) and the images and then splits this into train and test
y_train_df, y_test_df = train_test_split(train_ds[0][1], test_size=0.4)  # this accesses the 0th(one and only batch) and the true class labels and then splits this into train and test
#print(y_train_df[0])
#print(x_train_df[0])
print(y_train_df.shape)
print(x_train_df.shape)
print(type(x_train_df))
print(type(y_train_df))

# plt.imshow(x_train_df[0])
# plt.show()

xpixels = x_train_df.shape[1] # 256 pixels in x direction
ypixels = x_train_df.shape[2] # 256 pixels in y direction
rgb_pixel = x_train_df.shape[3] # 3 numbers for each pixel for RGB channels

ValidationSplit=0.5
BatchSize=10
Nepochs=5
DropoutValue=0.6

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(xpixels, ypixels, rgb_pixel)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))

model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))

model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(8, activation='softmax'))

print("--------------------------------------------------------------------------------------------------------------")
print("\033[92mWill train a convolutional neural network on the Martian data\033[0m")
print("--------------------------------------------------------------------------------------------------------------\n\n")
print("Input data Martian")
print("Dropout values       = ", DropoutValue)
print("Leaky relu parameter =  0.1")
print("ValidationSplit      = ", ValidationSplit)
print("BatchSize            = ", BatchSize)
print("Nepochs              = ", Nepochs, "\n")
print("N(train)             = ", len(x_train_df))
print("N(test)              = ", len(x_test_df))
model.summary()

loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)

model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
history=model.fit(x_train_df, y_train_df, validation_split=ValidationSplit, batch_size=BatchSize, epochs=Nepochs)

loss, acc = model.evaluate(x_test_df,  y_test_df, verbose=2)
print("\tloss = {:5.3f}\n\taccuracy = {:5.3f}".format(loss, acc))