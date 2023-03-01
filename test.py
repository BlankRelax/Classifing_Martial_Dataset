import matplotlib.pyplot as plt
import tensorflow as tf
import warnings
import numpy as np
warnings.filterwarnings("ignore")
from tensorflow import keras



def getCFAR10():
    print("\033[92mLoading the CFAR10 data from keras\033[0m")
    # Load the CFAR10 data via the tensorflow keras dataset:
    cifar10 = keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = getCFAR10()
print(x_train)
plt.imshow(x_train[120])
plt.show()