import tensorflow as tf
import sklearn as sklearn
from sklearn import datasets
from sklearn.datasets import fetch_california_housing
from numpy import random
from tensorflow import keras
from sklearn.model_selection import KFold
from keras.models import Model
from keras import metrics
import matplotlib.pyplot as plt
import numpy as np

'''
1.1 Load Housing Data
'''
data = fetch_california_housing()
output = data.target
input = data.data
print(f"Input Shape {input.shape}:")

'''
1.2 Prepare Data for CV
'''
kf = KFold(n_splits=10, shuffle=True, random_state=2)
folds = kf.split(input)

""" for i, (train_index, test_index) in enumerate(folds):
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}")
    print(f"  Test:  index={test_index}") """


my_adam = tf.optimizers.Adam(learning_rate=.001)

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(40, activation='relu', input_shape=(8,)),
  tf.keras.layers.Dense(1),
]) 
RMSElist = []
for i, (train_index, test_index) in enumerate(folds):
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}")
    print(f"  Test:  index={test_index}")
    x_train, x_test = input[train_index], input[test_index]
    y_train, y_test = output[train_index], output[test_index]
    model.compile(optimizer=my_adam, loss="mean_squared_error") 
    model.summary()
    history = model.fit(x_train, y_train, epochs=140, batch_size=32, verbose=1)
    RMSElist.append(np.sqrt(model.evaluate(x_test, y_test, batch_size=32)))
    print(f'Model RMSE: {np.sqrt(model.evaluate(x_test, y_test, batch_size=32))}')
print(f'ListofRMSE: {RMSElist}')
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
