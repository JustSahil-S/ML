import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
data = np.loadtxt('/Users/sahil/ML/airfoil_self_noise.dat')
print(data)
x_train, x_test, y_train, y_test = train_test_split(data[:, :-1], data[:, -1], test_size=0.2, random_state=1)
my_adam = tf.optimizers.Adam(learning_rate=.01)
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(50, activation='sigmoid', input_shape=(5,)),
  tf.keras.layers.Dropout(0.0),
  tf.keras.layers.Dense(50, activation='sigmoid'),
  tf.keras.layers.Dropout(0.0),
  tf.keras.layers.Dense(50, activation='sigmoid'),
  tf.keras.layers.Dropout(0.0),
  tf.keras.layers.Dense(10, activation='softmax'),
])
model.compile(optimizer=my_adam, loss="mean_squared_error") 
history = model.fit(x_train, y_train, epochs=360, batch_size=256, verbose=1)
print(f'Error: {np.sqrt(model.evaluate(x_test, y_test, batch_size=256))}')
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()