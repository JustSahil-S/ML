import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
data = np.loadtxt('/Users/sahil/ML/airfoil_self_noise.dat')
print(data)
x_train, x_test, y_train, y_test = train_test_split(data[:, :-1], data[:, -1], test_size=0.2, random_state=1)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=1)
grid = [[0.1, 5], [0.1, 50], [0.1, 250], [0.01, 250], [0.1, 50], [0.1, 5], [0.001, 250], [0.001, 50], [0.001, 5]]
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(50, activation='sigmoid', input_shape=(5,)),
  tf.keras.layers.Dropout(0.15),
  tf.keras.layers.Dense(50, activation='sigmoid'),
  tf.keras.layers.Dropout(0.15),
  tf.keras.layers.Dense(50, activation='sigmoid'),
  tf.keras.layers.Dropout(0.15),
  tf.keras.layers.Dense(1, activation=None),
])
RMSElist = []
Validationlist = []
for i in range(0, len(grid)):
  alpha = grid[i][0]
  epochs = grid[i][1]
  my_adam = tf.optimizers.Adam(learning_rate=alpha)
  model.compile(optimizer=my_adam, loss="mean_squared_error") 
  history = model.fit(x_train, y_train, epochs=epochs, batch_size=256, verbose=False)
  print(f'Error: {np.sqrt(model.evaluate(x_test, y_test, batch_size=256))}')
  RMSElist.append(np.sqrt(model.evaluate(x_test, y_test, batch_size=256)))
  Validationlist.append(np.sqrt(model.evaluate(x_val, y_val, batch_size=256)))

print(RMSElist)
print(Validationlist)
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()