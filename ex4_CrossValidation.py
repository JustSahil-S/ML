import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

data = np.loadtxt('/Users/sahil/ML/airfoil_self_noise.dat')
print(data)
input = data[:, :-1]
output = data[:, -1]
kf = KFold(n_splits=5, shuffle=True)
folds = kf.split(input)
my_adam = tf.optimizers.Adam(learning_rate=.01)
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(50, activation='sigmoid', input_shape=(5,)),
  tf.keras.layers.Dropout(0.15),
  tf.keras.layers.Dense(50, activation='sigmoid'),
  tf.keras.layers.Dropout(0.15),
  tf.keras.layers.Dense(50, activation='sigmoid'),
  tf.keras.layers.Dropout(0.15),
  tf.keras.layers.Dense(1, activation=None),
])
ErrorList = []
for i, (train_index, test_index) in enumerate(folds):
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}")
    print(f"  Test:  index={test_index}")
    x_train, x_test = input[train_index], input[test_index]
    y_train, y_test = output[train_index], output[test_index]
    model.compile(optimizer=my_adam, loss="mean_squared_error") 
    history = model.fit(x_train, y_train, epochs=360, batch_size=256, verbose=1)
    print(f'Error: {np.sqrt(model.evaluate(x_test, y_test, batch_size=256))}')
    ErrorList.append(np.sqrt(model.evaluate(x_test, y_test, batch_size=256)))
print(ErrorList)
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()