import tensorflow as tf
from keras import models, layers
import glob
import numpy as np
import matplotlib.pyplot as plt
import imageio
import imgaug as ia
import imgaug.augmenters as iaa
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

x_data = []
x_train = []
img_idx = 0
num_positives = 0
num_blanks = 0
num_augmented = 0
augment = True

def filer(files):
    for myFile in files:
        global img_idx, num_blanks, num_positives, num_augmented, augment
        global x_train, x_target_data
        img = imageio.imread(myFile)
        x_train.append(x_target_data[img_idx])
        """
        if img.shape[-1] == 4:  # Check if the image has an alpha channel (RGBA#)
            img = img[:, :, :3]  # Convert RGBA to RGB by removing the alpha channel

        if img.shape[:2] != (2487, 318):
            img_resized = tf.image.resize(img, size=(2487, 318))  # Resize the image to match the input shape
        else:
            img_resized = img
        """
        img_swapped = np.swapaxes(img, 0, 1)  # Swap height and width
        x_data.append(img_swapped)
        if x_target_data[img_idx] == 1 :
            num_positives += 1
            if augment :
                # Rotate image within a narrow range of angles
                rotate=iaa.Affine(rotate=(-10, 10))
                rotated_image=rotate.augment_image(img)
                x_data.append(np.swapaxes(rotated_image, 0, 1))
                x_train.append(1)
                # Flipping image horizontally
                flip_hr=iaa.Fliplr(p=1.0)
                flip_hr_image= flip_hr.augment_image(img)
                x_data.append(np.swapaxes(flip_hr_image, 0, 1))
                x_train.append(1)
                num_augmented += 2
        else:
            num_blanks += 1
        img_idx += 1


x_target_data = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,1,1,0,0,1,0,0,1,0,1,1,0,0,0,0,1,1,0,0,0,1,0,1,1,0,0,0,0,0,1,1,0,0,0,1,1,0,0,1,1,0,0,1,1,0,0,0,0,0,0,0,1,0,1,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0,1,0,1,1,0,0,0,0,1,0,0,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,1,1,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,1,1,0,0,0,0,1,0,0,0,0,1,0,1,1,0,0,0,0,0,1,1,0,0,0,1,1,0,0,1,1,0,0,1,1,0,0,0,0,0,0,1,1,0,1,0,0,1,0,1,0,0,1,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0,1,1,1,1,0,0,0,0,1,0,0,1,1,0,1,0,0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,0,0,1,1,0,1,1,1,1,0,0,0,0,0,1,0,1,1,0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1,0,0,1,1,0,1,1,0,0,0,1,1,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,1,1,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,
                    0,0,0,0,0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,1,0,0,1,0,0,0,0,1,1,0,0,0,0,1,0,1,1,0,0,0,0,0,1,1,0,0,0,1,1,0,0,1,1,0,0,1,1,0,0,0,0,0,0,1,1,0,1,1,0,1,0,1,0,0,1,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0,1,0,1,1,0,0,0,0,1,0,0,1,1,1,1,0,0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,1,0,1,1,0,1,0,0,0,0,0,0,0,0,1,0,0,1,1,0,0,0,0,1,0,0,1,1,0,1,1,0,0,0,1,1,1,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1,0,0,0,0,1,0,1,1,0,0,0,0,0,1,1,0,0,0,1,0,0,0,1,1,0,0,1,1,0,0,1,0,0,0,1,1,0,1,1,0,1,0,0,0,0,1,0,0,0,0,0,0,0,1,1,1,0,1,1,0,0,1,1,1,1,0,0,0,0,1,0,0,1,1,1,1,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,1,0,1,0,0,0,1,0,1,1,0,1,1,0,0,0,0,0,0,0,1,0,0,1,1,0,0,0,0,1,1,0,1,1,0,1,1,0,0,0,1,1,1,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,
                    0,0,0,0,0,0,0,1,0,0,0,0,0,1,1,0,0,1,0,1,0,0,0,1,0,0,1,1,0,1,1,0,0,1,0,1,1,0,0,0,1,1,0,0,0,1,1,0,1,1,0,0,1,0,0,1,1,0,0,0,1,0,0,0,1,1,0,0,1,1,0,0,1,0,0,0,1,1,0,1,1,0,1,0,0,0,0,1,0,0,0,0,0,0,0,1,1,1,0,1,1,0,0,1,1,1,1,0,0,0,0,1,0,0,1,1,1,1,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,1,0,1,1,0,1,1,0,0,0,0,0,0,0,1,1,0,1,1,0,0,0,0,1,1,0,1,1,0,1,1,0,0,0,1,1,1,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0]
files = glob.glob("60N/*.jpg")
filer(files)
files = glob.glob("80N/*.jpg")
filer(files)
files = glob.glob("100N/*.jpg")
filer(files)
files = glob.glob("120N/*.jpg")
filer(files)
files = glob.glob("140N/*.jpg")
filer(files)
files = glob.glob("160N/*.jpg")
filer(files)
files = glob.glob("180N/*.jpg")
filer(files)
print(f'num_positives: {num_positives}, num_blanks: {num_blanks}, num_augmented: {num_augmented}')
my_adam = tf.optimizers.legacy.Adam(learning_rate = 0.00001)

x_data = np.array(x_data)
x_train = np.array(x_train)
permutation = np.random.permutation(len(x_train))
x_train, x_test, y_train, y_test = train_test_split(x_data, x_train, test_size=0.2, random_state=69)
input_shape = (157, 2560, 4)

model = Sequential([
    Conv2D(16, (3, 3), input_shape=input_shape, padding='same', activation='relu'),
    Conv2D(16, (3, 3), activation='relu', padding='same', name="firesmokenet/conv1_2"),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

    Conv2D(16, (3, 3), activation='relu', padding='same', name="firesmokenet/conv2_1"),
    Conv2D(16, (3, 3), activation='relu', padding='same', name="firesmokenet/conv2_2"),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

    Conv2D(32, (3, 3), activation='relu', padding='same', name="firesmokenet/conv3_1"),
    Conv2D(32, (3, 3), activation='relu', padding='same', name="firesmokenet/conv3_2"),
    Conv2D(32, (3, 3), activation='relu', padding='same', name="firesmokenet/conv3_3"),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

    Conv2D(256, (3, 3), activation='relu', padding='same', name="firesmokenet/conv4_1"),
    Conv2D(256, (3, 3), activation='relu', padding='same', name="firesmokenet/conv4_2"),
    Conv2D(256, (3, 3), activation='relu', padding='same', name="firesmokenet/conv4_3"),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

    Flatten(),
    Dense(512, activation='relu', name="firesmokenet/fc6"),
    Dense(256, activation='relu', name="firesmokenet/fc7"),
    #adjust for num_classes below
    Dense(2, activation='softmax', name="firesmokenet/fc8")
])

opt = Adam(learning_rate=0.0001, weight_decay=0.0001)
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])



model.summary()

model.compile(optimizer=my_adam,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=50, verbose=True, batch_size=32, validation_data=(x_test, y_test))

print(f'Model Loss: {(model.evaluate(x_test, y_test, batch_size=32))}')

print(y_test)
print(model.predict(x_test))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
