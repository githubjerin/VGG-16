import cv2
import keras
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#set seed
# tf.random.set_seed(0)

# train_ds = tf.keras.utils.image_dataset_from_directory(
#   '../assets/flowers',
#   validation_split=0.2,
#   subset="training",
#   seed=123,
#   image_size=(224, 224),
#   batch_size=10)


img = cv2.imread('../assets/flowers/sunflower/164670455_29d8e02bbd_n.jpg')
img = cv2.resize(img, (224, 224))
# cv2.imshow("Img", img)
# cv2.waitKey(0)

model = keras.Sequential()

#Block 1
model.add(Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu", input_shape=(224, 224, 3)))
model.add(Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D((2, 2),  strides=(2, 2)))

#Block 2
model.add(Conv2D(128, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(128, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D((2, 2),  strides=(2, 2)))

# #Block 3
# model.add(Conv2D(256, kernel_size=(3, 3), padding="same", activation="relu"))
# model.add(Conv2D(256, kernel_size=(3, 3), padding="same", activation="relu"))
# model.add(Conv2D(256, kernel_size=(3, 3), padding="same", activation="relu"))
# model.add(MaxPooling2D((2, 2),  strides=(2, 2)))

# #Block 4
# model.add(Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu"))
# model.add(Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu"))
# model.add(Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu"))
# model.add(MaxPooling2D((2, 2),  strides=(2, 2)))

# #Block 5
# model.add(Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu"))
# model.add(Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu"))
# model.add(Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu"))
# model.add(MaxPooling2D((2, 2),  strides=(2, 2)))

# #Top
# model.add(Flatten())
# model.add(Dense(4096, activation="relu"))
# model.add(Dense(4096, activation="relu"))
# model.add(Dense(42, activation="softmax"))

model.build()
model.summary()


# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# model.fit(train_ds, epochs=10)
# model.save('.')
#Result
result = model.predict(np.array([img]))


#Display feature map
for i in range(128):
    feature_img = result[0, :, :, i]
    ax = plt.subplot(8, 16, i + 1)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.imshow(feature_img, cmap="gray")
plt.show()