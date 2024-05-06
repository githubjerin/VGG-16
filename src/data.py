import tensorflow as tf

train_ds = tf.keras.utils.image_dataset_from_directory(
  '../assets/flowers',
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(224, 224),
  batch_size=10)

for x, y in train_ds.take(1):
    print(x.shape, y)