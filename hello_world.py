from __future__ import absolute_import, division, print_function, unicode_literals


import tensorflow as tf

## Loading the MINIST dataset
mnist = tf.keras.datasets.mnist

## Converting the samples from integers to floating-point numbers
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

## Building the Sequential model by stacking layers. 
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

## 
predictions = model(x_train[:1]).numpy()
predictions

## The softmax function converts logits to propabilities for each class
tf.nn.softmax(predictions).numpy()

## The SparseCategoricalCrossentropy loss takes a vector of logits and a True index and returns a scalar loss
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

loss_fn(y_train[:1], predictions).numpy()

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)


model.evaluate(x_test,  y_test, verbose=2)

probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])

probability_model(x_test[:5])
