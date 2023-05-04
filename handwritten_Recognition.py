import os
import time
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt


model_not_exist=False

if model_not_exist:
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Normalize the pixel values from [0, 255] to [0, 1]
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    model = models.Sequential([
        layers.InputLayer(input_shape=(28, 28)),
        layers.Reshape((28, 28, 1)),
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(train_images, train_labels, epochs=10, validation_split=0.1)

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f'Test accuracy: {test_acc}')

    model.save('my_model')
else:
    # Load the model
    model = tf.keras.models.load_model('my_model')

    # Load custom images and predict them
n = 1
while os.path.isfile('numbers/num{}.png'.format(n)):
    try:
        img = cv2.imread('numbers/num{}.png'.format(n))[:, :, 0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print("The number is probably a {}".format(np.argmax(prediction)))
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
        time.sleep(3)
        n += 1
    except:
        print("Alert reading ! next image -----")
        n += 1