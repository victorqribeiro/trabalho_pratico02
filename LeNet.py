import sys
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models, losses

if len(sys.argv) < 3:
    print("Please inform dataset and number of epochs")
    print("Usage: LeNet.py dataset num_epochs")
    exit()

if (sys.argv[1] != 'cifar' and sys.argv[1] != 'fashion'):
    print('unkwown dataset')
    exit()

if (not sys.argv[2].isnumeric()):
    print('epochs must be an integer')
    exit()

epochs = int(sys.argv[2])
batch_size = int(sys.argv[3])

if (sys.argv[1] == 'cifar'):
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    x_train = np.mean(x_train, axis=-1, keepdims=True)
    x_test = np.mean(x_test, axis=-1, keepdims=True)
else:
    (x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()
    x_train = x_train[:, :, :, np.newaxis]
    x_test = x_test[:, :, :, np.newaxis]

x_val = x_train[-2000:, :, :, :]
y_val = y_train[-2000:]
x_train = x_train[:-2000, :, :, :]
y_train = y_train[:-2000]

model = models.Sequential()
model.add(layers.Conv2D(6, 5, activation='tanh',
          input_shape=x_train.shape[1:], padding="same"))
model.add(layers.AveragePooling2D(2))
model.add(layers.Activation('sigmoid'))
model.add(layers.Conv2D(16, 5, activation='tanh'))
model.add(layers.AveragePooling2D(2))
model.add(layers.Activation('sigmoid'))
model.add(layers.Conv2D(120, 5, activation='tanh'))
model.add(layers.Flatten())
model.add(layers.Dense(84, activation='tanh'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()

model.compile(optimizer='adam', loss=losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=batch_size,
                    epochs=epochs, validation_data=(x_val, y_val))
model.evaluate(x_test, y_test)

fig, axs = plt.subplots(2, 1, figsize=(15, 15))

axs[0].plot(history.history['loss'])
axs[0].plot(history.history['val_loss'])
axs[0].title.set_text('Training Loss vs Validation Loss')
axs[0].legend(['Train', 'Val'])

axs[1].plot(history.history['accuracy'])
axs[1].plot(history.history['val_accuracy'])
axs[1].title.set_text('Training Accuracy vs Validation Accuracy')
axs[1].legend(['Train', 'Val'])

# plt.show()
plt.savefig("./figuras/" + sys.argv[0] + "-" + sys.argv[1] +
            "-" + str(epochs) + "-" + str(batch_size) + ".png")
