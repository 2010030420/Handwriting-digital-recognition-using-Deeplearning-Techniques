import np as np
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.optimizers import Adam

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Define and compile CNN model
cnn_model = Sequential()
cnn_model.add(Flatten(input_shape=(28, 28)))
cnn_model.add(Dense(128, activation='relu'))
cnn_model.add(Dense(10, activation='softmax'))
cnn_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train CNN model
cnn_model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=30)

# Define GAN
generator = Sequential()
generator.add(Dense(128, input_dim=100))
generator.add(LeakyReLU(alpha=0.01))
generator.add(Dense(784, activation='sigmoid'))
generator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))

discriminator = Sequential()
discriminator.add(Flatten(input_shape=(28, 28)))
discriminator.add(Dense(128, activation='relu'))
discriminator.add(Dense(1, activation='sigmoid'))
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])

discriminator.trainable = False
gan_input = Input(shape=(100,))
x = generator(gan_input)
gan_output = discriminator(x)
gan = Model(gan_input, gan_output)
gan.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))

# Training GAN
batch_size = 64
total_batches = len(x_train) // batch_size

for epoch in range(30):
    for batch in range(total_batches):
        noise = np.random.normal(0, 1, (batch_size, 100))
        synthetic_images = generator.predict(noise)

        x_combined_batch = np.concatenate([x_train[batch_size * batch: batch_size * (batch + 1)], synthetic_images])
        y_combined_batch = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])

        d_loss = discriminator.train_on_batch(x_combined_batch, y_combined_batch)

        noise = np.random.normal(0, 1, (batch_size, 100))
        y_mislabeled = np.ones((batch_size, 1))
        g_loss = gan.train_on_batch(noise, y_mislabeled)

# Evaluate CNN on test data
cnn_evaluation = cnn_model.evaluate(x_test, y_test)
print("CNN Model Evaluation Loss:", cnn_evaluation[0])
print("CNN Model Evaluation Accuracy:", cnn_evaluation[1])
