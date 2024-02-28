#### Libraries
# Standard library
import gzip
import pickle as cPickle

# Third-party libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

#### Constants
GPU = True
if GPU:
    print("Trying to run under a GPU. If this is not desired, then modify " + \
        "network3.py\nto set the GPU flag to False.")
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("Running with a CPU. If this is not desired, then modify " + \
        "network3.py to set\nthe GPU flag to True.")

#### Load the MNIST data
def load_data_shared(filename="./data/mnist.pkl.gz"):
    f = gzip.open(filename, 'rb')
    training_data, validation_data, test_data = cPickle.load(f, encoding='latin1')
    f.close()
    return [training_data, validation_data, test_data]

#### Main class used to construct and train networks
class Network(object):
    def __init__(self, layers, mini_batch_size):
        """Takes a list of `layers`, describing the network architecture, and
        a value for the `mini_batch_size` to be used during training
        by stochastic gradient descent.

        """
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.model = None

    def build_model(self):
        model = models.Sequential()
        for layer in self.layers:
            model.add(layer)
        self.model = model

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            validation_data, test_data, lmbda=0.0):
        """Train the network using mini-batch stochastic gradient descent."""
        training_x, training_y = training_data
        validation_x, validation_y = validation_data
        test_x, test_y = test_data

        training_y = to_categorical(training_y)
        validation_y = to_categorical(validation_y)
        test_y = to_categorical(test_y)

        self.model.compile(optimizer='sgd',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

        self.model.fit(training_x, training_y,
                       batch_size=mini_batch_size,
                       epochs=epochs,
                       validation_data=(validation_x, validation_y))

        test_loss, test_accuracy = self.model.evaluate(test_x, test_y)
        print("Test accuracy: {:.2%}".format(test_accuracy))

    def predict(self, image):
        prediction = self.model.predict(np.array([image]))
        return np.argmax(prediction)

#### Define layer types
class ConvPoolLayer(layers.Layer):
    def __init__(self, filter_shape, pool_size=(2, 2),
                 activation_fn='sigmoid'):
        super(ConvPoolLayer, self).__init__()
        self.filter_shape = filter_shape
        self.pool_size = pool_size
        self.activation_fn = activation_fn

    def build(self, input_shape):
        self.conv2d = layers.Conv2D(filters=self.filter_shape[0],
                                    kernel_size=(self.filter_shape[2], self.filter_shape[3]),
                                    activation=self.activation_fn,
                                    padding='valid')
        self.maxpool = layers.MaxPooling2D(pool_size=self.pool_size,
                                           strides=self.pool_size)

    def call(self, inputs):
        conv_output = self.conv2d(inputs)
        pooled_output = self.maxpool(conv_output)
        return pooled_output

class FullyConnectedLayer(layers.Layer):
    def __init__(self, n_in, n_out, activation_fn='sigmoid'):
        super(FullyConnectedLayer, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn

    def build(self, input_shape):
        self.dense = layers.Dense(self.n_out, activation=self.activation_fn)

    def call(self, inputs):
        output = self.dense(inputs)
        return output

#### Network architecture
def network_architecture():
    network_layers = [
        ConvPoolLayer(filter_shape=(20, 1, 5, 5)),
        ConvPoolLayer(filter_shape=(40, 20, 5, 5)),
        FullyConnectedLayer(n_in=40*4*4, n_out=100),
        FullyConnectedLayer(n_in=100, n_out=10, activation_fn='softmax')
    ]
    return network_layers

#### Main function to build and train the network
def main():
    # Load MNIST data
    training_data, validation_data, test_data = load_data_shared()

    # Reshape and normalize the data
    training_x, training_y = training_data
    validation_x, validation_y = validation_data
    test_x, test_y = test_data

    training_x = training_x.reshape((-1, 1, 28, 28))
    validation_x = validation_x.reshape((-1, 1, 28, 28))
    test_x = test_x.reshape((-1, 1, 28, 28))

    training_x = training_x.astype('float32') / 255
    validation_x = validation_x.astype('float32') / 255
    test_x = test_x.astype('float32') / 255

    # Define network architecture
    layers = network_architecture()

    # Create and compile the network
    network = Network(layers=layers, mini_batch_size=10)
    network.build_model()

    # Train the network
    network.SGD(training_data=(training_x, training_y),
                epochs=60,
                mini_batch_size=10,
                eta=0.1,
                validation_data=(validation_x, validation_y),
                test_data=(test_x, test_y))

if __name__ == '__main__':
    main()

