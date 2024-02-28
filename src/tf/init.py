from net import Network, CrossEntropyCost
import loader as mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = Network([784, 30, 10], cost=CrossEntropyCost)

net.SGD(training_data, 30, 10, 0.5, evaluation_data=test_data, monitor_evaluation_accuracy=True)
