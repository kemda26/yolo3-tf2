import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
import cv2

# from yolo.net.darknet import DarkNet
# from yolo.net.headnet import Headnet
# from yolo.net.weights import WeightReader
# from yolo.net.mobilenet import MobileNet
# from yolo.net.effnet import EfficientNet
# from yolo.net.resnet import ResNet50

from mobilenet import MobileNet
from darknet import DarkNet
from headnet import Headnet
from weights import WeightReader
from effnet import EfficientNet
from resnet import ResNet50


# Yolo v3
class Yolonet(tf.keras.Model):
    def __init__(self, n_classes=10, arch='resnet50'):
        super(Yolonet, self).__init__(name='')
        
        print('using %s backbone' % arch)
        if arch == 'mobilenet':
            self.body = MobileNet(input_shape=(224,224,3))
        elif 'efficientnet' in arch:
            self.body = EfficientNet(arch, pretrained='imagenet')
        elif arch == 'resnet50':
            self.body = ResNet50(input_shape=(224,224,3))
        else:
            self.body = DarkNet()

        self.head = Headnet(n_classes)

        self.num_layers = 110
        self._init_vars()


    def load_darknet_params(self, weights_file, skip_detect_layer=False):
        weight_reader = WeightReader(weights_file)
        weight_reader.load_weights(self, skip_detect_layer)
    

    def predict(self, input_array):
        f5, f4, f3 = self.call(tf.constant(input_array.astype(np.float32)))
        # f5, f4, f3 = self.call(tf.constant(input_array))
        return f5.numpy(), f4.numpy(), f3.numpy()


    def call(self, input_tensor, training=False):
        s3, s4, s5 = self.body(input_tensor, training)
        # print('shape',s3.shape, s4.shape, s5.shape)
        f5, f4, f3 = self.head(s3, s4, s5, training)
        return f5, f4, f3


    def get_variables(self, layer_idx, suffix=None):
        if suffix:
            find_name = "layer_{}/{}".format(layer_idx, suffix)
        else:
            find_name = "layer_{}/".format(layer_idx)
        variables = []
        for v in self.variables:
            if find_name in v.name:
                variables.append(v)
        return variables


    def _init_vars(self):
        sample = tf.constant(np.random.randn(1, 224, 224, 3).astype(np.float32))
        self.call(sample, training=False)


    def fit_generator(self):
        pass


def preprocess_input(image, net_size):
    """
    # Args
        image : array, shape of (H, W, 3)
            RGB-ordered
        net_size : int
    """
    # resize the image to the new size
    preprocess_img = cv2.resize(image / 255., (net_size, net_size))
    return np.expand_dims(preprocess_img, axis=0)


if __name__ == '__main__':
    inputs = tf.constant(np.random.randn(1, 224, 224, 3).astype(np.float32))
    # print(inputs.shape)

    # (1, 256, 256, 3) => (1, 8, 8, 1024)
    yolonet = Yolonet()
    f5, f4, f3 = yolonet(inputs)
    # print(f5.shape, f4.shape, f3.shape)
    
    # print(yolonet.summary())
    # for v in yolonet.variables:
        # print(v.name)