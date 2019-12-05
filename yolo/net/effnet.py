from efficientnet.tfkeras import EfficientNetB0
import tensorflow as tf
import numpy as np

class EfficientNet(tf.keras.Model):
    def __init__(self, input_shape=(224, 224, 3), pretrained=None):
        super(EfficientNet, self).__init__()

        self.net = EfficientNetB0(include_top=False, 
                                  weights=pretrained,
                                  input_shape=input_shape,
                                  classes=10)
        
        self.feature_3 = self.net.get_layer('block4a_expand_activation').output
        self.feature_2 = self.net.get_layer('block6a_expand_activation').output
        self.feature_1 = self.net.get_layer('top_activation').output

        self.feature_extractor = tf.keras.Model(inputs=self.net.input,
                                                outputs=[self.feature_3,
                                                        self.feature_2,
                                                        self.feature_1])

        # print(self.net.summary())


    def call(self, tensor, training=False):
        f3, f2, f1 = self.feature_extractor(tensor)
        return f3, f2, f1


if __name__ == '__main__':
    inputs = tf.constant(np.random.randn(6, 224, 224, 3).astype(np.float32))

    net = EfficientNet()
    f3,f2,f1 = net(inputs)
    print(f3.shape)
    print(f2.shape)
    print(f1.shape)