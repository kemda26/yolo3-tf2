from efficientnet.tfkeras import EfficientNetB0, EfficientNetB2
import tensorflow as tf
import numpy as np

class EfficientNet(tf.keras.Model):
    def __init__(self, arch, pretrained=None):
        super(EfficientNet, self).__init__()

        if arch == 'efficientnet-b0':
            self.net = EfficientNetB0(include_top=False, 
                                    weights=pretrained,
                                    input_shape=(224, 224, 3),
                                    classes=10)
        elif arch == 'efficientnet-b2':
            self.net = EfficientNetB2(include_top=False, 
                                    weights=pretrained,
                                    input_shape=(260, 260, 3),
                                    classes=10)
        
        self.feature_3 = self.net.get_layer('block4a_expand_activation').output
        self.feature_2 = self.net.get_layer('block6a_expand_activation').output
        self.feature_1 = self.net.get_layer('top_activation').output

        self.feature_extractor = tf.keras.Model(inputs=self.net.input,
                                                outputs=[self.feature_3,
                                                        self.feature_2,
                                                        self.feature_1])

        print(self.net.summary())


    def call(self, tensor, training=False):
        f3, f2, f1 = self.feature_extractor(tensor)
        return f3, f2, f1


if __name__ == '__main__':
    input_shape = (260, 260, 3)
    inputs = tf.constant(np.random.randn(1, *input_shape).astype(np.float32))

    net = EfficientNet('efficientnet-b2')
    f3,f2,f1 = net(inputs)
    print(f3.shape)
    print(f2.shape)
    print(f1.shape)