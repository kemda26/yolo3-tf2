import tensorflow as tf

class ResNet50(tf.keras.Model):
    def __init__(self, input_shape=(224, 224, 3)):
        super(ResNet50, self).__init__()

        self.net = tf.keras.applications.resnet50.ResNet50(input_shape=input_shape,
                                                            include_top=False,
                                                            weights='imagenet',
                                                            classes=10,
                                                            pooling=None)
                                                                    
        self.feature_3 = self.net.get_layer('activation_21').output
        self.feature_2 = self.net.get_layer('activation_39').output
        self.feature_1 = self.net.get_layer('activation_48').output

        self.feature_extractor = tf.keras.Model(inputs=self.net.input, 
                                            outputs=[self.feature_3,
                                                     self.feature_2, 
                                                     self.feature_1],
                                            name='resnet50_extractor')

        # print(self.net.summary())


    def call(self, inputs, training=False):
        f3, f2, f1 = self.feature_extractor(inputs)

        return f3, f2, f1

if __name__ == '__main__':
    import numpy as np
    input_shape = (224, 224, 3)
    inputs = tf.constant(np.random.randn(1, *input_shape).astype(np.float32))

    net = ResNet50()
    x,y,z = net(inputs)
    print(x.shape)
    print(y.shape)
    print(z.shape)

        