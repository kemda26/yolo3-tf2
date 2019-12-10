import tensorflow as tf
import numpy as np

class MobileNet(tf.keras.Model):
    def __init__(self, input_shape=(224, 224, 3)):
        super(MobileNet, self).__init__()

        self.mobilenet = tf.keras.applications.mobilenet.MobileNet(input_shape=input_shape,
                                                                   include_top=False,
                                                                   weights='imagenet',
                                                                   classes=10)
                                                                    
        self.feature_3 = self.mobilenet.get_layer('conv_pw_5_relu').output
        self.feature_2 = self.mobilenet.get_layer('conv_pw_11_relu').output
        self.feature_1 = self.mobilenet.get_layer('conv_pw_13_relu').output

        self.feature_extractor = tf.keras.Model(inputs=self.mobilenet.input, 
                                            outputs=[self.feature_3,
                                                     self.feature_2, 
                                                     self.feature_1],
                                            name='mobilenet_extractor')


    def call(self, inputs, training=False):
        f3, f2, f1 = self.feature_extractor(inputs)

        return f3, f2, f1
        

if __name__ == '__main__':
    inputs = tf.constant(np.random.randn(6, 224, 224, 3).astype(np.float32))
    print(inputs.shape)

    mobilenet = MobileNet(input_shape=(224,224,3))
    x, y, z= mobilenet(inputs)
    suma = mobilenet.summary()
    print(suma)
        
    # print(x.shape, y.shape, z.shape)
    # print(tf.executing_eagerly())
    # print(x.numpy())