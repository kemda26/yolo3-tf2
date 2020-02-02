import tensorflow as tf
tf.enable_eager_execution()
import json
import os
import glob
from yolo.net import Yolonet
from yolo.dataset.generator import BatchGenerator
from yolo.utils.utils import download_if_not_exists
from yolo.frontend import YoloDetector
from yolo.evaluate import Evaluator
import numpy as np
import h5py


class ConfigParser(object):
    def __init__(self, config_file):
        with open(config_file) as data_file:    
            config = json.load(data_file)
        
        self._model_config = config['model']
        self._arch = config['model']['arch']
        self._pretrained_config = config['pretrained']
        self._train_config = config['train']
        

    def _load_weights(self, model, filename):
        f = h5py.File(filename,'r')
        weights = []
        for i in range(len(f.keys())):
            weights.append(f['weight' + str(i)][:])
        f.close()
        model.set_weights(weights)


    def create_model(self, skip_detect_layer=True):
        model = Yolonet(n_classes=len(self._model_config['labels']), arch=self._arch)
        
        # tf_format = self._pretrained_config['tf_format']
        # if os.path.exists(tf_format):
        #     model.load_weights(tf_format)
        #     print('TF pretrained weights loaded from {}'.format(tf_format))

        keras_weights = self._pretrained_config['keras_format']
        if os.path.exists(keras_weights):
            self._load_weights(model, keras_weights)
            # model.load_weights(keras_weights, by_name=True)
            print('Keras pretrained weights loaded from {}'.format(keras_weights))
            
        elif self._arch == 'darknet' and not os.path.exists(keras_weights):
            download_if_not_exists(self._pretrained_config['darknet_format'],
                                   'https://pjreddie.com/media/files/yolov3.weights')

            model.load_darknet_params(self._pretrained_config['darknet_format'], skip_detect_layer)
            print('Original yolov3 weights loaded')

        return model


    def create_generator(self, split_train_valid=False):
        train_ann_fnames = self._get_train_anns()
        valid_ann_fnames = self._get_valid_anns()
        img_folder = 'valid_image_folder'

        if split_train_valid:
            train_valid_split = int(0.8*len(train_ann_fnames))
            np.random.seed(55)
            np.random.shuffle(train_ann_fnames)
            np.random.seed()

            img_folder = 'train_image_folder'
            train_ann_fnames, valid_ann_fnames = train_ann_fnames[:train_valid_split], train_ann_fnames[train_valid_split:]
            # valid_generator = None
        
        valid_generator = BatchGenerator(valid_ann_fnames,
                                        self._train_config[img_folder],
                                        batch_size=self._train_config['batch_size'],
                                        labels=self._model_config['labels'],
                                        anchors=self._model_config['anchors'],
                                        min_net_size=self._model_config['net_size'],
                                        max_net_size=self._model_config['net_size'],
                                        jitter=False,
                                        shuffle=False)

        train_generator = BatchGenerator(train_ann_fnames,
                                        self._train_config['train_image_folder'],
                                        batch_size=self._train_config['batch_size'],
                                        labels=self._model_config['labels'],
                                        anchors=self._model_config['anchors'],
                                        min_net_size=self._train_config['min_size'],
                                        max_net_size=self._train_config['max_size'],
                                        jitter=self._train_config['jitter'],
                                        shuffle=True)

        print('Training samples : {}, Validation samples : {}'.format(len(train_ann_fnames), len(valid_ann_fnames)))
        return train_generator, valid_generator


    def create_detector(self, model):
        detector = YoloDetector(model, self._model_config['anchors'], net_size=self._model_config['net_size'])
        return detector

    def create_evaluator(self, model):

        detector = self.create_detector(model)
        # train_ann_fnames = self._get_train_anns()
        test_ann_fnames = self._get_test_anns()

        # train_evaluator = Evaluator(detector,
        #                             self._model_config['labels'],
        #                             train_ann_fnames,
        #                             self._train_config['train_image_folder'])
        if len(test_ann_fnames) > 0:
            test_evaluator = Evaluator(detector,
                                        self._model_config['labels'],
                                        test_ann_fnames,
                                        self._train_config['test_image_folder'])
        else:
            test_evaluator = None
        return test_evaluator


    def get_train_params(self):
        learning_rate = self._train_config['learning_rate']
        save_dir = self._train_config['save_folder']
        weight_name = self._train_config['weight_name']
        num_epoches = self._train_config['num_epoch']
        checkpoint_path = self._train_config['checkpoint_path']
        return learning_rate, save_dir, weight_name, num_epoches, checkpoint_path


    def get_labels(self):
        return self._model_config['labels']
    

    def _get_train_anns(self):
        ann_fnames = glob.glob(os.path.join(self._train_config['train_annot_folder'], '*.xml'))
        return ann_fnames


    def _get_valid_anns(self):
        ann_fnames = glob.glob(os.path.join(self._train_config['valid_annot_folder'], '*.xml'))
        return ann_fnames


    def _get_test_anns(self):
        ann_fnames = glob.glob(os.path.join(self._train_config['test_annot_folder'], '*.xml'))
        return ann_fnames


    def get_save_image_number(self):
        return self._train_config['save_image_number']

    def split_train_val(self):
        return self._train_config['split_train_valid']