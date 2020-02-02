import tensorflow as tf
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=tf_config)
# print(tf.executing_eagerly())
import gc, os, h5py, time, cv2
import numpy as np
from tqdm import tqdm
from datetime import datetime, date

from yolo.loss import loss_fn, loss_component
from yolo.utils.box import draw_boxes
from .utils.utils import EarlyStopping, Logger
from .frontend import YoloDetector 
from .eval.fscore import count_true_positives, calc_score
# from yolo.optimizer import AdamWeightDecayOptimizer


def train_fn(model,
             train_generator, 
             valid_generator=None,
             learning_rate=1e-4, 
             num_epoches=500, 
             save_dir=None, 
             weight_name='weights',
             num_warmups=5,
             configs=None) -> 'train function':
    save_file = _setup(save_dir=save_dir, weight_name=weight_name)
    es = EarlyStopping(patience=10)
    history = []
    current_time = date.today().strftime('%d_%m_%Y-') + datetime.now().strftime('%H_%M_%S')

    logger = Logger('resnet50', current_time)
    print('---Logged Files')

    writer_1 = tf.contrib.summary.create_file_writer('logs-tensorboard/%s/valid_loss' % current_time, flush_millis=10000)
    writer_2 = tf.contrib.summary.create_file_writer('logs-tensorboard/%s/train_loss' % current_time, flush_millis=10000)

    global_step = tf.Variable(0, trainable=False)
    warm_up_step = 1
    for epoch in range(1, num_epoches + 1):
        warm_up = True if epoch <= num_warmups else False
        if not warm_up:
            # learning rate scheduler
            learning_rate_fn = tf.train.exponential_decay(learning_rate=learning_rate, 
                                                        global_step=global_step,
                                                        decay_steps=10,
                                                        decay_rate=0.8,
                                                        staircase=False)
            lr = learning_rate_fn()
            global_step.assign_add(1)
        else:
            print('Warm up...')
            lr = learning_rate

        # 1. update params
        print('Training...')
        train_loss, train_loss_box, train_loss_conf, train_loss_class, train_fscore = _loop_train(model, train_generator, num_warmups, warm_up_step, warm_up, learning_rate=lr, configs=configs, epoch=epoch)

        # 2. monitor validation loss
        if valid_generator and valid_generator.steps_per_epoch != 0:
            print('Validating...')
            valid_loss, valid_loss_box, valid_loss_conf, valid_loss_class, highest_loss_imgs, valid_fscore = _loop_validation(model, valid_generator, configs=configs, epoch=epoch)
            logger.write_img(highest_loss_imgs)
            save_images(configs, model, highest_loss_imgs, epoch)
            del highest_loss_imgs
        else:
            valid_fscore = train_fscore
            valid_loss = train_loss # if no validation loss, use training loss as validation loss instead
            valid_loss_box, valid_loss_conf, valid_loss_class = train_loss_box, train_loss_conf, train_loss_class

        tensorboard_logger(writer_1, writer_2, train_loss, train_loss_box, train_loss_conf, train_loss_class, valid_loss, valid_loss_box, valid_loss_conf, valid_loss_class, epoch)
        print('{}-th'.format(epoch))
        print('--> train_loss = {:.4f}, train_loss_box = {:.4f}, train_loss_conf = {:.4f}, train_loss_class = {:.4f}'.format(train_loss, train_loss_box, train_loss_conf, train_loss_class))
        print('--> train_fscore: {}'.format(train_fscore))
        print('--> valid_loss = {:.4f}, valid_loss_box = {:.4f}, valid_loss_conf = {:.4f}, valid_loss_class = {:.4f}'.format(valid_loss, valid_loss_box, valid_loss_conf, valid_loss_class))
        print('--> valid_fscore: {}'.format(valid_fscore))
        logger.write({ 
            'train_loss': train_loss,
            'train_box': train_loss_box,
            'train_conf': train_loss_conf,
            'train_class': train_loss_class,
            'valid_loss': valid_loss,
            'valid_box': valid_loss_box,
            'valid_conf': valid_loss_conf,
            'valid_class': valid_loss_class,
        }, display=False)
        
        # 3. save weights
        history.append(round(valid_loss, 4))
        if save_file is not None and round(valid_loss, 4) == min(history):
            print("    update weight with loss: {:4f}".format(round(valid_loss, 4)))
            _save_weights(model, '{}.h5'.format(save_file))
            # model.save_weights('{}'.format(save_file), save_format='h5')
        
        if es.step(round(valid_loss, 4)):
            print('early stopping')
            break

        tf.keras.backend.clear_session()
        tf.set_random_seed(1)
        gc.collect()
        del train_loss, train_loss_box, train_loss_conf, train_loss_class, valid_loss, valid_loss_box, valid_loss_conf, valid_loss_class


def _loop_train(model, generator, num_warmups, warm_up_step, warm_up=False, learning_rate=1e-4, configs=None, epoch=None) -> 'one epoch':
    n_steps = generator.steps_per_epoch
    total_steps = n_steps * num_warmups
    loss_value, loss_box_value, loss_conf_value, loss_class_value = 0.0, 0.0, 0.0, 0.0
    n_true_positives, n_truth, n_pred = 0, 0, 0

    for _ in tqdm(range(n_steps)):
        image_tensor, yolo_1, yolo_2, yolo_3, anno_files, img_files, boxes, labels  = generator.next_batch()
        y_true = [yolo_1, yolo_2, yolo_3]
        y_pred = model(image_tensor)
        grads, loss = _grad_fn(model, image_tensor, y_true)
        _, loss_box, loss_conf, loss_class, _ = loss_component(y_true, y_pred)

        loss_value += float(tf.cast(loss, tf.float32))
        loss_box_value += float(tf.cast(loss_box, tf.float32))
        loss_conf_value += float(tf.cast(loss_conf, tf.float32))
        loss_class_value += float(tf.cast(loss_class, tf.float32))

        if warm_up:
            warm_up_learning_rate = (warm_up_step / total_steps) * learning_rate
            warm_up_step += 1
            optimizer = tf.train.AdamOptimizer(learning_rate=warm_up_learning_rate)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            del warm_up_learning_rate
        else:
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if epoch % 5 == 0:
            calculate_fscore(configs, model, anno_files, img_files, boxes, labels, n_true_positives, n_truth, n_pred)

        # prevent memory leak
        tf.keras.backend.clear_session()
        tf.set_random_seed(1)
        gc.collect()
        del image_tensor, y_true, y_pred, yolo_1, yolo_2, yolo_3, grads, loss, loss_box, loss_conf, loss_class, optimizer, _

    loss_value /= generator.steps_per_epoch
    loss_box_value /= generator.steps_per_epoch 
    loss_conf_value /= generator.steps_per_epoch
    loss_class_value /= generator.steps_per_epoch
    fscore = calc_score(n_true_positives, n_truth, n_pred)

    return loss_value, loss_box_value, loss_conf_value, loss_class_value, fscore


def _grad_fn(model, images_tensor, list_y_true, list_y_pred=None) -> 'compute gradient & loss':
    with tf.GradientTape() as tape:
        list_y_pred = model(images_tensor)
        loss = loss_fn(list_y_true, list_y_pred)
        with tape.stop_recording():
            grads = tape.gradient(loss, model.trainable_variables)
    return grads, loss


def _loop_validation(model, generator, configs=None, epoch=None):
    # one epoch
    n_steps = generator.steps_per_epoch
    loss_value, loss_box_value, loss_conf_value, loss_class_value = 0.0, 0.0, 0.0, 0.0
    highest_loss_imgs = {}
    n_true_positives, n_truth, n_pred = 0, 0, 0

    for _ in tqdm(range(n_steps)):
        image_tensor, yolo_1, yolo_2, yolo_3, anno_files, img_files, boxes, labels = generator.next_batch()
        y_true = [yolo_1, yolo_2, yolo_3]
        y_pred = model(image_tensor)
        loss, loss_box, loss_conf, loss_class, loss_each_img = loss_component(y_true, y_pred)
        find_highest_loss_each_class(loss_each_img, img_files, labels, highest_loss_imgs, epoch=epoch)

        loss_value += float(tf.cast(loss, tf.float32))
        loss_box_value += float(tf.cast(loss_box, tf.float32))
        loss_conf_value += float(tf.cast(loss_conf, tf.float32))
        loss_class_value += float(tf.cast(loss_class, tf.float32))

        if epoch % 5 == 0:
            calculate_fscore(configs, model, anno_files, img_files, boxes, labels, n_true_positives, n_truth, n_pred)

        # prevent memory leak
        tf.keras.backend.clear_session()
        tf.set_random_seed(1)
        gc.collect()
        del y_true, y_pred, yolo_1, yolo_2, yolo_3, img_files, labels, loss, loss_box, loss_conf, loss_class

    loss_value /= generator.steps_per_epoch
    loss_box_value /= generator.steps_per_epoch
    loss_conf_value /= generator.steps_per_epoch
    loss_class_value /= generator.steps_per_epoch
    fscore = calc_score(n_true_positives, n_truth, n_pred)
    
    return loss_value, loss_box_value, loss_conf_value, loss_class_value, highest_loss_imgs, fscore


def _setup(save_dir, weight_name='weights'):
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_name = os.path.join(save_dir, weight_name)
    else:
        file_name = None
    return file_name


def _save_weights(model, filename):
    f = h5py.File(filename, 'w')
    weights = model.get_weights()
    for i in range(len(weights)):
        f.create_dataset('weight' + str(i), data=weights[i])
    f.close()


def tensorboard_logger(writer_1, writer_2, train_loss, train_loss_box, train_loss_conf, train_loss_class, valid_loss, valid_loss_box, valid_loss_conf, valid_loss_class, idx):
    with writer_1.as_default(), tf.contrib.summary.always_record_summaries():
        tf.contrib.summary.scalar('loss', valid_loss, step=idx)
        tf.contrib.summary.scalar('loss_box', valid_loss_box, step=idx)
        tf.contrib.summary.scalar('loss_conf', valid_loss_conf, step=idx)
        tf.contrib.summary.scalar('loss_class', valid_loss_class, step=idx)
    tf.contrib.summary.flush()

    with writer_2.as_default(), tf.contrib.summary.always_record_summaries():
        tf.contrib.summary.scalar('loss', train_loss, step=idx)
        tf.contrib.summary.scalar('loss_box', train_loss_box, step=idx)
        tf.contrib.summary.scalar('loss_conf', train_loss_conf, step=idx)
        tf.contrib.summary.scalar('loss_class', train_loss_class, step=idx)
    tf.contrib.summary.flush()


def key_sort(value):
    return value[0]

def find_highest_loss_each_class(loss_each_img, img_files, list_labels, class_dict, epoch=None):
    number_of_images = 10
    losses = list(loss_each_img.numpy())
    for loss, img_name, labels in zip(losses, img_files, list_labels):
        for label in labels:
            if label not in class_dict:
                class_dict[label] = []
            # class_dict[label].append(( loss, img_name, labels ))
            class_dict[label].append(( loss, img_name ))

    for label, img_list in class_dict.items():
        img_list.sort(key=key_sort, reverse=True)
        img_list = img_list[:number_of_images]
        class_dict[label] = img_list


def calculate_fscore(configs, model, anno_files, img_files, boxes, labels, n_true_positives, n_truth, n_pred):
    detector = configs.create_detector(model)
    for anno_file, img_file, true_boxes, true_labels in zip(anno_files, img_files, boxes, labels):
        true_labels = np.array(true_labels)
        image = cv2.imread(img_file)[:,:,::-1]

        pred_boxes, pred_labels, pred_probs = detector.detect(image, cls_threshold=0.5)

        n_true_positives += count_true_positives(pred_boxes, true_boxes, pred_labels, true_labels)
        n_truth += len(true_boxes)
        n_pred += len(pred_boxes)


def save_images(configs, model, data, epoch) -> 'dictionary':
    class_labels = configs._model_config['labels']
    detector = configs.create_detector(model)
    folder_name = os.path.join('save_imgs', 'epoch_' + str(epoch))
    for label, values in data.items():
        save_folder = os.path.join(folder_name, str(class_labels[label]))
        print(save_folder)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for _, image_file in values:
            image = cv2.imread(image_file)[:,:,::-1]
            pred_boxes, pred_labels, pred_probs = detector.detect(image, cls_threshold=0.5)

            image_ = draw_boxes(image, pred_boxes, pred_labels, pred_probs, class_labels, desired_size=416)
            output_path = os.path.join(save_folder, os.path.split(image_file)[-1])
            cv2.imwrite(output_path, image_[:,:,::-1])

if __name__ == '__main__':
    pass
