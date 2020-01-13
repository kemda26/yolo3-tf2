import tensorflow as tf
tf.enable_eager_execution()
# print(tf.executing_eagerly())
import os
import h5py
from tqdm import tqdm
from datetime import datetime, date
import time

from yolo.loss import loss_fn, loss_component
from .utils.utils import EarlyStopping, Logger
from yolo.optimizer import AdamWeightDecayOptimizer


def train_fn(model,
             train_generator, 
             valid_generator=None, 
             learning_rate=1e-4, 
             num_epoches=500, 
             save_dir=None, 
             weight_name='weights',
             num_warmups=5) -> 'train function':
    
    save_file = _setup(save_dir=save_dir, weight_name=weight_name)
    es = EarlyStopping(patience=15)
    history = []
    current_time = date.today().strftime('%d-%m-%Y_') + datetime.now().strftime('%H:%M:%S')

    logger = Logger('resnet50', current_time)
    print('---Logged Files')

    writer_1 = tf.contrib.summary.create_file_writer('logs-tensorboard/%s/valid_loss' % current_time, flush_millis=10000)
    writer_2 = tf.contrib.summary.create_file_writer('logs-tensorboard/%s/train_loss' % current_time, flush_millis=10000)

    global_step = tf.Variable(0, trainable=False)
    learning_rate = 1e-4
    optimizer = None
    warm_up_step = 1
    train_loss_box, train_loss_conf, train_loss_class = 0, 0 ,0
    for epoch in range(1, num_epoches + 1):
        warm_up = True if epoch <= num_warmups else False
        if not warm_up:
            # learning rate scheduler
            learning_rate_fn = tf.train.exponential_decay(learning_rate=learning_rate, 
                                                        global_step=global_step,
                                                        decay_steps=10,
                                                        decay_rate=0.8,
                                                        staircase=False)
            # optimizer = AdamWeightDecayOptimizer(learning_rate=learning_rate_fn(),
            #                                     weight_decay_rate=0.01)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_fn())
            global_step.assign_add(1)
        else:
            print('Warm up...')

        # 1. update params
        print('Training...')
        train_loss, train_loss_box, train_loss_conf, train_loss_class = _loop_train(model, optimizer, train_generator, epoch, learning_rate, warm_up, warm_up_step)
        # train_loss = _loop_train(model, optimizer, train_generator, epoch, learning_rate, warm_up, warm_up_step)

        # 2. monitor validation loss
        if valid_generator and valid_generator.steps_per_epoch != 0:
            print('Validating...')
            valid_loss, valid_loss_box, valid_loss_conf, valid_loss_class, highest_loss_dict = _loop_validation(model, valid_generator)
            logger.write_img(highest_loss_dict)
            del highest_loss_dict
        else:
            valid_loss = train_loss # if no validation loss, use training loss as validation loss instead
            valid_loss_box, valid_loss_conf, valid_loss_class = train_loss_box, train_loss_conf, train_loss_class

        tensorboard_logger(writer_1, writer_2, train_loss, train_loss_box, train_loss_conf, train_loss_class, valid_loss, valid_loss_box, valid_loss_conf, valid_loss_class, epoch)
        print('{}-th'.format(epoch))
        print('--> train_loss = {:.4f}, train_loss_box = {:.4f}, train_loss_conf = {:.4f}, train_loss_class = {:.4f}'.format(train_loss, train_loss_box, train_loss_conf, train_loss_class))
        print('--> valid_loss = {:.4f}, valid_loss_box = {:.4f}, valid_loss_conf = {:.4f}, valid_loss_class = {:.4f}'.format(valid_loss, valid_loss_box, valid_loss_conf, valid_loss_class))
        logger.write({ 
            'train_loss': train_loss.numpy(),
            'train_box': train_loss_box.numpy(),
            'train_conf': train_loss_conf.numpy(),
            'train_class': train_loss_class.numpy(),
            'valid_loss': valid_loss.numpy(),
            'valid_box': valid_loss_box.numpy(),
            'valid_conf': valid_loss_conf.numpy(),
            'valid_class': valid_loss_class.numpy(),
        }, display=False)
        
        # 3. update weights
        history.append(round(valid_loss.numpy(), 4))
        if save_file is not None and round(valid_loss.numpy(), 4) == min(history):
            print("    update weight with loss: {:4f}".format(round(valid_loss.numpy(), 4)))
            _save_weights(model, '{}.h5'.format(save_file))
            # model.save_weights('{}'.format(save_file), save_format='h5')
        
        if es.step(valid_loss):
            print('early stopping')
            break

        del train_loss, train_loss_box, train_loss_conf, train_loss_class, valid_loss, valid_loss_box, valid_loss_conf, valid_loss_class

    return history


def _loop_train(model, optimizer, generator, epoch, learning_rate, warm_up, warm_up_step):
    # one epoch
    n_steps = generator.steps_per_epoch
    loss_value, loss_box_value, loss_conf_value, loss_class_value = 0, 0, 0, 0
    for _ in tqdm(range(n_steps)):
        image_tensor, yolo_1, yolo_2, yolo_3, _, _ = generator.next_batch()
        # image_tensor, yolo_1, yolo_2, yolo_3 = generator.next_batch()
        y_true = [yolo_1, yolo_2, yolo_3]
        # y_pred = model(image_tensor)
        grads, loss, y_pred = _grad_fn(model, image_tensor, y_true)
        _, loss_box, loss_conf, loss_class, _ = loss_component(y_true, y_pred)

        loss_value += loss
        loss_box_value += loss_box
        loss_conf_value += loss_conf
        loss_class_value += loss_class

        if warm_up:
            warm_up_learning_rate = (warm_up_step / (n_steps * epoch)) * learning_rate
            warm_up_step += 1
            # optimizer = AdamWeightDecayOptimizer(learning_rate=warm_up_learning_rate,
            #                                      weight_decay_rate=0.008)
            optimizer = tf.train.AdamOptimizer(learning_rate=warm_up_learning_rate)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        else:
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        del image_tensor, y_true, y_pred, yolo_1, yolo_2, yolo_3, grads, loss, loss_box, loss_conf, loss_class, optimizer, _

    loss_value /= generator.steps_per_epoch
    loss_box_value /= generator.steps_per_epoch 
    loss_conf_value /= generator.steps_per_epoch
    loss_class_value /= generator.steps_per_epoch

    return loss_value, loss_box_value, loss_conf_value, loss_class_value


def _grad_fn(model, images_tensor, list_y_true, list_y_pred=None) -> 'compute gradient & loss':
    with tf.GradientTape() as tape:
        list_y_pred = model(images_tensor)
        loss = loss_fn(list_y_true, list_y_pred)
    grads = tape.gradient(loss, model.trainable_variables)
    return grads, loss, list_y_pred


def _loop_validation(model, generator):
    # one epoch
    n_steps = generator.steps_per_epoch
    loss_value, loss_box_value, loss_conf_value, loss_class_value = 0, 0, 0, 0
    highest_loss_dict = { 0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [] }
    for _ in tqdm(range(n_steps)):
        image_tensor, yolo_1, yolo_2, yolo_3, img_names, labels = generator.next_batch()
        y_true = [yolo_1, yolo_2, yolo_3]
        y_pred = model(image_tensor)
        loss, loss_box, loss_conf, loss_class, loss_each_img = loss_component(y_true, y_pred)
        find_highest_loss_each_class(loss_each_img, img_names, labels, highest_loss_dict)

        loss_value += loss
        loss_box_value += loss_box
        loss_conf_value += loss_conf
        loss_class_value += loss_class

        del y_true, y_pred, yolo_1, yolo_2, yolo_3, img_names, labels, loss, loss_box, loss_conf, loss_class

    loss_value /= generator.steps_per_epoch
    loss_box_value /= generator.steps_per_epoch
    loss_conf_value /= generator.steps_per_epoch
    loss_class_value /= generator.steps_per_epoch
    
    return loss_value, loss_box_value, loss_conf_value, loss_class_value, highest_loss_dict


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

def find_highest_loss_each_class(loss_each_img, img_names, list_labels, class_dict):
    losses = list(loss_each_img.numpy())
    for loss, img_name, labels in zip(losses, img_names, list_labels):
        for label in labels:
            class_dict[label].append(( loss, img_name, labels ))
    for key, img_list in class_dict.items():
        img_list.sort(key=key_sort, reverse=True)
        img_list = img_list[:10]
        class_dict[key] = img_list


if __name__ == '__main__':
    pass
