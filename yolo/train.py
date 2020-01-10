import tensorflow as tf
tf.enable_eager_execution()
# print(tf.executing_eagerly())
import os
import h5py
from tqdm import tqdm
from datetime import datetime, date
import time

from yolo.loss import loss_fn
from .utils.utils import EarlyStopping, Logger
from yolo.optimizer import AdamWeightDecayOptimizer

def train_fn(model,
             train_generator, 
             valid_generator=None, 
             learning_rate=1e-4, 
             num_epoches=500, 
             save_dir=None, 
             weight_name='weights') -> 'train function':
    
    save_file = _setup(save_dir=save_dir, weight_name=weight_name)
    es = EarlyStopping(patience=10)
    history = []
    current_time = date.today().strftime('%d-%m-%Y_') + datetime.now().strftime('%H:%M:%S')

    logger = Logger('resnet50', current_time)
    print('---Logged Files')

    writer_1 = tf.contrib.summary.create_file_writer('logs-tensorboard/%s/valid_loss' % current_time, flush_millis=10000)
    writer_2 = tf.contrib.summary.create_file_writer('logs-tensorboard/%s/train_loss' % current_time, flush_millis=10000)

    global_step = tf.Variable(0, trainable=False)
    boundaries = [5, 15, 30]
    values = [5e-5, 3e-5, 1e-5, 5e-6]
    for epoch in range(1, num_epoches + 1):
        # learning rate scheduler
        learning_rate_fn = tf.train.piecewise_constant(global_step, boundaries, values)
        optimizer = AdamWeightDecayOptimizer(learning_rate=learning_rate_fn(),
                                             weight_decay_rate=0.0001)
        # optimizer = tf.train.AdamOptimizer( learning_rate=learning_rate_fn() )
        global_step.assign_add(1)

        # 1. update params
        train_loss = _loop_train(model, optimizer, train_generator, epoch)

        # 2. monitor validation loss
        if valid_generator:
            print('Validating...')
            val_loss = _loop_validation(model, valid_generator)
            valid_loss = val_loss
        else:
            valid_loss = train_loss

        tensorboard_logger(writer_1, writer_2, train_loss, valid_loss, epoch)
        print("{}-th loss = {}, train_loss = {}".format(epoch, valid_loss, train_loss))
        logger.write({ 'valid_loss': valid_loss, 'train_loss': train_loss })

        # 3. update weights
        history.append(valid_loss)
        if save_file is not None and valid_loss == min(history):
            print("    update weight with loss: {}".format(valid_loss))
            _save_weights(model, '{}.h5'.format(save_file))
            # model.save_weights('{}'.format(save_file), save_format='h5')
        
        if es.step(valid_loss):
            print('early stopping')
            break

    return history


def _loop_train(model, optimizer, generator, epoch):
    # one epoch
    n_steps = generator.steps_per_epoch
    loss_value = 0
    for _ in tqdm(range(n_steps)):
        x, yolo_1, yolo_2, yolo_3 = generator.next_batch()
        y_true = [yolo_1, yolo_2, yolo_3]
        grads, loss = _grad_fn(model, x, y_true)
        loss_value += loss
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    loss_value /= generator.steps_per_epoch
    return loss_value


def _grad_fn(model, images_tensor, list_y_true) -> 'compute gradient & loss':
    with tf.GradientTape() as tape:
        logits = model(images_tensor)
        loss = loss_fn(list_y_true, logits)
    grads = tape.gradient(loss, model.trainable_variables)
    return grads , loss


def _loop_validation(model, generator):
    # one epoch
    n_steps = generator.steps_per_epoch
    loss_value = 0
    for _ in tqdm(range(n_steps)):
        x, yolo_1, yolo_2, yolo_3 = generator.next_batch()
        y_true = [yolo_1, yolo_2, yolo_3]
        y_pred = model(x)
        loss_value += loss_fn(y_true, y_pred)
    loss_value /= generator.steps_per_epoch
    return loss_value


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


def tensorboard_logger(writer_1, writer_2, train_loss, valid_loss, idx):

    with writer_1.as_default(), tf.contrib.summary.always_record_summaries():
        tf.contrib.summary.scalar('loss', valid_loss, step=idx)
    tf.contrib.summary.flush()

    with writer_2.as_default(), tf.contrib.summary.always_record_summaries():
        tf.contrib.summary.scalar('loss', train_loss, step=idx)
    tf.contrib.summary.flush()


if __name__ == '__main__':
    pass
