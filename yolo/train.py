
# import comet_ml as comet
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
from tensorflow.python.keras.optimizer_v2.adam import Adam
import os
import h5py
from tqdm import tqdm

from yolo.loss import loss_fn
from .utils.utils import EarlyStopping


def train_fn(model, 
             train_generator, 
             valid_generator=None, 
             learning_rate=1e-4, 
             num_epoches=500, 
             save_dir=None, 
             weight_name='weights'):
    
    save_file = _setup(save_dir=save_dir, weight_name=weight_name)
    es = EarlyStopping(patience=10)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # optimizer = Adam(learning_rate=learning_rate) # adam_v2

    epoch = -1

    history = []
    for idx in range(epoch + 1, num_epoches):
        # 1. update params
        train_loss = _loop_train(model, optimizer, train_generator, idx)
        
        # 2. monitor validation loss
        if valid_generator:
            valid_loss = _loop_validation(model, valid_generator)
            loss_value = valid_loss
        else:
            loss_value = train_loss
        print("{}-th loss = {}, train_loss = {}".format(idx, loss_value, train_loss))

        # 3. update weights
        history.append(loss_value)
        if save_file is not None and loss_value == min(history):
            print("    update weight with loss: {}".format(loss_value))
            _save_weights(model, '{}.h5'.format(save_file))
            # model.save_weights('{}'.format(save_file), save_format='h5')
        
        if es.step(loss_value):
            print('early stopping')
            return history

    return history


def _loop_train(model, optimizer, generator, epoch):
    # one epoch
    
    n_steps = generator.steps_per_epoch
    loss_value = 0
    for _ in tqdm(range(n_steps)):
        xs, yolo_1, yolo_2, yolo_3 = generator.next_batch()
        ys = [yolo_1, yolo_2, yolo_3]
        grads, loss = _grad_fn(model, xs, ys)
        loss_value += loss
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    loss_value /= generator.steps_per_epoch

    return loss_value


def _grad_fn(model, images_tensor, list_y_trues):
    # print(images_tensor.shape)
    with tf.GradientTape() as tape:
        logits = model(images_tensor)
        loss = loss_fn(list_y_trues, logits)
        # print("loss = ", loss)
    return tape.gradient(loss, model.trainable_variables), loss


def _loop_validation(model, generator):
    # one epoch
    n_steps = generator.steps_per_epoch
    loss_value = 0
    for _ in range(n_steps):
        xs, yolo_1, yolo_2, yolo_3 = generator.next_batch()
        ys = [yolo_1, yolo_2, yolo_3]
        ys_ = model(xs)
        loss_value += loss_fn(ys, ys_)
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


if __name__ == '__main__':
    pass
