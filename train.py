import argparse
from yolo.train import train_fn
from yolo.config import ConfigParser

argparser = argparse.ArgumentParser(
    description='train yolo-v3 network')

argparser.add_argument(
    '-c',
    '--config',
    default="configs/test.json",
    help='config file')


if __name__ == '__main__':
    args = argparser.parse_args()
    # config = './configs/svhn.json'
    config = args.config
    config_parser = ConfigParser(config)

    split_train_valid = config_parser.split_train_val()
    # 1. create generator
    train_generator, valid_generator = config_parser.create_generator(split_train_valid=split_train_valid)

    # 2. create model
    model = config_parser.create_model()

    # 3. training
    learning_rate, save_dir, weight_name, n_epoches, checkpoint_path = config_parser.get_train_params()
    train_fn(model,
            train_generator,
            valid_generator,
            learning_rate=learning_rate,
            save_dir=save_dir,
            weight_name=weight_name,
            num_epoches=n_epoches)

