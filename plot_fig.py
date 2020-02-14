import matplotlib.pyplot as plt
import numpy as np
import csv

def readcsv(filepath):
    result = {}
    with open(filepath, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for key in next(iter(reader)).keys():
            result[key] = []
        
        for row in reader:
            for key, value in row.items():
                result[key].append(float(value))
    return result

def plot_figure(data):
    # train = [0.142836692997410,0.326573811438973,0.421301522267281,0.503599746783832,0.589345088161209,0.659247126268683,0.699467887637668]
    # valid = [0.1282022597605996,0.3857637212296796,0.4234792208521228,0.5191102756892231,0.5482034863038066,0.6639102544614356,0.6958355650776648,]

    # train = [score * 100 for score in train]
    # valid = [score * 100 for score in valid]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    # fig.tight_layout()
    fig.subplots_adjust(left=0.1, bottom=0.125, right=0.9, top=0.9, wspace=0.3, hspace=0.2)

    ax1.title.set_text('loss')
    ax1.plot(data['train_loss'], '-b', label='train loss')
    ax1.plot(data['valid_loss'], 'r', label='valid loss')
    ax1.legend(loc=0)
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    # ax1.set_ylim(0, 80)

    ax1.set_xticks(range(0, len(data['train_loss'])+1, 5))
    ax1.set_xticklabels([i for i in range(1,50) if i % 5 == 0 or i == 1])
    ax1.set_yticks(range(0,70,10))


    train_fscore = [i*100 for i in data['train_fscore'] if i != 0.] 
    valid_fscore = [i*100 for i in data['valid_fscore'] if i != 0.]
    ax2.title.set_text('f-score')
    ax2.plot(train_fscore, '-b', label='train f-score')
    ax2.plot(valid_fscore, '-r', label='valid f-score')
    ax2.legend(loc=0)
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('f-score')
    ax2.set_ylim(0, 60)

    # ax2.set_xticks(range())
    ax2.set_xticklabels([i for i in range(len(data['train_fscore']) +1 ) if i % 5 == 0])
    # ax2.set_yticks(range(0, 60, 1))

    # plt.savefig('resnet50_fig')
    plt.show()

if __name__ == '__main__':
    # ['train_loss', 'train_box', 'train_conf', 'train_class', 
    #  'valid_loss', 'valid_box', 'valid_conf', 'valid_class', 
    #  'train_fscore', 'valid_fscore']
    data = readcsv('resnet50/02_02_2020-21_22_35/log.csv')
    plot_figure(data)