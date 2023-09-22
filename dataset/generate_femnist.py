import numpy as np
import os
import random
import torchvision.transforms as transforms
from utils.dataset_utils import check, separate_data, split_data, save_file


random.seed(1)
np.random.seed(1)


# Allocate data to users
def generate_femnist(dir_path='dataset/femnist_reduced', num_clients=40, num_classes=10):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Setup directory for train/test data
    config_path = dir_path + "/config.json"
    train_path = dir_path + "/train/"
    test_path = dir_path + "/test/"

    # if check(config_path, train_path, test_path, num_clients, num_classes, niid, balance, partition):
    #     return

    train_data = [None]*num_clients
    test_data = [None]*num_clients
    statistic = [None]*num_clients

    for client_num in np.arange(40):
        # client train data
        images = np.loadtxt(os.path.join(
            dir_path, 'raw',
            'task_'+str(client_num),
            'train_images.csv'
        ))
        labels = np.loadtxt(os.path.join(
            dir_path, 'raw',
            'task_'+str(client_num),
            'train_labels.csv'
        ))
        train_data[client_num] = {
            'x':np.reshape(images, (-1, 1, 28, 28)),
            'y':labels
        }
        # client test data
        images = np.loadtxt(os.path.join(
            dir_path, 'raw',
            'task_'+str(client_num),
            'test_images.csv'
        ))
        labels = np.loadtxt(os.path.join(
            dir_path, 'raw',
            'task_'+str(client_num),
            'test_labels.csv'
        ))
        test_data[client_num] = {
            'x':np.reshape(images, (-1, 1, 28, 28)),
            'y':labels
        }
        # data statistics
        for i in np.unique(labels):
            # statistic[client_num] = (int(i), int(labels[client_num].count(i)))
            statistic[client_num] = (int(i), int(sum(labels==i)))

    # print(type(train_data)) # list
    # print(len(train_data))  # num_clients
    # print(type(train_data[0])) # 'dict'
    # print(train_data[0].keys())  # 'x', 'y'
    # print(type(train_data[0]['x'])) # numpy.ndarray
    # print(type(train_data[0]['y'])) # numpy.ndarray
    # print(train_data[0]['x'].shape) # (1411, 1, 28, 28)
    # print(train_data[0]['y'].shape) # (1411,)


    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes,
        statistic, niid=False, balance=True, partition=None)


if __name__ == "__main__":
    generate_femnist()