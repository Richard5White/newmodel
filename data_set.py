
import argparse
import os
import random
import json
import torch

from torch.utils.data import Dataset, DataLoader
import numpy as np

SEED = 2021
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


class TestDate(Dataset):
    def __init__(self, user_count, item_count, samples=None):
        self.user_count = user_count
        self.item_count = item_count
        self.samples = samples

    def __getitem__(self, idx):
        return int(self.samples[idx])

    def __len__(self):
        return len(self.samples)


class BehaviorDate(Dataset):
    def __init__(self, user_count, item_count, behavior_dict=None, behaviors=None):
        self.user_count = user_count
        self.item_count = item_count
        self.behavior_dict = behavior_dict
        self.behaviors = behaviors

    def __getitem__(self, idx):
        # generate positive and negative samples pairs under each behavior
        total = []
        all_inter = self.behavior_dict['all'].get(str(idx + 1), None)
        for behavior in self.behaviors:

            items = self.behavior_dict[behavior].get(str(idx + 1), None)
            if items is None:
                signal = [0, 0, 0]
            else:
                pos = random.sample(items, 1)[0]
                neg = random.randint(1, self.item_count)
                while np.isin(neg, all_inter):
                    neg = random.randint(1, self.item_count)
                signal = [idx + 1, pos, neg]
            total.append(signal)

        return np.array(total)

    def __len__(self):
        return self.user_count


class NegativeSampleData(Dataset):

    def __init__(self, user_count, item_count, behavior_dict=None, behaviors=None):
        self.user_count = user_count
        self.item_count = item_count
        self.behavior_dict = behavior_dict
        self.behaviors = behaviors
        self.item_list = [x + 1 for x in range(self.item_count)]

    def __getitem__(self, idx):

        total = []

        for behavior in self.behaviors:

            items = self.behavior_dict[behavior].get(str(idx + 1), None)
            if items is None:
                signal = [0] * 12
            else:
                pos = random.sample(items, 1)[0]
                neg = np.random.choice(np.setdiff1d(self.item_list, [pos]), 10).tolist()
                signal = [idx + 1, pos]
                signal.extend(neg)
            total.append(signal)

        all_inter = self.behavior_dict['all'].get(str(idx + 1), None)
        if all_inter is None:
            total.append([0] * 12)
        else:
            pos = random.sample(all_inter, 1)[0]
            neg = np.random.choice(np.setdiff1d(self.item_list, [pos]), 10).tolist()
            signal = [idx + 1, pos]
            signal.extend(neg)
            total.append(signal)

        return np.array(total)

    def __len__(self):
        return self.user_count

class DataSet(object):

    def __init__(self, args):

        self.behaviors = args.behaviors
        self.path = args.data_path
        self.loss_type = args.loss_type

        self.__get_count()
        self.__get_behavior_items()
        self.__get_validation_dict()
        self.__get_test_dict()
        self.__get_sparse_interact_dict()

        self.validation_gt_length = np.array([len(x) for _, x in self.validation_interacts.items()])
        self.test_gt_length = np.array([len(x) for _, x in self.test_interacts.items()])

    def __get_count(self):
        with open(os.path.join(self.path, 'count.txt'), encoding='utf-8') as f:
            count = json.load(f)
            self.user_count = count['user']
            self.item_count = count['item']

    def __get_behavior_items(self):
        """
        load the list of items corresponding to the user under each behavior
        :return:
        """
        self.train_behavior_dict = {}
        for behavior in self.behaviors:
            with open(os.path.join(self.path, behavior + '_dict.txt'), encoding='utf-8') as f:
                b_dict = json.load(f)
                self.train_behavior_dict[behavior] = b_dict
        with open(os.path.join(self.path, 'all_dict.txt'), encoding='utf-8') as f:
            b_dict = json.load(f)
            self.train_behavior_dict['all'] = b_dict

    def __get_test_dict(self):
        """
        load the list of items that the user has interacted with in the test set
        :return:
        """
        with open(os.path.join(self.path, 'test_dict.txt'), encoding='utf-8') as f:
            b_dict = json.load(f)
            self.test_interacts = b_dict

    def __get_validation_dict(self):
        """
        load the list of items that the user has interacted with in the validation set
        :return:
        """
        with open(os.path.join(self.path, 'validation_dict.txt'), encoding='utf-8') as f:
            b_dict = json.load(f)
            self.validation_interacts = b_dict

    def __get_sparse_interact_dict(self):
        """
        load graphs

        :return:
        """
        self.edge_index = {}
        self.item_behaviour_degree = []
        self.user_behaviour_degree = []
        all_row = []
        all_col = []
        for behavior in self.behaviors:
            with open(os.path.join(self.path, behavior + '.txt'), encoding='utf-8') as f:
                data = f.readlines()
                row = []
                col = []
                for line in data:
                    line = line.strip('\n').strip().split()
                    row.append(int(line[0]))
                    col.append(int(line[1]))
                indices = np.vstack((row, col))
                indices = torch.LongTensor(indices)

                values = torch.ones(len(row), dtype=torch.float32)
                user_inter = torch.sparse.FloatTensor(indices, values, [self.user_count + 1, self.item_count + 1]).to_dense()
                self.item_behaviour_degree.append(user_inter.sum(dim=0))
                self.user_behaviour_degree.append(user_inter.sum(dim=1))
                col = [x + self.user_count + 1 for x in col]
                row, col = [row, col], [col, row]
                row = torch.LongTensor(row).view(-1)
                all_row.append(row)
                col = torch.LongTensor(col).view(-1)
                all_col.append(col)
                edge_index = torch.stack([row, col])
                self.edge_index[behavior] = edge_index
        self.item_behaviour_degree = torch.stack(self.item_behaviour_degree, dim=0).T
        self.user_behaviour_degree = torch.stack(self.user_behaviour_degree, dim=0).T
        all_row = torch.cat(all_row, dim=-1)
        all_col = torch.cat(all_col, dim=-1)
        all_row = all_row.tolist()
        all_col = all_col.tolist()
        self.all_edge_index = list(set(zip(all_row, all_col)))
        self.all_edge_index = torch.LongTensor(self.all_edge_index).T


    def behavior_dataset(self):
        if self.loss_type == 'bpr':
            return BehaviorDate(self.user_count, self.item_count, self.train_behavior_dict, self.behaviors)
        else:
            return NegativeSampleData(self.user_count, self.item_count, self.train_behavior_dict, self.behaviors)

    def validate_dataset(self):
        return TestDate(self.user_count, self.item_count, samples=list(self.validation_interacts.keys()))

    def test_dataset(self):
        return TestDate(self.user_count, self.item_count, samples=list(self.test_interacts.keys()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Set args', add_help=False)
    parser.add_argument('--behaviors', type=list, default=['cart', 'click', 'collect', 'buy'], help='')
    parser.add_argument('--data_path', type=str, default='./data/Tmall', help='')
    parser.add_argument('--loss_type', type=str, default='bpr', help='')
    args = parser.parse_args()
    dataset = DataSet(args)
    loader = DataLoader(dataset=dataset.behavior_dataset(), batch_size=5)
    for index, item in enumerate(loader):
        print(index, '-----', item)
