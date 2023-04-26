import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
import sys
import time
import json

def get_dataset(config):
    config = config.copy()
    dataset = getattr(sys.modules['dataset'], config['name'])
    dataset = dataset(config)
    return dataset


def update_ui_sets(u, i, user_inter_sets, item_inter_sets):
    if u in user_inter_sets:
        user_inter_sets[u].add(i)
    else:
        user_inter_sets[u] = {i}
    if i in item_inter_sets:
        item_inter_sets[i].add(u)
    else:
        item_inter_sets[i] = {u}


def update_user_inter_lists(u, i, t, user_map, item_map, user_inter_lists):
    if u in user_map and i in item_map:
        duplicate = False
        for i_t in user_inter_lists[user_map[u]]:
            if i_t[0] == item_map[i]:
                i_t[1] = min(i_t[1], t)
                duplicate = True
                break
        if not duplicate:
            user_inter_lists[user_map[u]].append([item_map[i], t])


def output_data(file_path, data):
    with open(file_path, 'w') as f:
        for user in range(len(data)):
            u_items = [str(user)] + [str(item) for item in data[user]]
            f.write(' '.join(u_items) + '\n')


class BasicDataset(Dataset):
    def __init__(self, dataset_config):
        print(dataset_config)
        self.config = dataset_config
        self.name = dataset_config['name']
        self.min_interactions = dataset_config.get('min_inter')
        self.split_ratio = dataset_config.get('split_ratio')
        self.device = dataset_config['device']
        self.negative_sample_ratio = dataset_config.get('neg_ratio', 1)
        self.shuffle = dataset_config.get('shuffle', False)
        self.n_users = 0
        self.n_items = 0
        self.user_inter_lists = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.train_array = None
        self.val_array = None
        self.test_array = None
        self.loader_set = 'train'
        print('init dataset ' + dataset_config['name'])

    def remove_sparse_ui(self, user_inter_sets, item_inter_sets):
        not_stop = True
        while not_stop:
            not_stop = False
            users = list(user_inter_sets.keys())
            for user in users:
                if len(user_inter_sets[user]) < self.min_interactions:
                    not_stop = True
                    for item in user_inter_sets[user]:
                        item_inter_sets[item].remove(user)
                    user_inter_sets.pop(user)
            items = list(item_inter_sets.keys())
            for item in items:
                if len(item_inter_sets[item]) < self.min_interactions:
                    not_stop = True
                    for user in item_inter_sets[item]:
                        user_inter_sets[user].remove(item)
                    item_inter_sets.pop(item)
        user_map = dict()
        for idx, user in enumerate(user_inter_sets):
            user_map[user] = idx
        item_map = dict()
        for idx, item in enumerate(item_inter_sets):
            item_map[item] = idx
        self.n_users = len(user_map)
        self.n_items = len(item_map)
        return user_map, item_map
    
    def generate_data(self):
        self.train_data = [[] for _ in range(self.n_users)]
        self.val_data = [[] for _ in range(self.n_users)]
        self.test_data = [[] for _ in range(self.n_users)]

        self.train_data_item = [[] for _ in range(self.n_items)]
        self.val_data_item = [[] for _ in range(self.n_items)]
        self.test_data_item = [[] for _ in range(self.n_items)]
        train_u, val_u, test_u = set(), set(), set()
        self.train_array = []
        average_inters = []
        for item in range(self.n_items):
            self.item_inter_lists[item].sort(key=lambda entry: entry[1])
            if self.shuffle:
                np.random.shuffle(self.item_inter_lists[item])

            n_inter_users = len(self.item_inter_lists[item])
            average_inters.append(n_inter_users)
            n_train_users = int(n_inter_users * self.split_ratio[0])
            n_test_users = int(n_inter_users * self.split_ratio[2])
            self.train_data_item[item] += [u_t[0] for u_t in self.item_inter_lists[item][:n_train_users]]
            for u_t in self.item_inter_lists[item][:n_train_users]:
                train_u.add(u_t[0])
            self.val_data_item[item] += [u_t[0] for u_t in self.item_inter_lists[item][n_train_users:-n_test_users]]
            for u_t in self.item_inter_lists[item][n_train_users:-n_test_users]:
                val_u.add(u_t[0])
            self.test_data_item[item] += [u_t[0] for u_t in self.item_inter_lists[item][-n_test_users:]]
            for u_t in self.item_inter_lists[item][-n_test_users:]:
                test_u.add(u_t[0])
        all_u = train_u.intersection(val_u).intersection(test_u)
        print(f'(before filter) train users: {len(train_u)}, val users: {len(val_u)}, test users: {len(test_u)}')
        print(f'(after filter) train/val/test users: {len(all_u)}')

        def data_item_to_data(data_item, data, all_u=all_u):
            for item, users in enumerate(data_item):
                for user in users:
                    # if user in all_u:
                    data[user].append(item)
        
        data_item_to_data(self.train_data_item, self.train_data)
        data_item_to_data(self.val_data_item, self.val_data)
        data_item_to_data(self.test_data_item, self.test_data)


        average_inters = np.mean(average_inters)
        print('Users {:d}, Items {:d}, Average number of interactions {:.3f}, Total interactions {:.1f}'
              .format(self.n_users, self.n_items, average_inters, average_inters * self.n_users))
    
    def __len__(self):
        data_array = self.__getattribute__(self.loader_set + '_array')
        return len(data_array)

    def __getitem__(self, index):
        data_with_negs = self.get_data_with_neg_from(self.loader_set)
        return data_with_negs

    def get_data_with_neg_from(self, loader_set):
        data = self.__getattribute__(loader_set + '_data')

        user = random.randint(0, self.n_users - 1)
        while not data[user]:
            user = random.randint(0, self.n_users - 1)
        pos_item = random.choice(data[user])
        data_with_negs = [[user, pos_item] for _ in range(self.negative_sample_ratio)]
        for idx in range(self.negative_sample_ratio):
            neg_item = random.randint(0, self.n_items - 1)
            while neg_item in data[user] or neg_item in self.train_data[user]:
                neg_item = random.randint(0, self.n_items - 1)
            data_with_negs[idx].append(neg_item)
        data_with_negs = np.array(data_with_negs, dtype=np.int64)
        return data_with_negs

    def output_dataset(self, path):
        if not os.path.exists(path): os.mkdir(path)
        output_data(os.path.join(path, 'train.txt'), self.train_data)
        output_data(os.path.join(path, 'val.txt'), self.val_data)
        output_data(os.path.join(path, 'test.txt'), self.test_data)


class ProcessedDataset(BasicDataset):
    def __init__(self, dataset_config):
        super(ProcessedDataset, self).__init__(dataset_config)
        self.train_data = self.read_data(os.path.join(dataset_config['path'], 'train.txt'))
        self.val_data = self.read_data(os.path.join(dataset_config['path'], 'val.txt'))
        self.test_data = self.read_data(os.path.join(dataset_config['path'], 'test.txt'))
        self.train_item_result, self.item_groups, self.q_list = self.get_item_result(os.path.join(dataset_config['path'], 'train.txt'), n_groups=dataset_config['n_groups'])
        self.val_item_result = self.get_item_result(os.path.join(dataset_config['path'], 'val.txt'))
        self.test_item_result = self.get_item_result(os.path.join(dataset_config['path'], 'test.txt'))
        assert len(self.train_data) == len(self.val_data)
        assert len(self.train_data) == len(self.test_data)
        self.n_users = len(self.train_data)

        self.train_array = []
        self.val_array = []
        self.test_array = []
        for user in range(self.n_users):
            self.train_array.extend([[user, item] for item in self.train_data[user]])
            self.val_array.extend([[user, item] for item in self.val_data[user]])
            self.test_array.extend([[user, item] for item in self.test_data[user]])

        self.gamma = dataset_config['gamma']

    def read_data(self, file_path):
        data = []
        with open(file_path, 'r') as f:
            lines = f.read().strip().split('\n')
        for line in lines:
            try:
                items = line.split(' ')[1:]
                items = [int(item) for item in items]
                if items:
                    self.n_items = max(self.n_items, max(items) + 1)
                data.append(items)
            except:
                data.append([])
        return data
    
    def get_item_result(self, file_path, n_groups=None, return_pop=False):
        item_result = [set() for _ in range(self.n_items)]
        item_pop = [0 for _ in range(self.n_items)]
        with open(file_path, 'r') as f:
            lines = f.read().strip().split('\n')
        for line in lines:
            items = line.split(' ')[1:]
            for item in items:
                item_result[int(item)].add(line.split(' ')[0])
                if n_groups:
                    item_pop[int(item)] += 1
        if n_groups:
            groups = [[] for _ in range(n_groups)]
            sorted_items = np.argsort(item_pop)
            group_size = int(self.n_items // n_groups)
            for g in range(n_groups):
                groups[g] = [sorted_items[g*group_size + i] for i in range(group_size)]
            return item_result, groups, item_pop
            
        if return_pop:
            return item_result, item_pop
        else:
            return item_result

class GowallaDataset(BasicDataset):
    def __init__(self, dataset_config):
        super(GowallaDataset, self).__init__(dataset_config)

        input_file_path = os.path.join(dataset_config['path'], 'Gowalla_totalCheckins.txt')
        user_inter_sets, item_inter_sets = dict(), dict()
        with open(input_file_path, 'r') as f:
            lines = f.read().strip().split('\n')
        for line in lines:
            u, _, _, _, i = line.strip().split('\t')
            u, i = int(u), int(i)
            update_ui_sets(u, i, user_inter_sets, item_inter_sets)
        user_map, item_map = self.remove_sparse_ui(user_inter_sets, item_inter_sets)

        self.user_inter_lists = [[] for _ in range(self.n_users)]
        self.item_inter_lists = [[] for _ in range(self.n_items)]
        for line in lines:
            u, t, _, _, i = line.split('\t')
            t = time.strptime(t, '%Y-%m-%dT%H:%M:%SZ')
            t = int(time.mktime(t))
            u, i = int(u), int(i)
            update_user_inter_lists(u, i, t, user_map, item_map, self.user_inter_lists)
            update_user_inter_lists(i, u, t, item_map, user_map, self.item_inter_lists)

        self.generate_data()

class YelpDataset(BasicDataset):
    def __init__(self, dataset_config):
        super(YelpDataset, self).__init__(dataset_config)

        input_file_path = os.path.join(dataset_config['path'], 'yelp_academic_dataset_review.json')
        user_inter_sets, item_inter_sets = dict(), dict()
        with open(input_file_path, 'r') as f:
            line = f.readline().strip()
            while line:
                record = json.loads(line)
                r = float(record['stars'])
                if r > 3.:
                    u = record['user_id']
                    i = record['business_id']
                    update_ui_sets(u, i, user_inter_sets, item_inter_sets)
                line = f.readline().strip()
        user_map, item_map = self.remove_sparse_ui(user_inter_sets, item_inter_sets)

        self.user_inter_lists = [[] for _ in range(self.n_users)]
        self.item_inter_lists = [[] for _ in range(self.n_items)]
        with open(input_file_path, 'r') as f:
            line = f.readline().strip()
            while line:
                record = json.loads(line)
                r = float(record['stars'])
                if r > 3.:
                    u = record['user_id']
                    i = record['business_id']
                    t = record['date']
                    t = time.strptime(t, '%Y-%m-%d %H:%M:%S')
                    t = int(time.mktime(t))
                    update_user_inter_lists(u, i, t, user_map, item_map, self.user_inter_lists)
                    update_user_inter_lists(i, u, t, item_map, user_map, self.item_inter_lists)
                line = f.readline().strip()

        self.generate_data()

class AmazonDataset(BasicDataset):
    def __init__(self, dataset_config):
        super(AmazonDataset, self).__init__(dataset_config)

        input_file_path = os.path.join(dataset_config['path'], 'ratings_Books.csv')
        user_inter_sets, item_inter_sets = dict(), dict()
        with open(input_file_path, 'r') as f:
            line = f.readline().strip()
            while line:
                u, i, r, _ = line.split(',')
                r = float(r)
                if r > 3.:
                    update_ui_sets(u, i, user_inter_sets, item_inter_sets)
                line = f.readline().strip()
        user_map, item_map = self.remove_sparse_ui(user_inter_sets, item_inter_sets)

        self.user_inter_lists = [[] for _ in range(self.n_users)]
        self.item_inter_lists = [[] for _ in range(self.n_items)]
        with open(input_file_path, 'r') as f:
            line = f.readline().strip()
            while line:
                u, i, r, t = line.split(',')
                r = float(r)
                if r > 3.:
                    t = int(t)
                    update_user_inter_lists(u, i, t, user_map, item_map, self.user_inter_lists)
                    update_user_inter_lists(i, u, t, item_map, user_map, self.item_inter_lists)
                line = f.readline().strip()

        self.generate_data()

class ML1MDataset(BasicDataset):
    def __init__(self, dataset_config):
        super(ML1MDataset, self).__init__(dataset_config)
        rating_file_path = os.path.join(dataset_config['path'], 'ratings.dat')
        user_inter_sets, item_inter_sets = dict(), dict()
        with open(rating_file_path, 'r') as f:
            lines = f.read().strip().split('\n')
        for line in lines:
            u, i, r, _ = line.split('::')
            u, i, r = int(u), int(i), int(r)
            if r < 4:
                continue
            if u in user_inter_sets:
                user_inter_sets[u].add(i)
            else:
                user_inter_sets[u] = {i}
            if i in item_inter_sets:
                item_inter_sets[i].add(u)
            else:
                item_inter_sets[i] = {u}
        user_map, item_map = self.remove_sparse_ui(user_inter_sets, item_inter_sets)

        self.user_inter_lists = [[] for _ in range(self.n_users)]
        self.item_inter_lists = [[] for _ in range(self.n_items)]
        for line in lines:
            u, i, r, t = line.split('::')
            u, i, r, t = int(u), int(i), int(r), int(t)
            if r > 3 and u in user_map and i in item_map:
                duplicate = False
                for i_t in self.user_inter_lists[user_map[u]]:
                    if i_t[0] == item_map[i]:
                        i_t[1] = min(i_t[1], t)
                        duplicate = True
                        break
                if not duplicate:
                    self.user_inter_lists[user_map[u]].append([item_map[i], t])

                duplicate = False
                for u_t in self.item_inter_lists[item_map[i]]:
                    if u_t[0] == user_map[u]:
                        u_t[1] = min(u_t[1], t)
                        duplicate = True
                        break
                if not duplicate:
                    self.item_inter_lists[item_map[i]].append([user_map[u], t])
        self.generate_data()

class ML10MDataset(BasicDataset):
    def __init__(self, dataset_config):
        super(ML10MDataset, self).__init__(dataset_config)
        rating_file_path = os.path.join(dataset_config['path'], 'ratings.dat')
        user_inter_sets, item_inter_sets = dict(), dict()
        with open(rating_file_path, 'r') as f:
            lines = f.read().strip().split('\n')
        for line in lines:
            u, i, r, _ = line.split('::')
            u, i, r = int(u), int(i), float(r)
            if r < 4:
                continue
            if u in user_inter_sets:
                user_inter_sets[u].add(i)
            else:
                user_inter_sets[u] = {i}
            if i in item_inter_sets:
                item_inter_sets[i].add(u)
            else:
                item_inter_sets[i] = {u}
        user_map, item_map = self.remove_sparse_ui(user_inter_sets, item_inter_sets)

        self.user_inter_lists = [[] for _ in range(self.n_users)]
        self.item_inter_lists = [[] for _ in range(self.n_items)]
        for line in lines:
            u, i, r, t = line.split('::')
            u, i, r, t = int(u), int(i), float(r), int(t)
            if r > 3 and u in user_map and i in item_map:
                duplicate = False
                for i_t in self.user_inter_lists[user_map[u]]:
                    if i_t[0] == item_map[i]:
                        i_t[1] = min(i_t[1], t)
                        duplicate = True
                        break
                if not duplicate:
                    self.user_inter_lists[user_map[u]].append([item_map[i], t])

                duplicate = False
                for u_t in self.item_inter_lists[item_map[i]]:
                    if u_t[0] == user_map[u]:
                        u_t[1] = min(u_t[1], t)
                        duplicate = True
                        break
                if not duplicate:
                    self.item_inter_lists[item_map[i]].append([user_map[u], t])
        self.generate_data()


class DatasetWithLoader():
    def __init__(self, dataset: BasicDataset, batch_size, dataloader_num_workers):
        self.dataset = dataset
        self.loader_set = dataset.loader_set
        self.batch_size = batch_size
        self.dataloader_num_workers = dataloader_num_workers
    def __enter__(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.dataloader_num_workers)
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.dataset.loader_set = self.loader_set

def get_dataloader(dataset, loader_set, batch_size, dataloader_num_workers):
    data_with_loader = DatasetWithLoader(dataset, batch_size, dataloader_num_workers)
    data_with_loader.dataset.loader_set = loader_set
    return data_with_loader