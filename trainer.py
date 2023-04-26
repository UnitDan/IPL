import torch
import sys
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam, SGD
import time
import numpy as np
import os
from utils import AverageMeter, gen_name_string, groupby_apply
import torch.nn.functional as F
from dataset import get_dataloader
import random
from math import sqrt
from scipy import stats
from sklearn import metrics as mr
from scipy.stats import pearsonr

def get_trainer(config, dataset, dataset_name, model):
    config = config.copy()
    config['dataset'] = dataset
    config['dataset_name'] = dataset_name
    config['model'] = model
    trainer = getattr(sys.modules['trainer'], config['name'])
    trainer = trainer(config)
    return trainer


class BasicTrainer:
    def __init__(self, trainer_config):
        print(trainer_config)
        self.config = trainer_config
        self.name = trainer_config['name']
        self.dataset = trainer_config['dataset']
        self.model = trainer_config['model']
        self.topks = trainer_config['topks']
        self.device = trainer_config['device']
        self.n_epochs = trainer_config['n_epochs']
        self.max_patience = trainer_config.get('max_patience', 50)
        self.val_interval = trainer_config.get('val_interval', 1)
        self.epoch = 0
        self.best_ndcg = -np.inf
        self.save_path = None
        self.opt = None

        data_user = TensorDataset(torch.arange(self.dataset.n_users, dtype=torch.int64, device=self.device))
        self.data_user_loader = DataLoader(data_user, batch_size=trainer_config['test_batch_size'])

        tground_truth_pairs = []
        for user, rec_item_set in enumerate(self.dataset.test_data):
            rec_item_set = [int(i) for i in rec_item_set]
            tground_truth_pairs.extend(list(zip([user]*len(rec_item_set), rec_item_set)))
        tground_truth_pairs = torch.tensor(tground_truth_pairs)
        self.tgt_users, self.tgt_items = tground_truth_pairs[:, 0], tground_truth_pairs[:, 1]

    def initialize_optimizer(self):
        opt = getattr(sys.modules[__name__], self.config['optimizer'])
        self.opt = opt(self.model.parameters(), lr=self.config['lr'])

    def train_one_epoch(self):
        raise NotImplementedError

    def record(self, writer, stage, metrics):
        for metric in metrics:
            for k in self.topks:
                writer.add_scalar('{:s}_{:s}/{:s}_{:s}@{:d}'
                                  .format(self.model.name, self.name, stage, metric, k)
                                  , metrics[metric][k], self.epoch)

    def train(self, verbose=True, writer=None):
        if not self.model.trainable:
            results, metrics = self.eval('val')
            if verbose:
                print('Validation result. {:s}'.format(results))
            ndcg = metrics['NDCG'][self.topks[0]]
            return ndcg

        if not os.path.exists('checkpoints'): os.mkdir('checkpoints')
        if not os.path.exists(f'checkpoints/{self.config["save_flag"]}'): os.mkdir(f'checkpoints/{self.config["save_flag"]}')
        patience = self.max_patience
        for self.epoch in range(self.n_epochs):
            start_time = time.time()
            self.model.train()
            loss = self.train_one_epoch()
            _, metrics = self.eval('train')
            consumed_time = time.time() - start_time
            if verbose:
                print('Epoch {:d}/{:d}, Loss: {:.6f}, Time: {:.3f}s'
                      .format(self.epoch, self.n_epochs, loss, consumed_time))
            if writer:
                writer.add_scalar('{:s}_{:s}/train_loss'.format(self.model.name, self.name), loss, self.epoch)
                self.record(writer, 'train', metrics)

            if (self.epoch + 1) % self.val_interval != 0:
                continue

            start_time = time.time()
            results, metrics = self.eval('val')
            consumed_time = time.time() - start_time
            if verbose:
                print('Validation result. {:s}Time: {:.3f}s'.format(results, consumed_time))
            if writer:
                self.record(writer, 'validation', metrics)

            ndcg = metrics['NDCG'][self.topks[0]]
            if ndcg > self.best_ndcg:
                if self.save_path:
                    os.remove(self.save_path)
                if self.name == 'BPRTrainer':
                    self.save_path = os.path.join(f'checkpoints/{self.config["save_flag"]}', '{:s}_{:s}_{:s}.pth'
                        .format(self.model.name, self.name, self.config['dataset_name']))
                else:
                    self.save_path = os.path.join(f'checkpoints/{self.config["save_flag"]}', '{:s}_{:s}_{:s}_{:f}_{:f}_{:f}.pth'
                        .format(self.model.name, self.name, self.config['dataset_name'], self.config['lr'], self.config['l2_reg'], self.config['lambda_f']))
                self.best_epoch = self.epoch
                self.best_ndcg = ndcg
                self.model.save(self.save_path)
                patience = self.max_patience
                print('Best NDCG, save model to {:s}'.format(self.save_path))
            else:
                patience -= self.val_interval
                if patience <= 0:
                    print('Early stopping!')
                    print('Best Epoch: {}'.format(self.best_epoch))
                    break
        self.model.load(self.save_path)
        return self.best_ndcg

    def calculate_metrics(self, eval_data, rec_items, ks=None):
        results = {'Precision': {}, 'Recall': {}, 'NDCG': {}}
        hit_matrix = np.zeros_like(rec_items, dtype=np.float32)
        for user in range(rec_items.shape[0]):
            for item_idx in range(rec_items.shape[1]):
                if rec_items[user, item_idx] in eval_data[user]:
                    hit_matrix[user, item_idx] = 1.
        eval_data_len = np.array([len(items) for items in eval_data], dtype=np.int32)

        if ks == None:
            ks = self.topks
        for k in ks:
            hit_num = np.sum(hit_matrix[:, :k], axis=1)
            precisions = hit_num / k
            with np.errstate(invalid='ignore'):
                recalls = hit_num / eval_data_len

            max_hit_num = np.minimum(eval_data_len, k)
            max_hit_matrix = np.zeros_like(hit_matrix[:, :k], dtype=np.float32)
            for user, num in enumerate(max_hit_num):
                max_hit_matrix[user, :num] = 1.
            denominator = np.log2(np.arange(2, k + 2, dtype=np.float32))[None, :]
            dcgs = np.sum(hit_matrix[:, :k] / denominator, axis=1)
            idcgs = np.sum(max_hit_matrix / denominator, axis=1)
            with np.errstate(invalid='ignore'):
                ndcgs = dcgs / idcgs

            user_masks = (max_hit_num > 0)
            results['Precision'][k] = precisions[user_masks].mean()
            results['Recall'][k] = recalls[user_masks].mean()
            results['NDCG'][k] = ndcgs[user_masks].mean()

            gamma = self.dataset.gamma
            C_list, Q_list, _ = self.get_CQR('test', binarize=True, k=k)
            r = C_list / (Q_list**(2-gamma))

            results['DI'][k] = torch.std(r) / torch.mean(r)
            results['MI'][k] = mr.mutual_info_score(r, Q_list)

        return results
    
    def get_propensity_weight(self):
        return 1 / torch.Tensor(self.dataset.q_list)

    def calculate_unbiased_metrics(self, eval_data, rec_items, ks=None):
        propensity =  1 / torch.Tensor(self.dataset.q_list)

        results = {'Precision': {}, 'Recall': {}, 'NDCG': {}}
        hit_matrix = np.zeros_like(rec_items, dtype=np.float32)
        for user in range(rec_items.shape[0]):
            for item_idx in range(rec_items.shape[1]):
                if rec_items[user, item_idx] in eval_data[user]:
                    hit_matrix[user, item_idx] = 1.0*propensity[item_idx]
        eval_data_len = np.array([len(items) for items in eval_data], dtype=np.int32)

        if ks == None:
            ks = self.topks
        for k in ks:
            hit_num = np.sum(hit_matrix[:, :k], axis=1)
            precisions = hit_num / k
            with np.errstate(invalid='ignore'):
                recalls = hit_num / eval_data_len

            max_hit_num = np.minimum(eval_data_len, k)
            max_hit_matrix = np.zeros_like(hit_matrix[:, :k], dtype=np.float32)
            for user, num in enumerate(max_hit_num):
                max_hit_matrix[user, :num] = 1.
            denominator = np.log2(np.arange(2, k + 2, dtype=np.float32))[None, :]
            dcgs = np.sum(hit_matrix[:, :k] / denominator, axis=1)
            idcgs = np.sum(max_hit_matrix / denominator, axis=1)
            with np.errstate(invalid='ignore'):
                ndcgs = dcgs / idcgs

            user_masks = (max_hit_num > 0)

            sn = propensity[self.tgt_items]
            snw = (1 / groupby_apply(self.tgt_users, sn, bins=self.dataset.n_users, reduction='sum')).numpy()

            results['Precision'][k] = (precisions[user_masks]*snw[user_masks]).mean()
            results['Recall'][k] = (recalls[user_masks]*snw[user_masks]).mean()
            results['NDCG'][k] = (ndcgs[user_masks]*snw[user_masks]).mean()

            gamma = self.dataset.gamma
            C_list, Q_list, _ = self.get_CQR('test', binarize=True, k=k)
            r = C_list / (Q_list**(2-gamma))

            results['DI'][k] = torch.std(r) / torch.mean(r)
            results['MI'][k] = mr.mutual_info_score(r, Q_list)
        return results

    def eval(self, val_or_test, bias=True, banned_items=None, ks=None):
        if ks == None:
            ks = self.topks

        self.model.eval()
        eval_data = getattr(self.dataset, val_or_test + '_data')
        rec_items = []
        with torch.no_grad():
            for users in self.data_user_loader:
                users = users[0]
                scores = self.model.predict(users)

                if val_or_test != 'train':
                    users = users.cpu().numpy().tolist()
                    exclude_user_indexes = []
                    exclude_items = []
                    for user_idx, user in enumerate(users):
                        items = self.dataset.train_data[user]
                        if val_or_test == 'test':
                            items = items + self.dataset.val_data[user]
                        exclude_user_indexes.extend([user_idx] * len(items))
                        exclude_items.extend(items)
                    scores[exclude_user_indexes, exclude_items] = -np.inf
                if banned_items is not None:
                    scores[:, banned_items] = -np.inf

                _, items = torch.topk(scores, k=max(ks))
                rec_items.append(items.cpu().numpy())


        rec_items = np.concatenate(rec_items, axis=0)
        if bias:
            metrics = self.calculate_metrics(eval_data, rec_items, ks=ks)
        else:
            metrics = self.calculate_unbiased_metrics(eval_data, rec_items, ks=ks)

        precison = ''
        recall = ''
        ndcg = ''
        for k in ks:
            precison += '{:.3f}%@{:d}, '.format(metrics['Precision'][k] * 100., k)
            recall += '{:.3f}%@{:d}, '.format(metrics['Recall'][k] * 100., k)
            ndcg += '{:.3f}%@{:d}, '.format(metrics['NDCG'][k] * 100., k)
        results = 'Precision: {:s}Recall: {:s}NDCG: {:s}'.format(precison, recall, ndcg)
        return results, metrics

    def get_CQR(self, val_or_test, banned_items=None, binarize=True, save_result=False, k=None, ban_user_rate=None, ban_item_rate=None, disturbing_rate=None):
        self.model.eval()
        rec_map = torch.zeros(self.dataset.n_users, self.dataset.n_items)

        eval_data = getattr(self.dataset, val_or_test + '_item_result')
        
        if save_result:
            result_path = 'output/rec_result/{}'.format(self.config['save_flag'])
            if not os.path.exists(result_path): os.mkdir(result_path)
            # name_str = '{}/{}_{}_{}_lambdaf={}_rec.pt'.format(
            #     result_path, self.config['dataset_name'], self.config['model_name'], self.config['name'], self.config['lambda_f']
            # )
            name_str = gen_name_string(
                result_path,
                self.config['dataset_name'], self.config['model_name'], self.config['name'],
                lr=self.config['lr'], wd=self.config['l2_reg'], lambda_f=self.config['lambda_f'])
            result_saver = open(name_str, 'wb')
            score_agg = []

        for item, rec_user_set in enumerate(eval_data):
            rec_user_set = [int(u) for u in rec_user_set]
            rec_map[rec_user_set, item] = 1

        Q_list = torch.zeros(self.dataset.n_items).to(device='cpu', dtype=torch.float)
        Q_list += rec_map.sum(dim=0)

        if disturbing_rate != None:
            l = int(rec_map.numel()*disturbing_rate)
            mask = torch.tensor([True]*l+[False]*(rec_map.numel()-l)).reshape_as(rec_map)
            rec_map[mask] = 1 - rec_map[mask]

        C_list = torch.zeros(self.dataset.n_items).to(device='cpu', dtype=torch.float)
        R_list = torch.zeros(self.dataset.n_items).to(device='cpu', dtype=torch.float)
        for users in self.data_user_loader:
            users = users[0]

            if ban_user_rate != None:
                l = int(len(users)*ban_user_rate)
                mask = torch.tensor([False]*l+[True]*(len(users)-l))
                random.shuffle(mask)
                users = users[mask]

            scores = self.model.predict(users).to(device='cpu', dtype=torch.float)

            users = users.cpu().numpy().tolist()
            if val_or_test != 'train':
                exclude_user_indexes = []
                exclude_items = []
                for user_idx, user in enumerate(users):
                    items = self.dataset.train_data[user]
                    if val_or_test == 'test':
                        items = items + self.dataset.val_data[user]
                    exclude_user_indexes.extend([user_idx] * len(items))
                    exclude_items.extend(items)
                scores[exclude_user_indexes, exclude_items] = -np.inf
            if banned_items is not None:
                scores[:, banned_items] = -np.inf

            if not binarize:
                scores = torch.sigmoid(scores)

            if save_result:
                score_agg.append([users, scores])

            if k == None:
                k = int(self.config['eval_fairk'])

            top, idx = scores.topk(k, dim=1)
            score_mask = torch.zeros_like(scores)
            score_mask.scatter_(1, idx, torch.ones_like(top))

            
            R_list += score_mask.sum(dim=0)
            
            recs = rec_map[users]
            if binarize:
                hit = score_mask.mul(recs)
                C_list += hit.nansum(dim=0)
            else:
                hit_scores = scores.mul(recs)
                C_list += hit_scores.nansum(dim=0)

        

        if ban_item_rate != None:
            l = int(len(C_list)*ban_item_rate)
            mask = torch.tensor([False]*l+[True]*(len(C_list)-l))
            random.shuffle(mask)
            C_list = C_list[mask]
            Q_list = Q_list[mask]
            R_list = R_list[mask]


        if save_result:
            torch.save(score_agg, result_saver)
            print('rec scores are saved')
            result_saver.close()

        return C_list, Q_list, R_list

class BPRTrainer(BasicTrainer):
    def __init__(self, trainer_config):
        super(BPRTrainer, self).__init__(trainer_config)

        self.dataloader = DataLoader(self.dataset, batch_size=trainer_config['batch_size'],
                                     num_workers=trainer_config['dataloader_num_workers'])
        self.initialize_optimizer()
        self.l2_reg = trainer_config['l2_reg']

    def train_one_epoch(self):
        losses = AverageMeter()
        for batch_data in self.dataloader:
            inputs = batch_data[:, 0, :].to(device=self.device, dtype=torch.int64)
            users, pos_items, neg_items = inputs[:, 0],  inputs[:, 1],  inputs[:, 2]

            users_r, pos_items_r, neg_items_r, l2_norm_sq = self.model.bpr_forward(users, pos_items, neg_items)
            pos_scores = torch.sum(users_r * pos_items_r, dim=1)
            neg_scores = torch.sum(users_r * neg_items_r, dim=1)

            bpr_loss = F.softplus(neg_scores - pos_scores).mean()
            reg_loss = self.l2_reg * l2_norm_sq.mean()
            loss = bpr_loss + reg_loss
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            losses.update(loss.item(), inputs.shape[0])
        return losses.avg

class IPLBPRTrainer(BPRTrainer):
    def __init__(self, trainer_config):
        super(IPLBPRTrainer, self).__init__(trainer_config)

        self.loader_batch_size = trainer_config['batch_size']
        self.loader_num_workers = trainer_config['dataloader_num_workers']

        self.initialize_optimizer()
        self.l2_reg = trainer_config['l2_reg']
        self.rec_map = torch.zeros(self.dataset.n_users, self.dataset.n_items)
        ground_truth_pairs = []
        for item, rec_user_set in enumerate(self.dataset.train_item_result):
            rec_user_set = [int(u) for u in rec_user_set]
            self.rec_map[rec_user_set, item] = 1
            ground_truth_pairs.extend(list(zip(rec_user_set, [item]*len(rec_user_set))))
        print('n_gt', len(ground_truth_pairs))
        print('n_items', self.dataset.n_items)
        ground_truth_pairs = torch.tensor(ground_truth_pairs, device=self.device)
        self.gt_users, self.gt_items = ground_truth_pairs[:, 0], ground_truth_pairs[:, 1]

        gt_dataset = TensorDataset(ground_truth_pairs)
        self.gt_loader = DataLoader(gt_dataset, batch_size=trainer_config['batch_size'])

        self.best_eval_loss = np.inf

    def sample_items(self, num=10):
        items = []
        for g in self.dataset.item_groups:
            items.extend(random.choices(g, k=num))
        return items  

    def train_one_epoch(self):
        ipl_loss = self.ipl_loss()
        bpr_loss = self.bpr_loss()
        loss = ipl_loss + bpr_loss
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        print(f'BPR loss: {bpr_loss}, IPL loss: {ipl_loss}')
        return bpr_loss.item() + ipl_loss.item()
    
    def ipl_loss(self):
        all_ground_truth = self.rec_map
        q_list = all_ground_truth.sum(dim=0).to(device=self.device, dtype=torch.float)

        scores = torch.sigmoid(self.model.predict_interactions(self.gt_users, self.gt_items))
        c_list = groupby_apply(self.gt_items, scores, bins=self.dataset.n_items, reduction='sum').to(device=self.device, dtype=torch.float)

        with np.errstate(invalid='ignore'):
            r_list = c_list/(q_list**(2-self.config['gamma']))
        
        ipl_loss = self.config['lambda_f']*torch.sqrt(torch.var(r_list))

        return ipl_loss

    def bpr_loss(self):
        performance_loss = torch.tensor(0.).to(device=self.device, dtype=torch.float)
        count = 0

        with get_dataloader(self.dataset, 'train', self.loader_batch_size, self.loader_num_workers) as dataloader:
            loader = dataloader

            for batch_data in loader:
                inputs = batch_data[:, 0, :].to(device=self.device, dtype=torch.int64)
                users, pos_items, neg_items = inputs[:, 0],  inputs[:, 1],  inputs[:, 2]

                users_r, pos_items_r, neg_items_r, l2_norm_sq = self.model.bpr_forward(users, pos_items, neg_items)
                pos_scores = torch.sum(users_r * pos_items_r, dim=1)
                neg_scores = torch.sum(users_r * neg_items_r, dim=1)

                bpr_loss = F.softplus(neg_scores - pos_scores).mean()

                reg_loss = self.l2_reg * l2_norm_sq.mean()
                performance_loss = performance_loss + bpr_loss + reg_loss
                count += 1

        performance_loss = performance_loss / count

        return performance_loss