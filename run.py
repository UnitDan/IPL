from dataset import get_dataset
from model import get_model
from trainer import get_trainer
import torch
from utils import init_run
from tensorboardX import SummaryWriter
from configs.dataset_config import get_dataset_config
from configs.model_config import get_model_config
from configs.trainer_config import get_trainer_config
import argparse
from datetime import datetime

parser = argparse.ArgumentParser()

parser.add_argument('--lambda_f', type=float, help='the value of lambda_f', default=0)
parser.add_argument('--data', type=str, choices=['yelp', 'amazon', 'gowalla', 'ml1m'], help='name of dataset', default='ml1m_pro')
parser.add_argument('--model', type=str, choices=['mf', 'lightgcn'], help='name of model')
parser.add_argument('--save_flag', type=str, help='save flag')
parser.add_argument('--lr', type=float, help='learning rate')
parser.add_argument('--wd', type=float, help='weight decay')
parser.add_argument('--trainer', type=str, choices=['ipl', 'bpr'], help='trainer name')
parser.add_argument('--cuda', type=int)

args = parser.parse_args()
if args.trainer == 'bpr':
    args.lambda_f = -1
else:
    assert args.lambda_f > 0

def load_model(model, checkpoint_dir, trainer_name, dataset_name, lr=-1, wd=-1, lambda_f=-1):
    trainer_name_dict = {'bpr': 'BPRTrainer', 'ipl': 'IPLBPRTrainer'}
    trainer_name = trainer_name_dict[trainer_name]
    if trainer_name == 'BPRTrainer':
        checkpoint = '{}/{:s}_{:s}_{:s}.pth'.format(checkpoint_dir, model.name, trainer_name, dataset_name)
    elif trainer_name == 'IPLBPRTrainer':
        checkpoint = '{}/{:s}_{:s}_{:s}_{:f}_{:f}_{:f}.pth'.format(checkpoint_dir, model.name, trainer_name, dataset_name, lr, wd, lambda_f)
    
    print('load model')
    
    print('\nloading model from {}...\n'.format(checkpoint))
    model.load(checkpoint)

def main():
    log_name = f'{args.model}_{args.data}_{args.trainer}_lr({args.lr})_wd({args.wd})_lambda({args.lambda_f}).{datetime.today().strftime(format="%m-%d")}'
    log_path = 'logs/'+__file__[:-3]+f'/{args.save_flag}'
    init_run(log_path, log_name, 2021)

    torch.cuda.set_device(args.cuda)
    args.device = torch.device('cuda')

    dataset_config = get_dataset_config(args)
    model_config = get_model_config(args)
    trainer_config = get_trainer_config(args)

    dataset_config['path'] = dataset_config['path'] + '/' + str(1)

    writer = SummaryWriter(log_path+'/events')
    dataset = get_dataset(dataset_config)
    print('num of users', dataset.n_users, 'num of items', dataset.n_items,
        '\nnum of edges', len(dataset.train_array)+len(dataset.val_array)+len(dataset.test_array),
        'num of training edges', len(dataset))
    model = get_model(model_config, dataset)

    if args.trainer == 'ipl':
        try:
            load_model(model, f'checkpoints/{trainer_config["save_flag"]}', 'bpr', args.data)
        except:
            print('Can not load the pretrained model for IPL.')

    trainer = get_trainer(trainer_config, dataset, args.data, model)

    if args.trainer == 'ipl':
        print('-------------------------------- before training ------------------------------')
        results, _ = trainer.eval('test')
        print('Test result. {:s}'.format(results))
        gm, im= trainer.eval_fairness('test')
        print(f'group_metrics: {gm}, individual_metrics: {im}')
        print('-------------------------------------------------------------------------------')


    trainer.train(verbose=True, writer=writer)
    writer.close()

    results, _ = trainer.eval('val')
    print('Eval result. {:s}'.format(results))

    results, _ = trainer.eval('test')
    print('Test result. {:s}'.format(results))


if __name__ == '__main__':
    main()
