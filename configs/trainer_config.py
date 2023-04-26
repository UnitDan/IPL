configs = {
    'fairbpr': lambda args: {
        'name': 'FairBPRTrainer', 'optimizer': 'Adam',
        'n_epochs': 2000, 'batch_size': 2048, 'dataloader_num_workers': 6,
        'test_batch_size': 512, 'topks': [20], 'eval_fairk': 20,
        'lr': args.lr, 'l2_reg': args.wd, 'device': args.device,
        'lambda_f': args.lambda_f, 'model_name': args.model,
        'save_flag': args.save_flag, 'dataset_name': args.data, 'gamma': args.gamma
    },
    
    'bpr': lambda args: {
        'name': 'BPRTrainer', 'optimizer': 'Adam',
        'n_epochs': 2000, 'batch_size': 2048, 'dataloader_num_workers': 6,
        'test_batch_size': 512, 'topks': [20], 'eval_fairk': 20,
        'lr': args.lr, 'l2_reg': args.wd, 'device': args.device, 
        'lambda_f': -1, 'model_name': args.model,
        'save_flag': args.save_flag, 'dataset_name': args.data
    },

    'basic': lambda args: {
        'name': 'BasicTrainer', 'optimizer': 'Adam',
        'n_epochs': 2000, 'batch_size': 2048, 'dataloader_num_workers': 6,
        'test_batch_size': 512, 'topks': [20], 'eval_fairk': 20,
        'lr': args.lr, 'l2_reg': args.wd, 'device': args.device, 
        'lambda_f': -1, 'model_name': args.model,
        'save_flag': args.save_flag, 'dataset_name': args.dataset
    },
}

def get_trainer_config(args):
    return configs[args.trainer](args)