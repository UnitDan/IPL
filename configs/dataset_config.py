configs = {
    'yelp': lambda args: {'name': 'ProcessedDataset', 'path': 'data/Yelp_pro', 
        'device': args.device, 'split_ratio': [0.7, 0.1, 0.2], 'min_inter': 10, 'n_groups': 10, 'gamma': 1.552},
    'amazon': lambda args: {'name': 'ProcessedDataset', 'path': 'data/Amazon_pro',
        'device': args.device, 'split_ratio': [0.7, 0.1, 0.2], 'min_inter': 10, 'n_groups': 10, 'gamma': 1.446},
    'gowalla': lambda args: {'name': 'ProcessedDataset', 'path': 'data/Gowalla_pro',
        'device': args.device, 'split_ratio': [0.7, 0.1, 0.2], 'min_inter': 10, 'n_groups': 10, 'gamma': 1.285},
    'ml1m': lambda args: {'name': 'ProcessedDataset', 'path': 'data/ML1M_pro',
        'device': args.device, 'split_ratio': [0.7, 0.1, 0.2], 'min_inter': 10, 'n_groups': 10, 'gamma': 1.826},
}

def get_dataset_config(args):
    return configs[args.data](args)