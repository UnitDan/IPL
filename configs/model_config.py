configs = {
    'mf': lambda args: {
        'name': 'MF', 'embedding_size': 64, 'device': args.device
    },
    'lightgcn': lambda args: {
        'name': 'LightGCN', 'embedding_size': 64, 'device': args.device, 'n_layers': 3
    },
    'popularity': lambda args: {
        'name': 'Popularity', 'device': args.device, 'dataset': args.dataset
    }
}

def get_model_config(args):
    return configs[args.model](args)