cnfgs = [
    {"device": 'cuda',
        "num_task": 50,
        "num_class": 10,
        "seed": 42,
        "batch_size": 128,
        "lr": 0.001,
        "epoch": 200,
        "lambda": 1,
        "train_mode": 'online_laplace_diagonal',
        "n_samples_fisher": 200,
        },
    
    {"device": 'cuda',
        "num_task": 50,
        "num_class": 10,
        "seed": 42,
        "batch_size": 128,
        "lr": 0.001,
        "epoch": 200,
        "lambda": 1,
        "train_mode": 'ewc_diagonal',
        "n_samples_fisher": 200,  
        },
    
    {"device": 'cuda',
        "num_task": 50,
        "num_class": 10,
        "seed": 42,
        "batch_size": 128,
        "lr": 0.001,
        "epoch": 200,
        "lambda": 1,
        "c": 0.9,
        "epsilon": 0.1,
        "omega_max": None,
        "train_mode": 'si_diagonal',        
        },

    {"device": 'cuda',
        "num_task": 50,
        "num_class": 10,
        "seed": 42,
        "batch_size": 128,
        "lr": 0.001,
        "epoch": 200,
        "lambda": 1,
        "train_mode": 'cumulative',        
        },
    
    {"device": 'cuda',
        "num_task": 50,
        "num_class": 10,
        "seed": 42,
        "batch_size": 128,
        "lr": 0.001,
        "epoch": 200,
        "lambda": 1,
        "train_mode": 'normal',
        },
     
    {"device": 'cuda',
        "num_task": 50,
        "num_class": 10,
        "seed": 42,
        "batch_size": 128,
        "lr": 0.001,
        "epoch": 200,
        "lambda": 1,
        "train_mode": 'ewc_kfac',
        "n_samples_fisher": 200,  
        },
]