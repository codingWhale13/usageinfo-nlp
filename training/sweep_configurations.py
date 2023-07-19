sweep_configurations = {
    "adam-vs-amsgrad": {
        "method": "grid",
        "metric": {"goal": "minimize", "name": "best_val_loss"},
        "parameters": {
            "optimizer.name": {"value": "AdamW"},
            "accumulate_grad_batches": {"values": [8, 32]},
            "optimizer.lr": {"values": [1e-3, 1e-4, 1e-5]},
            "optimizer.weight_decay": {"values": [0.01, 0.1]},
            "optimizer.amsgrad": {"values": [True, False]},
        },
    },
    "sgdm-vs-nesterov": {
        "method": "grid",
        "metric": {"goal": "minimize", "name": "best_val_loss"},
        "parameters": {
            "accumulate_grad_batches": {"values": [8, 32]},
            "optimizer.lr": {"values": [1e-3, 1e-4, 1e-5]},
            "optimizer.weight_decay": {"values": [0.01, 0.1]},
            "optimizer.momentum": {"value": 0.9},
            "optimizer.nesterov": {"values": [True, False]},
        },
    },
}
