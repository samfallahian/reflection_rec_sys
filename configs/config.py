CFG = {
    "data": {
        "path": "/mnt/d/sources/data/DL-PTV/merged/"
    },
    "train": {
        "learning_rate": 1e-3,
        "batch_size": 2048,
        "buffer_size": 1000,
        "epochs": 200
    },
    "model": {
        "input": [128, 128, 3],
        "up_stack": {
            "layer_1": 512,
            "layer_2": 256,
            "layer_3": 128,
            "layer_4": 64,
            "kernels": 3
        },
        "output": 3
    }
}