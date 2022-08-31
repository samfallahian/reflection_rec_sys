CFG = {
    "data": {
        "path": "/mnt/d/sources/reflection_rec_sys/data"
    },
    "train": {
        "learning_rate": 1e-3,
        "batch_size": 2048,
        "buffer_size": 1000,
        "epochs": 200,
        "challenge_prompt": "challenge",
        "solution_prompt": "solution"
    },
    "model": {
        "input": [128, 128, 3]
    }
}