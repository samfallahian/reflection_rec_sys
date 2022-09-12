from dataloader.datareader import DataReader
from configs import config as cfg
from utils.helpers import Config
from executor.executor import Executor
from datetime import datetime


def run():
    conf = Config.from_json(cfg.CFG)
    data_reader = DataReader(conf)
    print(f"Start reading csv files: {datetime.now()}")
    data = data_reader.load_and_standardize_data()
    print(f"Load the model: {datetime.now()}")
    executor = Executor(cfg=conf.train, data=data)
    print(f"Start training: {datetime.now()}")

    print(executor.get_top_k(challenge="time management", name="Bob"))
    print(f"Finished: {datetime.now()}")


if __name__ == '__main__':
    run()

