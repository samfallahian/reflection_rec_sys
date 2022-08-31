from dataloader.datareader import DataReader
from configs import config as cfg
from utils.helpers import Config
from executor.executor import Executor


def run():
    conf = Config.from_json(cfg.CFG)
    data_reader = DataReader(conf)
    data = data_reader.load_and_standardize_data()
    executor = Executor(cfg=conf.train, data=data)

    print(executor.get_top_k(challenge="time management", name="Bob"))


if __name__ == '__main__':
    run()

