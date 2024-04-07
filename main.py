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
    executor = Executor(cfg=conf, data=data)
    print(f"Start training: {datetime.now()}")

    executor.get_results(input_file_name=conf.data.course_input)
    print(f"Finished: {datetime.now()}")


if __name__ == '__main__':
    run()


