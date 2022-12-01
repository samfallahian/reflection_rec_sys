import os
import glob
import pandas as pd

class DataReader:

    def __init__(self, cfg):
        self.cfg = cfg

    def load_and_standardize_data(self):
        path = self.cfg.data.path
        csv_files = glob.glob(os.path.join(f"{path}/data", "*.csv"))
        data = []
        for csv in csv_files:
            frame = pd.read_csv(csv, encoding="cp1252", engine='python')  # "ISO-8859-1" , 'utf-8'
            data.append(frame)
        df = pd.concat(data, ignore_index=True)
        df.columns = ["name","semester","reflection","challenge","solution"]

        # df = pd.read_csv(f"{path}/data1.csv", encoding="cp1252", engine='python')
        df.dropna(inplace=True)
        return df
