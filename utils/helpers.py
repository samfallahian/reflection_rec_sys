import json
from datetime import datetime
import os.path


class Config:
    def __init__(self, data, train, model):
        self.data = data
        self.train = train
        self.model = model

    @classmethod
    def from_json(cls, cfg):
        """Creates config from json"""
        params = json.loads(json.dumps(cfg), object_hook=HelperObject)
        return cls(params.data, params.train, params.model)


class HelperObject(object):
    """Helper class to convert json into Python object"""

    def __init__(self, dict_):
        self.__dict__.update(dict_)


class Log:
    @staticmethod
    def write_file(data, file_name, cfg):
        with open(os.path.join("logs", f"{datetime.now()}-{file_name}-{cfg.data.folder}.json"), "w") as f:
            json.dump(data, f, indent=4)

    def append_file(self, data, file_name):
        pass
