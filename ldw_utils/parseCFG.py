# read cfg
# data.yaml model.yaml default.yaml
import os, yaml


class NestedConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):  # 如果值是字典，则递归地创建一个 NestedConfig 实例
                setattr(self, key, NestedConfig(**value))
            else:  # 否则，直接设置属性
                setattr(self, key, value)


class Config:
    def __init__(self, config_dict):  # 递归地处理传入的字典，并设置属性
        self._set_config(config_dict, self)

    def _set_config(self, config_dict, obj):
        for key, value in config_dict.items():
            if isinstance(value, dict):  # 如果值是字典，则创建一个 NestedConfig 实例并设置为属性
                setattr(obj, key, NestedConfig(**value))
            else:  # 否则，直接设置属性
                setattr(obj, key, value)


def parse_cfg(data_cfg, model_cfg):
    root_dir = os.getcwd()+"/"
    data_cfg_path = root_dir+"ldw_cfg/dataset/"+data_cfg
    with open(data_cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    model_cfg_path = root_dir+"ldw_cfg/model/" + model_cfg
    with open(model_cfg_path, "r", encoding="utf-8") as f:
        model = yaml.safe_load(f)
    default_cfg_path = root_dir+"ldw_cfg/default.yaml"
    with open(default_cfg_path, "r", encoding="utf-8") as f:
        default = yaml.safe_load(f)
        pre_dict = default["Predict"]
        del default["Predict"]
        Train = default["Train"]
        del default["Train"]
        Eval = default["Eval"]
        del default["Eval"]
    config_dict = {"data":data,"model":model,"default":default,"predict":pre_dict,"train":Train,"eval":Eval}
    cfg = Config(config_dict)
    return cfg



