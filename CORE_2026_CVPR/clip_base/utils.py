import os
import json
import yaml
from omegaconf import DictConfig, OmegaConf





def get_dataset_class_names( long=False):
    with open("./imagenet100_classes.txt", "r") as f:
        lines = f.read().splitlines()
    return [line.split("\t")[-1] for line in lines]



