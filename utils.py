import argparse
import yaml

def read_cfg(args):
    with open(args.cfg) as yaml_file:
        cfg_dict = yaml.load(yaml_file, Loader=yaml.loader.SafeLoader)
    return cfg_dict


def args_parser():
    parser = argparse.ArgumentParser(description="Simple example of training script using timm.")
    parser.add_argument("--cfg", help="Config file to training.", default="./cfg/cfg.yaml")
    args = parser.parse_args()
    return args
