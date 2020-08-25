import argparse
from models.engine import load_engine
from utils.utils import read_config


if __name__ == "__main__":

    config = read_config("config.yml")
    engine = load_engine(config)
    engine.train()