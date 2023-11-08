from argparse import ArgumentParser
import logging
import os
import re
import json
from io import BytesIO
from pathlib import Path
from typing import Union

import yaml
import msgpack
import pandas as pd




def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", type=Path, default="config/newbaseM.yml")
    args = parser.parse_args()
    return args

def main():
    

if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%d-%m-%Y %H:%M:%S",
        level=logging.INFO,
    )

    args = parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config = config["model_params"]

    main()
