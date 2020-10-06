import shutil
import os
from pathlib import Path
import subprocess

# dataset name
dataset = "ml-1m"
assert dataset in ["ml-1m", "pinterest-20"]

# paths
main_path = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(main_path, exist_ok=True)

train_rating = main_path + "{}.train.rating".format(dataset)
test_rating = main_path + "{}.test.rating".format(dataset)
test_negative = main_path + "{}.test.negative".format(dataset)

model_path = "./models/"
BPR_model_path = model_path + "NeuMF.pth"


def download():
    """Download data files from GitHub"""
    root = "https://raw.githubusercontent.com/hexiangnan/neural_collaborative_filtering/master/Data/"
    files = ["ml-1m.test.negative", "ml-1m.test.rating", "ml-1m.train.rating"]

    for fname in files:
        if not os.path.exists(os.path.join(main_path, fname)):
            url = root + fname
            subprocess.run("curl -OL {}".format(url), shell=True, check=True)
            shutil.move(fname, main_path)

download()