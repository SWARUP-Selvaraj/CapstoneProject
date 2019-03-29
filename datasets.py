import tensorflow as tf
import tensorflow_hub as hub
import os
import pandas as pd
from tqdm import tqdm
from datetime import datetime as dt
def printf(string):
    print(str(dt.now()) + ' > ' + string)
def load_data(directory, label=""):
    """
    Reads data from text files in a folder and returns a dataframe
    """
    data = {}
    data["text"] = []
    for file in tqdm(os.listdir(directory), desc='Loading ' + label + ' : '):
        with tf.gfile.GFile(os.path.join(directory, file), "r") as f:
            data["text"].append(f.read())
    return pd.DataFrame.from_dict(data)
def load_dataset(directory):
    """
    Reads the dataset and returns train and test dataframes
    """
    trp = load_data(os.path.join(directory,"train","pos"),"+ve Training")
    trn = load_data(os.path.join(directory,"train","neg"),"-ve Training")
    tep = load_data(os.path.join(directory,"test","pos"),"+ve Testing ")
    ten = load_data(os.path.join(directory,"test","neg"),"-ve Testing ")    
    trp['label'] = 1
    trn['label'] = 0
    tep['label'] = 1
    ten['label'] = 0
    train = pd.concat([trp, trn]).sample(frac=1).reset_index(drop=True)
    test = pd.concat([tep, ten]).sample(frac=1).reset_index(drop=True)
    return train, test
def download_dataset(force=False):
    dataset_path = os.path.join(os.path.expanduser('~'),'.keras','datasets','aclImdb')
    download_stat = os.path.isdir(dataset_path)
    if download_stat and (not force):
        printf('Dataset already available!')
    else:
        printf('Downloading dataset..')
        dataset = tf.keras.utils.get_file(fname="aclImdb.tar.gz", origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz", extract=True)
    printf('Extracting Data..')
    return load_dataset(dataset_path)