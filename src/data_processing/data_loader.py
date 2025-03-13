from torch.utils.data import DataLoader
from src.data_processing.dataset import SignData
import glob 
from src.utils.utils import load_config
from sklearn.model_selection import train_test_split
import re

config = load_config("./configs/config.yaml")

actions = config["labels"]["actions"]

path = config["paths"]["data_dir"] 

ans = []
for action in actions:
    seq = []
    for sequence in range(10):
        seq.append(glob.glob(f"{path}/{action}/{sequence}/**.npy" ))
        
    ans.append(seq)

new_ans = []
for a in ans:
    for i in range(10):
        arr = sorted(a[i], key = lambda x: int(re.search(r'\\(\d+)\.npy$',x).group(1)))
    
        new_ans.append(arr)

train_videos, test_videos  = train_test_split(
    new_ans, test_size=0.2, random_state=42
)

label_map = {label: num for num, label in enumerate(actions)}
label_map

train_dataset = SignData(path= train_videos,label_map=label_map, mode = "train", transform= config["aug_param"])
valid_dataset = SignData(path= test_videos,label_map=label_map, mode = "valid")


def data_loader():
    train_data_loader = DataLoader(train_dataset, shuffle= True, batch_size = 4, num_workers = 0)
    valid_data_loader = DataLoader(valid_dataset, shuffle= False, batch_size = 2, num_workers = 0)

    return train_data_loader, valid_data_loader
