import torch 
from torch.utils.data import Dataset
import numpy as np 
from src.utils.utils import augmentation, take_all_landmarks_processing, load_config

config = load_config("./configs/config.yaml")

class SignData(Dataset):
    def __init__(self, path, label_map,frame_drop = 0.0 ,mode = "train", transform = None):
        super().__init__()
        self.path = path
        self.label_map = label_map
        self.mode = mode
        self.transform = transform
        self.dict = ["left_hand", "right_hand", "lips"] 
        self.frame_drop = frame_drop

    def __len__(self):
        return len(self.path)

    def apply_frame_drop(self, data, frame_drop):
        """
        Input:
        - data: numpy array of shape (T, P, 2), where T is the number of frames and P is the number of points.
        - frame_drop: float, percentage of frames to randomly drop.

        Output:
        - data: numpy array with frames dropped based on the given percentage.

        Description: Randomly drops frames from the data based on the specified frame drop percentage.
        """
        if frame_drop > 0.0:
            drop_mask = np.random.random(len(data)) >= frame_drop
            dropped_data = data[drop_mask]

            if len(dropped_data) >= 2:
                data = dropped_data
        return data

    def process_data(self, landmark_dict, aug_param= None, frame_drop = 0.0):
        if aug_param:
            landmark_dict["left_hand"] = augmentation(landmark_dict["left_hand"], aug_param)
            landmark_dict["right_hand"] = augmentation(landmark_dict["right_hand"], aug_param)
            landmark_dict["lips"] = augmentation(landmark_dict["lips"], aug_param)
        
            # Concatenate landmark data from all parts
        landmark = np.concatenate([landmark_dict[key] for key in self.dict], axis=1)
        landmark = self.apply_frame_drop(landmark, frame_drop)
        
        # Select only x, y coordinates
        landmark = landmark[:, :, :2]
        
        # Convert to tensor
        landmark = torch.tensor(landmark)
        
        # Normalize the data
        landmark = landmark - landmark[~torch.isnan(landmark)].mean(0, keepdims=True)
        landmark = landmark / landmark[~torch.isnan(landmark)].std(0, keepdims=True)
        
        # Handle NaN values
        landmark[torch.isnan(landmark)] = 0.0  # TxPx2
        landmark = torch.reshape(landmark, (landmark.shape[0], -1))

        # Permute to change shape
        landmark = torch.permute(landmark, (1, 0))  # 2P x T

        return landmark

    def __getitem__(self, idx):
        videos = self.path[idx]
        

        if self.mode == "train" or self.mode == "valid":
            landmarks = []
            for video in videos:
                landmark = np.load(video)
                landmarks.append(landmark)
            
                label_string = videos[0].replace("\\","/").split("/")[2]   
                label = self.label_map[label_string]

            landmarks = np.array(landmarks).reshape(-1, 543, 3)
            landmark_dict = take_all_landmarks_processing(landmarks, config["landmarks"]["face_landmarks"], "./configs/config.yaml")

            landmark = self.process_data(landmark_dict, self.transform, frame_drop= self.frame_drop)
            
            landmark = torch.tensor(landmark, dtype=torch.float32)
            label = torch.tensor(label, dtype = torch.long)
            return landmark, label    