import cv2
import numpy as np
import yaml
import random
from src.data_processing.data_augmentation import Affine_Transformation

# Load config
def load_config(file_path):
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)
    return config

# take frame and detect the landmark
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

# draw and color landmark
def draw_landmark(image, results, mp_drawing, mp_holistic):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                            mp_drawing.DrawingSpec(color = (80, 110, 10), thickness = 1, circle_radius = 1),
                            mp_drawing.DrawingSpec(color = (80, 256, 121), thickness = 1, circle_radius = 1)    )
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

# Convert into numpy
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

# Left hand indice
def l_hand_indexes():
    l_hand_indexes =  []
    for keypoint_name in load_config["hand_landmarks"]:
        for index in load_config["hand_landmarks"][keypoint_name]:
            l_hand_indexes.append(index + load_config["left_start_index"])
    l_hand_indexes =  list(set(l_hand_indexes))
    l_hand_indexes.sort()
    l_hand_indexes =  np.array(l_hand_indexes)
    return l_hand_indexes

# Right hand indice
def r_hand_indexes():
    r_hand_indexes =  []
    for keypoint_name in load_config["hand_landmarks"]:
        for index in load_config["hand_landmarks"][keypoint_name]:
            r_hand_indexes.append(index + load_config["right_start_index"])
    r_hand_indexes =  list(set(r_hand_indexes))
    r_hand_indexes.sort()
    r_hand_indexes =  np.array(r_hand_indexes)
    return r_hand_indexes


def take_all_landmarks_processing(full_landmarks, face_landmarks):
    midwayBetweenEyes = full_landmarks[:, 168]
    mean_lips = np.nanmean(midwayBetweenEyes, axis = 0, keepdims = True)
    full_landmarks = full_landmarks - mean_lips
    left_hand = full_landmarks[:, l_hand_indexes]
    right_hand = full_landmarks[:, r_hand_indexes]
    lips_indexes = face_landmarks["lipsUpperOuter"] + face_landmarks["lipsLowerOuter"] + face_landmarks["lipsUpperInner"] + face_landmarks["lipsLowerInner"]
    lips = full_landmarks[:, lips_indexes]
    landmark_dict = dict(left_hand=left_hand, right_hand=right_hand, lips=lips)
    return landmark_dict

def augmentation(landmarks, aug_params):
    angle_rotation = random.gauss(0, aug_params["angle"]/2)
    scale = random.gauss(1, aug_params["scale"]/2)
    translation_x = random.gauss(0, aug_params["shift_x"]/2)
    translation_y = random.gauss(0, aug_params["shift_y"]/2)
    shift_x = random.gauss(0, aug_params["shift_x"]/2)
    shift_y = random.gauss(0, aug_params["shift_y"]/2)
    angle_skew_x = random.gauss(0, aug_params["angle_skew_x"]/2)
    angle_skew_y = random.gauss(0, aug_params["angle_skew_y"]/2)
    Affine = Affine_Transformation()
    Affine.random_rotation(-angle_rotation, angle_rotation)
    Affine.scaling(scale)
    Affine.translation(-translation_x, translation_y)
    Affine.skew_x_degree(angle_skew_x)
    Affine.skew_y_degree(angle_skew_y)
    aug_landmarks_z = landmarks[:, :, 2][:, :, None]
    aug_landmarks = landmarks[:, :, :2]
    aug_landmarks = Affine.transform(aug_landmarks)
    aug_landmarks = np.concatenate((aug_landmarks, aug_landmarks_z), axis=2)
    aug_landmarks = aug_landmarks + landmarks[:, 0][:, None]
    aug_landmarks = aug_landmarks.astype(np.float32)
    return aug_landmarks

