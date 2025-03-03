from src.utils.utils import *

import numpy as np
import mediapipe as mp
import cv2
import os

config =load_config("./configs/config.yaml")

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities
actions = np.array(["Hello", "My", "Name", "Q", "u", "y", "Bye"])
no_sequence = 10 # video 
no_frame_sequence = 30 # FPS

def create_folder():
    for action in actions: 
        for sequence in range(no_sequence):
            try: 
                os.makedirs(os.path.join(config["paths"]["data_dir"], action, str(sequence)))
            except:
                pass

def data_collection():
    cap = cv2.VideoCapture(1)
    with mp_holistic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as holistic:

        for action in actions:
            for sequence in range(no_sequence):
                for frame_num in range(no_frame_sequence):

                    ret, frame = cap.read()

                    image, result = mediapipe_detection(frame, holistic)

                    draw_landmark(image, result, mp_drawing, mp_holistic)

                    if frame_num == 0:
                        cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # Show to screen
                        cv2.imshow('OpenCV Feed', image)
                        cv2.waitKey(500)
                    else: 
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # Show to screen
                        cv2.imshow('OpenCV Feed', image)
                
                    # NEW Export keypoints
                    keypoints = extract_keypoints(result)
                    npy_path = os.path.join(config["paths"]["data_dir"], action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)

                    # Break gracefully
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
    
        cap.release()
        cv2.destroyAllWindows()



