import os
import random
import natsort
import torch
import cv2
import pandas as pd
from torchvision.io import read_image
import numpy as np
from sklearn.preprocessing import LabelEncoder

def read_image_and_resize(path, size=256):
    img = cv2.imread(path)
    img = cv2.resize(img, (256, 256))
    img = np.transpose(img, (2, 0, 1))/255.0
    return img.astype(np.float32)

def read_single_csv_to_np(path):
    # List all files in the directory
    files = os.listdir(path)

    # Filter only csv files
    csv_files = [file for file in files if file.endswith('.csv')]

    # Ensure there's exactly one csv file
    if len(csv_files) != 1:
        raise ValueError("Expected one CSV file, found {}.".format(len(csv_files)))

    # Construct the full path to the CSV file
    csv_path = os.path.join(path, csv_files[0])

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_path)

    # Convert DataFrame to numpy arrays
    np_arrays = df.to_numpy()

    return np_arrays

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir="./SampleData/", T=14):
        self.T = T
        self.categories = []
        self.video_paths = []
        self.label_encoder = LabelEncoder()

        # Iterate over categories (folders)
        for category_folder in natsort.natsorted(os.listdir(root_dir)):
            category_path = os.path.join(root_dir, category_folder)
            if os.path.isdir(category_path):  # Check if it's a directory
                self.categories.append(category_folder)

                # Iterate over videos in the category
                for video_folder in natsort.natsorted(os.listdir(category_path)):
                    video_path = os.path.join(category_path, video_folder)
                    self.video_paths.append(video_path)

        # Fit label encoder
        self.label_encoder.fit(self.categories)
        self.num_categories = len(self.label_encoder.classes_)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        fp = self.video_paths[idx]
        f_in = fp + "/IN/"
        f_out = fp + "/OUT/"

        loi_in = natsort.natsorted(os.listdir(f_in))
        loi_out = natsort.natsorted(os.listdir(f_out))
        assert len(loi_in) == len(loi_out), "Frame length mismatch in: " + fp

        frames_in = []
        frames_out = []

        N = random.randint(0, len(loi_in) - self.T - 1)
        #print("[Dataset]:", f_in, len(loi_in))
        for i in range(N, N + self.T):
            frames_in.append(read_image_and_resize(f_in + loi_in[i]))
            frames_out.append(read_image_and_resize(f_out + loi_out[i]))

        excel_data = read_single_csv_to_np(fp)

        # Encode category
        category = os.path.basename(os.path.dirname(fp))
        category_encoded = self.label_encoder.transform([category])[0]
        category_one_hot = torch.zeros(self.num_categories)
        category_one_hot[category_encoded] = 1

        return frames_in, frames_out, torch.tensor(excel_data[N:N+self.T, :].astype(np.float32)), category_one_hot
