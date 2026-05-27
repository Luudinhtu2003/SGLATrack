import torch
import os
import os.path
import numpy as np
import pandas
import random
from collections import OrderedDict

from lib.train.data import jpeg4py_loader
from .base_video_dataset import BaseVideoDataset
from lib.train.admin import env_settings


SEQ_PATH = "F:\Tu_workspace\SGLATrack\data\UAV123\data_seq\UAV123"

def list_sequences(root):
    """ Lists all the videos in the input set_ids. Returns a list of tuples (set_id, video_name)

    args:
        root: Root directory to UAV123

    returns:
        list - list of tuples (set_id, video_name) containing the set_id and video_name for each sequence
    """
    sequence_list = []

    for s in os.listdir(root):
        if os.path.isdir(os.path.join(root, s)):
            sequence_list.append(s)
    return sequence_list

class UAV123(BaseVideoDataset):
    """ UAV123 dataset.

    Publication:
        The UAV123 Dataset for Evaluating Single-Object Tracking.
        Matthias Mueller, Adel Bibi, Silvio Giancola, Salman Al-Subaihi and Bernard Ghanem
        ECCV, 2016
        https://ivul.kaust.edu.sa/Documents/Publications/2016/The%20UAV123%20Dataset%20for%20Evaluating%20Single-Object%20Tracking.pdf
    Download the dataset from http://ivul.kaust.edu.sa/Pages/Publications.aspx.
    """
    def __init__(self, root=None, image_loader=jpeg4py_loader):
        """
        args:
            root        - The path to the UAV123 folder, containing the training sets.
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (
        """
        root = env_settings().uav123_dir if root is None else root
        super().__init__('UAV123', root, image_loader)

        sequence_list = list_sequences(self.root)
        self.sequence_list = sequence_list

    def _load_class_info(self):
        pass

    def get_name(self):
        """ Name of the dataset

        returns:
            string - Name of the dataset
        """
        return "UAV123"
    
    def has_class_info(self):
        """ Whether the dataset has class info

        returns:
            bool - True if the dataset has class info
        """
        return False

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]
    
    def _read_bb_anno(self, seq_id):
        """ Reads the bounding box annotation for a sequence. The annotation is expected to be in a text file with the same name as the sequence, and each line in the text file should contain the bounding box coordinates for the corresponding frame in the format: x,y,width,height

        args:
            sequence_name: Name of the sequence for which to read the annotation

        returns:
            list - A list of bounding box annotations, where each annotation is a list of [x, y, width, height]
        """
        video_name = self.sequence_list[seq_id]
        anno_folder = os.path.join(self.root, "anno", "UAV123")
        anno_file = os.path.join(anno_folder, video_name + ".txt")
        if not os.path.isfile(anno_file):
            raise FileNotFoundError("Annotation file not found for sequence: {}".format(video_name))
        with open(anno_file, "r") as f:
            lines = f.readlines()
            bb_anno = []
            for line in lines:
                line = line.strip()
                if line:
                    coords = list(map(float, line.split(",")))
                    bb_anno.append(coords)
        return torch.tensor(bb_anno, dtype=torch.float32)

    def get_sequence_info(self, seq_id):
        bbox = self._read_bb_anno(seq_id)

        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.clone().byte()
        return {'bbox': bbox, 'valid': valid, 'visible': visible}
    

    def _get_frame(self, seq_id, frame_id):
        video_name = self.sequence_list[seq_id]
        frame_path = os.path.join(self.root, "data_seq", "UAV123", video_name, str(frame_id) + ".jpg")
        if not os.path.isfile(frame_path):
            raise FileNotFoundError("Frame file not found: {}".format(frame_path))
        return self.image_loader(frame_path)
    
    def _get_class(self, seq_id):
        seq_name = self.sequence_list[seq_id][1]
        return self.seq_to_class_map[seq_name]
    
    def get_class_name(self, seq_id):
        obj_class = self._get_class(seq_id)

        return obj_class
    
    def get_frames(self, seq_id, frame_ids, anno=None):
        frame_list = [self._get_frame(seq_id, f) for f in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}

        return frame_list, anno_frames, {'class': self.get_class_name(seq_id)}
    

