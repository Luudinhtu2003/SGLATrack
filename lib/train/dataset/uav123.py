print("Hello 22")

import torch
import os
import os.path
import numpy as np
import pandas
import random
print("Hello 22")

from collections import OrderedDict
print("Hello 22")
from lib.train.data import jpeg4py_loader
from .base_video_dataset import BaseVideoDataset
from lib.train.admin import env_settings
from tqdm import tqdm
print("Hello 22")
SEQ_PATH = r"/home/getac2/tuld3/tu_workspace/track_uav/SGLATrack/data/UAV123/data_seq/UAV123"


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

        sequence_list = list_sequences(SEQ_PATH)
        self.sequence_list = sequence_list
        print("UAV123: {} sequences".format(len(self.sequence_list)))
        print("UAV123 sequences: ", self.sequence_list)
        self.seq_to_class_map, self.seq_per_class = self._load_class_info()

    def _load_class_info(self):
        ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
        class_map_path = os.path.join(ltr_path, 'data_specs', 'mapping.txt')

        with open(class_map_path, 'r') as f:
            seq_to_class_map = {seq_class.split('\t')[0]: seq_class.rstrip().split('\t')[1] for seq_class in f}
        print("seq_to_class_map: ", seq_to_class_map)
        
        seq_per_class = {}
        for i, seq in enumerate(self.sequence_list):
            class_name = seq_to_class_map.get(seq, 'Unknown')
            if class_name not in seq_per_class:
                seq_per_class[class_name] = [i]
            else:
                seq_per_class[class_name].append(i)

        return seq_to_class_map, seq_per_class

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
            anno_file = os.path.dirname(anno_file) + "/" + video_name + "_1.txt"
            if os.path.isfile(anno_file):
                pass
            else:
                raise FileNotFoundError("Annotation file not found: {}".format(anno_file))
            


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

        frame_id = str(frame_id).zfill(6)
        frame_path = os.path.join(self.root, "data_seq", "UAV123", video_name, str(frame_id) + ".jpg")
        # print("Frame path: ", frame_path)

        if not os.path.isfile(frame_path):
            raise FileNotFoundError("Frame file not found: {}".format(frame_path))
        return self.image_loader(frame_path)
    
    def _get_class(self, seq_id):
        seq_name = self.sequence_list[seq_id]
        return self.seq_to_class_map[seq_name]
    
    def get_class_name(self, seq_id):
        obj_class = self._get_class(seq_id)

        return obj_class
    
    def get_frames(self, seq_id, frame_ids, anno=None):
        
        frame_list = [self._get_frame(seq_id, f) for f in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        object_meta = OrderedDict({'object_class_name': self.get_class_name(seq_id),
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, object_meta
    

