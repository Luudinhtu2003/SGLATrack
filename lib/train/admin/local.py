class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/getac2/tuld3CVPR25_REB/sgla/Deit_MLP_withnograd_maxcos'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/home/getac2/tuld3CVPR25_REB/sgla/Deit_MLP_withnograd_maxcos/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/home/getac2/tuld3CVPR25_REB/sgla/Deit_MLP_withnograd_maxcos/pretrained_networks'
        self.lasot_dir = '/home/getac2/tuld3lasot'
        self.got10k_dir = '/home/getac2/tuld3got10k/train'
        self.got10k_val_dir = '/home/getac2/tuld3got10k/val'
        self.lasot_lmdb_dir = '/home/getac2/tuld3lasot_lmdb'
        self.got10k_lmdb_dir = '/home/getac2/tuld3got10k_lmdb'
        self.trackingnet_dir = '/home/getac2/tuld3trackingnet'
        self.trackingnet_lmdb_dir = '/home/getac2/tuld3trackingnet_lmdb'
        self.coco_dir = '/home/getac2/tuld3coco'
        self.coco_lmdb_dir = '/home/getac2/tuld3coco_lmdb'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '/home/getac2/tuld3vid'
        self.imagenet_lmdb_dir = '/home/getac2/tuld3vid_lmdb'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
        self.uav123_dir = r"/home/getac2/tuld3/tu_workspace/track_uav/SGLATrack/data/UAV123"
        self.uav123_lmdb_dir = r"/home/getac2/tuld3/tu_workspace/track_uav/SGLATrack/data"
