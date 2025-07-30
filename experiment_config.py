from pathlib import Path
import os


class Configuration(object):
    def __init__(self) -> None:

        # Working directory
        self.WORKDIR = Path("./submit")
        self.RESOURCES = self.WORKDIR / "resources"
        # Starting weights for the I3D model
        self.MODEL_RGB_I3D = (
            self.RESOURCES / "model_rgb.pth"
        )
        
        # Data parameters
        # Path to the nodule blocks folder provided for the LUNA25 training data. 
        self.DATADIR = Path("./dataset/luna25_nodule_blocks")
        self.MASK_DATADIR = Path("./dataset/luna25_nodule_blocks_mask")

        # Path to the folder containing the CSVs for training and validation.
        self.CSV_DIR = Path("./dataset")

        self.CSV_DIR_TRAIN = self.CSV_DIR / "train_other.csv" # Path to the training CSV
        self.CSV_DIR_VALID = self.CSV_DIR / "valid_1-1.csv" # Path to the validation CSV
        # self.CSV_DIR_TRAIN = self.CSV_DIR / "luna25_training.csv" # Path to the training CSV
        self.VALID_SAMPLES_NUM = 200
        
        # Results will be saved in the /results/ directory, inside a subfolder named according to the specified EXPERIMENT_NAME and MODE.
        self.EXPERIMENT_DIR = self.WORKDIR / "results"
        if not self.EXPERIMENT_DIR.exists():
            self.EXPERIMENT_DIR.mkdir(parents=True)
            
        self.EXPERIMENT_NAME = "amp-batch_32_multitask_seed_2000"
        self.MODE = "3D" # 2D or 3D

        self.DEVICE = "cuda:0"

        # Training parameters
        self.SEED = 2000
        self.NUM_WORKERS = 8
        self.SIZE_MM = 50
        self.SIZE_PX = 64
        self.BATCH_SIZE = 32
        self.ROTATION = ((-20, 20), (-20, 20), (-20, 20))
        self.TRANSLATION = True #default: None
        self.EPOCHS = 60
        self.PATIENCE = 30
        self.PATCH_SIZE = [64, 128, 128]  # default: [64, 128, 128]
        self.LEARNING_RATE = 1e-4   # default: 1e-4
        self.WEIGHT_DECAY = 5e-4
        self.SCHEDULER = "step_custom" # "step" or "step_custom" or "cosine"
        
        # step
        self.STEP_LR_SIZE = 20
        # step_custom
        self.DOWNSTEPS = [60, 80]
        
        ####
        self.SUBMISSION = False
        self.SAVE_EPOCHS = []
        self.ROT_FLIP = 'yz' # 'xyz' or 'yz'
        self.RF_RATIO = [0.3, 0.3]
        self.DROP_RATE = 0.0
        self.UP_SCALE = 2.0
        self.EMA_RATE = 0.998
        self.ELASTIC = False
        self.MIX_PROB = 0.0
        self.LABEL_SMOOTHING_ALPHA = 0.0
        
        # #### Auxiliary config
        self.AUX_TASK = "Segmentation"
        self.AUX_BATCH_SIZE = 16
        self.DETECTOR_DROP_RATE = 0.0
        self.AUX_LOSS_WEIGHT = 0.5
        
config = Configuration()