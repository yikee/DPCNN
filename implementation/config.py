import torch

class Config(object):
    def __init__(self):
        self.USE_CUDA     = True
        self.Model = "DPCNN"
        self.MODEL_NAME  = "DPCNN"
        self.RUNNING_ON_SERVER = True
        self.SUMMARY_PATH = "summary/DPCNN"
        self.NET_SAVE_PATH = "./source/trained_net"
        self.TRAIN_DATASET_PATH = "new_train"
        self.TEST_DATASET_PATH = "new_test"
        self.NUM_EPOCHS = 50
        self.BATCH_SIZE = 40
        self.NUM_TRAIN = self.NUM_EPOCHS * self.BATCH_SIZE
        self.NUM_TEST = 0
        self.TEST_STEP = 100
        self.TOP_NUM = 3
        self.NUM_WORKERS = 8
        self.IS_TRAINING = True
        self.ENSEMBLE_TEST = False
        self.LEARNING_RATE = 0.001
        self.RE_TRAIN = False
        self.USE_PAIR_MAPPING = False
        self.USE_TRAD2SIMP = False
        self.TEST_POSITION = 'Gangge Server'

        self.OPTIMIZER = 'SGD'
        self.USE_CHAR = False
        self.SENT_LEN = 13
        self.USE_WORD2VEC = False
        self.BANLANCE = True
        self.NUM_CLASSES = 3
        self.EMBEDDING_DIM = 91
        self.NUM_ID_FEATURE_MAP = 7

        self.MODEL_NAME_LIST = ['DPCNN']
        self.MODEL_THETA_LIST = [0]