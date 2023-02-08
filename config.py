import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#TRAIN_DIR = "data/train"
#VAL_DIR = "data/val"
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 1
LOAD_MODEL = False
SAVE_MODEL = False
CHECKPOINT_GEN_A2B = "Checkpoints/Generated_data/gen_cos.pth.tar"
CHECKPOINT_GEN_B2A = "Checkpoints/Generated_data/gen_sin.pth.tar"
CHECKPOINT_DISC_A =  "Checkpoints/Generated_data/disc_sin.pth.tar"
CHECKPOINT_DISC_B =  "Checkpoints/Generated_data/disc_cos.pth.tar"
