import albumentations as A
from albumentations.pytorch import ToTensorV2

IBSR_or_iSeg = 'iSeg'  # or 'IBSR'
FOLDS_PATHS = f"/datasets/{IBSR_or_iSeg}_all_folds"
if IBSR_or_iSeg == 'iSeg':
    NUM_EPOCHS = 120
elif IBSR_or_iSeg == 'IBSR':
    NUM_EPOCHS = 60

LEARNING_RATE = 1e-4
BATCH_SIZE = 32
NUM_WORKERS = 2

IMAGE_SIZE = 184
CHANNELS_IMG = 1
L1_LAMBDA = 100
LAMBDA_BOTTLENECK = 10

MEAN = 0.2337
STD = 0.2653

AMOUNT_OF_EPOCHS_TO_SAVE_MODEL = 5
AMOUNT_OF_EPOCHS_TO_SAVE_EXAMPLE = 1

LAMBDA_ADV = 10
LAMBDA_SYM = 5e-2
LAMBDA_UC = 1e-2

# ~~~~~~~~~~~ Augmentations ~~~~~~~~~~~


augmentation_train = A.Compose(
    [
        # A.Resize(RES,RES),
        A.HorizontalFlip(p=0.5),
        # A.VerticalFlip(p=0.5),
        # A.Normalize(mean=0.2337,std=0.2653),
        ToTensorV2(),
    ], additional_targets={"image0": "image",
                           "image1": "image",
                           "image2": "image"}, )

augmentation_val = A.Compose(
    [
        # A.Resize(RES,RES),
        # A.HorizontalFlip(p=0.5),
        # A.VerticalFlip(p=0.5),
        # A.Normalize(mean=0.2337,std=0.2653),
        ToTensorV2(),
    ], additional_targets={"image0": "image",
                           "image1": "image",
                           "image2": "image"}, )