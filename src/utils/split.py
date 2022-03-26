from enum import Enum


class Split(Enum):
    TRAIN = 0
    VAL = 1
    TRAINVAL = 2
    TEST = 3
    TRAIN_SUBSET = 4
    VAL_SUBSET = 5


# ALL_SPLITS = [Split.TRAIN, Split.VAL, Split.TRAINVAL, Split.TEST]
ALL_SPLITS = [Split.TRAIN, Split.TRAIN_SUBSET, Split.VAL, Split.VAL_SUBSET, Split.TRAINVAL]
