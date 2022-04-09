from enum import Enum


class Split(Enum):
    TRAIN = 0
    TRAIN_SUBSET = 1
    VAL = 2
    VAL_SUBSET = 3
    TEST = 4


DEV_SPLITS = [Split.TRAIN, Split.TRAIN_SUBSET, Split.VAL, Split.VAL_SUBSET]
ALL_SPLITS = [Split.TRAIN, Split.TRAIN_SUBSET, Split.VAL, Split.VAL_SUBSET, Split.TEST]
