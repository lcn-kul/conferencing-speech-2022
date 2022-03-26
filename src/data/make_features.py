# -*- coding: utf-8 -*-
import click
import logging
from dotenv import find_dotenv, load_dotenv

from src.data.extract_features.extract_features import extract_features
from src.utils.split import Split


@click.command()
@click.option('-e', '--example', is_flag=True)
@click.option('-i', '--partition_idx', default=0)
@click.option('-n', '--num_partitions', default=1)
def main(example, partition_idx, num_partitions):
    """Extract features for the datasets."""
    logger = logging.getLogger(__name__)
    logger.info('extracting features')

    # Extract features.
    extract_features(Split.TRAIN, True, partition_idx, num_partitions)
    extract_features(Split.VAL, True, partition_idx, num_partitions)
    # extract_features(Split.VAL_SUBSET, True, partition_idx, num_partitions)
    # extract_features(Split.TRAINVAL, example=True, partition_idx, num_partitions)
    extract_features(Split.TEST, True, partition_idx, num_partitions)
    if not example:
        extract_features(Split.TRAIN, False, partition_idx, num_partitions)
        extract_features(Split.VAL, False, partition_idx, num_partitions)
        # extract_features(Split.VAL_SUBSET, False, partition_idx, num_partitions)
        # extract_features(Split.TRAINVAL, example=False)
        extract_features(Split.TEST, False, partition_idx, num_partitions)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
