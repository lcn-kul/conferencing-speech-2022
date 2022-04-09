# -*- coding: utf-8 -*-
import click
import logging
from dotenv import find_dotenv, load_dotenv

from src.data.shard.create_shards import create_shards
from src.utils.split import Split, ALL_SPLITS, DEV_SPLITS


@click.command()
@click.option('-e', '--example', is_flag=True)
@click.option('-i', '--partition_idx', default=0)
@click.option('-n', '--num_partitions', default=1)
def main(example, partition_idx, num_partitions):
    """Create WebDataset shards from the extracted features."""
    logger = logging.getLogger(__name__)
    logger.info('creating shards')

    N = len(DEV_SPLITS)

    start_idx = int(partition_idx*N/num_partitions)
    end_idx = int((partition_idx+1)*N/num_partitions)
    splits = DEV_SPLITS[start_idx:end_idx]


    # Extract features.
    for split in splits:
        create_shards(split, example=True)
        if not example:
            create_shards(split, example=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
