# -*- coding: utf-8 -*-
import click
import logging
from dotenv import find_dotenv, load_dotenv

from src.model.config import ALL_CONFIGS
from src.predict.predict_model import predict_model
from src.utils.split import Split


@click.command()
@click.option('-e', '--example', is_flag=True)
@click.option('-i', '--partition_idx', default=0)
@click.option('-n', '--num_partitions', default=1)
@click.option('-c', '--cpus', default=0)
@click.option('-g', '--gpus', default=0)
def main(example, partition_idx, num_partitions, cpus, gpus):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('predicting model')

    # Predict jobs for each config in ALL_CONFIGS...
    #  - TRAIN_SUBSET + VAL
    #  - TRAIN_SUBSET + VAL_SUBSET
    #  - TRAIN + VAL
    #  - TRAIN + VAL_SUBSET
    jobs = [
        (config, use_subset, split, norm_split)
        for use_subset in [True, False]
        for config in ALL_CONFIGS
        for split in [Split.VAL, Split.VAL_SUBSET]
        for norm_split in [Split.TRAIN_SUBSET if use_subset else Split.TRAIN,]
    ]

    N = len(jobs)
    start_idx = int(partition_idx*N/num_partitions)
    end_idx = int((partition_idx+1)*N/num_partitions)
    jobs_i = jobs[start_idx:end_idx]

    # Train models.
    for job in jobs_i:
        config, use_subset, split, norm_split = job
        print(f"Predicting config: {str(config)} on split {split} using norm {norm_split} (use_subset == {use_subset})")
        predict_model(config, example, use_subset, split, norm_split, cpus, gpus)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
