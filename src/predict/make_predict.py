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
@click.option('-c', '--cpus', default=4)
def main(example, partition_idx, num_partitions, cpus):
    """Make model predictions on validation splits."""
    logger = logging.getLogger(__name__)
    logger.info('predicting model')

    # Predict jobs on VAL and VAL_SUBSET for each config in ALL_CONFIGS
    jobs = [
        (config, use_subset, split, norm_split)
        for use_subset in [True,]
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
        print(f"Predicting config: {config.name} on split {split} using norm {norm_split} (use_subset == {use_subset})")
        predict_model(config, example, use_subset, split, norm_split, cpus)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
