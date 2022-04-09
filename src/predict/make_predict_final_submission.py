# -*- coding: utf-8 -*-
import click
import logging
from dotenv import find_dotenv, load_dotenv

from src.model.config import ALL_CONFIGS
from src.predict.predict_final_model_submission import predict_final_model_submission
from src.utils.split import Split


@click.command()
@click.option('-e', '--example', is_flag=True)
@click.option('-i', '--partition_idx', default=0)
@click.option('-n', '--num_partitions', default=1)
@click.option('-c', '--cpus', default=1)
@click.option('-g', '--gpus', default=0)
def main(example, partition_idx, num_partitions, cpus, gpus):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('predicting submission model')

    print("Since some of the test data is unavailable, this code has been commented out.")
    exit(1)

    # Train models.
    modelpath = "final_model_17mar_xlsr_blstm/best-epoch=012-val_loss=0.014164.ckpt"
    split = Split.TEST
    print(f"Predicting final model on split {split})")
    predict_final_model_submission(example, modelpath, split, cpus, gpus)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
