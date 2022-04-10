# -*- coding: utf-8 -*-
import click
import logging
from dotenv import find_dotenv, load_dotenv

from src.data.download.download_train import download_train
from src.data.download.download_test import download_test
from src.data.process_raw_test_csv.process_raw_test_csv import process_raw_test_csv
from src.data.process_raw_csvs.process_raw_csvs import process_raw_csvs
from src.utils.split import Split


@click.command()
def main():
    """Download datasets and process raw CSVs. Results will be saved in
    conferencing-speech-2022/data/processed.
    """
    logger = logging.getLogger(__name__)
    logger.info('Downloading datasets + processing raw CSVs')

    # Download datasets.
    download_train()
    # download_test() # *** TUB data unavailable as of 07/04/2022! (Permission Denied)

    # Preprocess raw test CSV since its format is slightly different from the
    # rest of the pipeline.
    # process_raw_test_csv() # *** TUB data unavailable as of 07/04/2022! (Permission Denied)

    # Transform the raw CSVs into the standardized format.
    process_raw_csvs(Split.TRAIN, example=True)
    process_raw_csvs(Split.TRAIN_SUBSET, example=True)
    process_raw_csvs(Split.VAL, example=True)
    process_raw_csvs(Split.VAL_SUBSET, example=True)
    # process_raw_csvs(Split.TEST, example=True)
    process_raw_csvs(Split.TRAIN, example=False)
    process_raw_csvs(Split.TRAIN_SUBSET, example=False)
    process_raw_csvs(Split.VAL, example=False)
    process_raw_csvs(Split.VAL_SUBSET, example=False)
    # process_raw_csvs(Split.TEST, example=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
