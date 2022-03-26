# -*- coding: utf-8 -*-
import click
import logging
from dotenv import find_dotenv, load_dotenv

from src.eval.combine_csvs import combine_csvs
from src.eval.eval import eval


@click.command()
def main():
    """Evaluate models on the validation set(s)."""
    logger = logging.getLogger(__name__)
    logger.info('evaluating models')

    # Combine ground-truth + prediction CSVs.
    combine_csvs()

    # Evaluate combined CSV's.
    eval()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
