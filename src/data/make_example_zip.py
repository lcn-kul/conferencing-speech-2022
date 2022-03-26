# -*- coding: utf-8 -*-
import click
import logging
from dotenv import find_dotenv, load_dotenv

from src.data.create_example_zip import create_example_zip


@click.command()
def main():
    """Create ZIP with example data.."""
    logger = logging.getLogger(__name__)
    logger.info('creating example ZIP')
    create_example_zip()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
