
import os
from path import Path


MAIN_PATH = Path(os.path.dirname(__file__)).parent

RESOURCES_PATH = MAIN_PATH / 'resources'

INDICATORS_OUTPUT_PATH = RESOURCES_PATH / 'indicators'

SYSTEMS_PATH = RESOURCES_PATH / 'systems'

SAMPLE_PATH = RESOURCES_PATH / 'sample'

PORTFOLIO_PATH = RESOURCES_PATH / 'portfolio'

TRADING_SYSTEMS_PATH = SAMPLE_PATH / 'trading_systems'

HISTORICAL_DATA_PATH = RESOURCES_PATH / 'historical_data.pkl'


def makedirs_ifneeded(path):
    if not os.path.exists(path):
        os.makedirs(path)


def init_path(config):
    config.INDICATORS_OUTPUT_PATH = config.RESOURCES_PATH / 'indicators'
    config.SYSTEMS_PATH = config.RESOURCES_PATH / 'systems'
    config.PORTFOLIO_PATH = RESOURCES_PATH / 'portfolio'
    config.SAMPLE_PATH = config.RESOURCES_PATH / 'sample'
    config.TRADING_SYSTEMS_PATH = config.SAMPLE_PATH / 'trading_systems'

    makedirs_ifneeded(INDICATORS_OUTPUT_PATH)
    makedirs_ifneeded(SYSTEMS_PATH)
    makedirs_ifneeded(PORTFOLIO_PATH)
    makedirs_ifneeded(TRADING_SYSTEMS_PATH)


def init_market_dirs_path(market_name, config):
    config.RESOURCES_PATH  = config.RESOURCES_PATH / market_name
    init_path(config)


class Config(object):
    def __init__(self, name='_config'):
        self.name = name
