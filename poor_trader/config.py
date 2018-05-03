
import os
from path import Path
from poor_trader.utils import makedirs


class Config(object):
    def __init__(self, name='_config'):
        self.name = name

MAIN_PATH = Path(os.path.dirname(__file__)).parent
RESOURCES_PATH = MAIN_PATH / 'resources'

INDICATORS_OUTPUT_PATH = RESOURCES_PATH / 'indicators'
SYSTEMS_PATH = RESOURCES_PATH / 'systems'
PORTFOLIO_PATH = RESOURCES_PATH / 'portfolio'
SAMPLE_PATH = RESOURCES_PATH / 'sample'

TRADING_SYSTEMS_PATH = SAMPLE_PATH / 'trading_systems'
HISTORICAL_DATA_PATH = RESOURCES_PATH / 'historical_data.pkl'

makedirs(INDICATORS_OUTPUT_PATH)
makedirs(SYSTEMS_PATH)
makedirs(PORTFOLIO_PATH)
makedirs(TRADING_SYSTEMS_PATH)

