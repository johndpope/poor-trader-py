
from poor_trader import trading


class PSEBroker(trading.Broker):
    def __init__(self, market):
        super(PSEBroker, self).__init__(name='PSEBroker')
        self.market = market

    def trade(self, trading_period, symbol, quantity, action, position=None):
        if quantity <= self.market.get_volume(trading_period, symbol) or action == trading.Action.CLOSE:
            if action == action.OPEN_LONG:
                price = self.market.get_price(trading_period, symbol, field='Open')
                value = price * quantity
                value_with_fees = self.calculate_buy_value(price, quantity)
                position = position or trading.Position(symbol, quantity, trading.Direction.LONG)
                position.add_open_long_transaction(trading_period, symbol, price, quantity, value_with_fees)
                return position
            elif action == action.CLOSE:
                price = self.market.get_price(trading_period, symbol, field='Open')
                value = price * quantity
                value_with_fees = self.calculate_sell_value(price, quantity)
                position = position or trading.Position(symbol, quantity, trading.Direction.LONG)
                position.add_close_transaction(trading_period, symbol, price, quantity, value_with_fees)
                return position
            elif action == action.OPEN_SHORT:
                return None
            else:
                raise RuntimeError
        else:
            return None


