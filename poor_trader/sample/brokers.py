
from poor_trader import trading


class PSEBroker(trading.Broker):
    def __init__(self, market):
        super(PSEBroker, self).__init__(name='PSEBroker')
        self.market = market

    def calculate_commission(self, price, shares, commission=0.0025):
        value = price * shares
        com = value * commission
        return com if com > 20 else 20

    def calculate_buying_fees(self, price, shares, commission=0.0025, vat_on_commission=0.12, pse_trans_fee=0.00005, sccp=0.0001):
        value = price * shares
        com = self.calculate_commission(price, shares, commission=commission)
        vat_com = com * vat_on_commission
        trans = value * pse_trans_fee
        sccp_fee = value * sccp
        return com + vat_com + trans + sccp_fee

    def calculate_selling_fees(self, price, shares, sales_tax=0.006):
        tax = price * shares * sales_tax
        return self.calculate_buying_fees(price, shares) + tax

    def calculate_buy_value(self, price, shares):
        if shares <= 0:
            return 0
        return price * shares + self.calculate_buying_fees(price, shares)

    def calculate_sell_value(self, price, shares):
        if shares <= 0:
            return 0
        return price * shares - self.calculate_selling_fees(price, shares)

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


