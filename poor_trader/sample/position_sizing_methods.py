
from poor_trader import trading


class EquityPctBased(trading.PositionSizing):
    def __init__(self, market, total_risk_pct=0.01, unit_risk_pct=0.2):
        super(EquityPctBased, self).__init__(name='EquityPctBased')
        self.market = market
        self.total_risk_pct = total_risk_pct
        self.unit_risk_pct = unit_risk_pct

    def calculate_quantity(self, trading_period, symbol, portfolio):
        price = self.market.get_price(trading_period, symbol, field='Open')
        boardlot = self.market.get_boardlot(trading_period, symbol)
        C = portfolio.equity * self.total_risk_pct
        R = price * self.unit_risk_pct
        P = C / R
        multiplier = int(P / boardlot)
        multiplier = 1 if multiplier <= 0 else multiplier
        return multiplier * boardlot

    def calculate_total_risk(self, price, shares):
        return shares * (self.unit_risk_pct * price)
