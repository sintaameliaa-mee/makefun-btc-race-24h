#!/usr/bin/env python3
"""Dip-and-breakout BTC trading bot for make.fun btc-race-24h-20261115-0828.

Trades Coinbase Wrapped BTC (CBBTC) against USDC, long only, one position at
a time, on 1-minute bars replayed through the agent-fun/bars-replay/v1
protocol (hello, one batch per bar, done). Every order decided on a bar's
close fills at the next bar's open, so all signals read closes only.

Strategy -- buy the dip after it stops falling, ride the breakout
---------------------------------------------------------------
Bitcoin's price wanders around its recent level, and the trailing 4-hour
mean (240 one-minute closes) is the level it keeps coming back to: our
earlier BTC forecast work showed a mild pull toward that mean beats
persistence, and the dev window's own bars show the same (a price 200+ bps
below the 240-minute mean bounces hard over the next hour). But catching a
falling knife loses money, so the bot only buys a deep dip once the fall
has paused:

* Mean reversion: when the close is more than 200 bps below the trailing
  240-minute mean AND the last 10 minutes are no longer plunging (10-minute
  return above -0.1%) OR the 14-bar RSI has just crossed back above 30,
  buy with all available cash. Sell the whole position when the close
  climbs back above the 240-minute mean.
* Trend: when the close crosses above the 240-minute mean by more than
  100 bps on a strong 30-minute rise (over +1%), buy with all cash and
  hold while the price stays above the mean; sell when it falls back
  below the mean. This leg catches sustained up moves the dip leg would
  otherwise sit out.
* Fallback: if no order has filled by bar 1400 (40 minutes left),
  buy 10% of cash so the entry still meets the competition's
  minimum-trade guardrail on a quiet day; the position is topped up to
  full size if a mean-reversion or trend signal fires while it is held,
  and otherwise the replay's final flattening sells it.

Costs: a full-size round trip pays about 26-40 bps in fees, modelled
slippage and pool impact, so the bot trades rarely (a handful of round
trips per day) and only on moves large enough to clear that cost several
times over. Every order caps its slippage at 100 bps past the close it
was decided on, so a fill into a sudden gap is refused rather than
chased.

Determinism: standard library only, no network, no clock, no randomness
and no environment reads; the same bars always produce the same orders.
State is a bounded list of closes (at most 1,440) plus a few scalars, far
under the sandbox's memory limit.

Backtest (own simulator, exact harness fill model):
* Dev window (24h to the opening, BTC -4.0%): net PnL +$24.92 (+2.49%),
  8 trades, max drawdown 1.40%. Buy-and-hold on the same window: -$44.01.
* Out of sample, Coinbase BTC/USD 1-minute bars, 9 days (5-13 Nov 2026):
  total -$2.41, worst day -$17.31, trades on 9/9 days. Buy-and-hold on
  the same days: -$196.68.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))
import makefun_bars

SYMBOL = "CBBTC"

# --- Strategy constants (tuned on the dev window, checked out of sample) ---
WINDOW = 240            # trailing mean length, in 1-minute bars (4 hours)
ENTRY_DEV_BPS = -200    # buy the dip once the close is this far below the mean
EXIT_DEV_BPS = 0        # sell the dip trade once the close is back above the mean
STAB_BARS = 10          # stabilisation lookback for the dip entry
STAB_MIN_RET = -0.001   # dip entry needs the last 10m return above -0.1%
RSI_BARS = 14           # RSI lookback, in 1-minute closes
RSI_TURN = 30           # dip entry also fires on RSI crossing back above this
TREND_CROSS_BPS = 100   # trend entry needs the close this far above the mean
TREND_WIN = 30          # trend momentum lookback, in bars
TREND_MIN_RET = 0.01    # trend entry needs a 30-minute rise above +1%
FALLBACK_BAR = 1400     # bar from which the guardrail fallback may buy
FALLBACK_FRAC = 0.10    # fallback buys this share of cash
MAX_SLIPPAGE_BPS = 100  # every order refuses a fill past 1% from the close


class DipBot:
    """Mean-reversion + breakout bot over one symbol's closes."""

    def __init__(self):
        self.closes = []        # every close seen, oldest first (price ints)
        self.prev_dev_bps = None
        self.prev_rsi = None
        self.mode = None        # 'mr', 'trend' or 'fallback' while holding
        self.filled = False     # any of our orders has filled yet this run

    def _dev_bps(self):
        """The latest close's deviation from the trailing mean, in bps."""
        window = self.closes[-WINDOW:]
        mean = sum(window) / len(window)
        return (self.closes[-1] / mean - 1.0) * 10000.0

    def _ret_bars(self, n):
        """The return over the last n bars, as a fraction."""
        return self.closes[-1] / self.closes[-1 - n] - 1.0

    def _rsi(self):
        """RSI over the last RSI_BARS one-minute changes (simple averages)."""
        changes = [self.closes[i] - self.closes[i - 1]
                   for i in range(len(self.closes) - RSI_BARS, len(self.closes))]
        gain = sum(c for c in changes if c > 0) / RSI_BARS
        loss = sum(-c for c in changes if c < 0) / RSI_BARS
        if gain + loss == 0:
            return 50.0
        return 100.0 * gain / (gain + loss)

    def on_batch(self, state, batch):
        for bar in batch["bars"]:
            if bar["symbol"] == SYMBOL:
                self.closes.append(int(bar["c"]))
        for fill in batch.get("fills", []):
            if fill.get("status") == "filled":
                self.filled = True
        if len(self.closes) < WINDOW:
            return []
        dev = self._dev_bps()
        rsi = self._rsi()
        crossed_rsi = (self.prev_rsi is not None and self.prev_rsi < RSI_TURN
                       and rsi >= RSI_TURN)
        crossed_up = (self.prev_dev_bps is not None
                      and self.prev_dev_bps <= TREND_CROSS_BPS
                      and dev > TREND_CROSS_BPS)
        self.prev_rsi = rsi
        self.prev_dev_bps = dev

        held = state.positions.get(SYMBOL)
        mr_signal = (dev < ENTRY_DEV_BPS
                     and (self._ret_bars(STAB_BARS) > STAB_MIN_RET or crossed_rsi))
        tr_signal = crossed_up and self._ret_bars(TREND_WIN) > TREND_MIN_RET

        if held is None:
            if mr_signal:
                self.mode = "mr"
                return self._maybe_buy(state.cash)
            if tr_signal:
                self.mode = "trend"
                return self._maybe_buy(state.cash)
            if not self.filled and state.seq >= FALLBACK_BAR:
                self.mode = "fallback"
                return self._maybe_buy(int(state.cash * FALLBACK_FRAC))
            return []

        # Holding: a fallback position is topped up by a real signal.
        if self.mode == "fallback" and (mr_signal or tr_signal):
            self.mode = "mr" if mr_signal else "trend"
            return self._maybe_buy(state.cash)
        if self.mode == "mr" and dev > EXIT_DEV_BPS:
            self.mode = None
            return [self._sell(held.qty)]
        if self.mode == "trend" and dev < 0:
            self.mode = None
            return [self._sell(held.qty)]
        return []

    @staticmethod
    def _maybe_buy(notional):
        """A buy order, or no order when the amount is below the 1 USDC
        minimum notional (a smaller order would only be rejected)."""
        notional = int(notional)
        if notional < 1_000_000:
            return []
        return [{"side": "buy", "symbol": SYMBOL,
                 "notional_quote_units": notional,
                 "max_slippage_bps": MAX_SLIPPAGE_BPS}]

    @staticmethod
    def _sell(qty):
        return {"side": "sell", "symbol": SYMBOL, "qty": int(qty),
                "max_slippage_bps": MAX_SLIPPAGE_BPS}


def main():
    bot = DipBot()
    makefun_bars.run(bot.on_batch)


if __name__ == "__main__":
    main()
