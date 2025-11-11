#!/usr/bin/env python3
from __future__ import annotations
import asyncio
import csv
import dataclasses as dc
import json
import logging
import math
import os
import random
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import aiohttp
import pandas as pd
from dateutil import tz
import argparse

# =========================
# Helpers
# =========================

UTC = timezone.utc
LOCAL_TZ = tz.tzlocal()

def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

def utc_ms(dt_: datetime) -> int:
    return int(dt_.replace(tzinfo=UTC).timestamp() * 1000)

def now_ms() -> int:
    return int(time.time() * 1000)

def roostoo_symbol_from(symbol: str) -> str:
    # Convert "BTC-USD" -> "BTCUSD" for execution
    return "".join(ch for ch in symbol.upper() if ch.isalnum())

# =========================
# Config
# =========================

@dc.dataclass
class Config:
    # Coinbase-only (works from US AWS)
    symbol: str = "BTC-USD"                 # Coinbase format with dash
    interval: str = "1m"
    months_history: int = 3
    data_csv: Path = Path("data_1m.csv")
    trades_csv: Path = Path("trades.csv")

    # Execution (Roostoo)
    roostoo_base: str = os.environ.get("ROOSTOO_BASE", "https://api.roostoo.com")
    roostoo_api_key: Optional[str] = os.environ.get("ROOSTOO_API_KEY")
    roostoo_api_secret: Optional[str] = os.environ.get("ROOSTOO_API_SECRET")

    # Trading risk
    dry_run: bool = False                    # LIVE by default
    max_position_usd: float = 1000.0
    order_notional_fraction: float = 0.5

    # HTTP hardening
    roostoo_timeout: float = float(os.environ.get("ROOSTOO_TIMEOUT", 30))
    roostoo_retries: int = int(os.environ.get("ROOSTOO_RETRIES", 5))

    # Optional override of execution symbol (e.g., BTCUSD)
    roostoo_symbol: Optional[str] = None

    def finalize(self):
        if not self.roostoo_symbol:
            self.roostoo_symbol = roostoo_symbol_from(self.symbol)

# =========================
# Trade Logger
# =========================

class TradeLogger:
    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        ensure_parent(self.csv_path)
        self._init_csv()
        self.log = logging.getLogger("trader")
        self.log.setLevel(logging.INFO)
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        if not any(isinstance(x, logging.StreamHandler) for x in self.log.handlers):
            self.log.addHandler(h)

    def _init_csv(self):
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "timestamp_utc",
                    "symbol",
                    "side",
                    "price",
                    "qty",
                    "notional",
                    "reason",
                    "balance_before",
                    "balance_after",
                    "position_before",
                    "position_after",
                    "order_id",
                    "status",
                ])

    def record(self, *, timestamp: datetime, symbol: str, side: str, price: float, qty: float,
               reason: str, balance_before: float, balance_after: float,
               position_before: float, position_after: float,
               order_id: str, status: str):
        row = [
            timestamp.astimezone(UTC).isoformat(), symbol, side,
            f"{price:.8f}", f"{qty:.8f}", f"{price*qty:.2f}",
            reason, f"{balance_before:.2f}", f"{balance_after:.2f}",
            f"{position_before:.8f}", f"{position_after:.8f}",
            order_id, status
        ]
        with self.csv_path.open("a", newline="") as f:
            csv.writer(f).writerow(row)
        self.log.info(
            f"TRADE {side} {symbol} qty={qty:.6f} @ {price:.2f} | reason={reason} | "
            f"bal {balance_before:.2f}->{balance_after:.2f} | "
            f"pos {position_before:.6f}->{position_after:.6f} | status={status} | oid={order_id}"
        )

# =========================
# Coinbase Data Handler
# =========================

class CoinbaseDataHandler:
    """
    Uses Coinbase Exchange REST candles.
    Polls every 5s to simulate near-real-time minute bars (no WS dependency).
    """
    def __init__(self, cfg: Config, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger
        self.df: pd.DataFrame = pd.DataFrame()
        self.session: Optional[aiohttp.ClientSession] = None
        self.product_id = self.cfg.symbol  # expect "BTC-USD"
        self.base = "https://api.exchange.coinbase.com"
        self.max_candles = 300
        self.granularity = 60
        self.max_span = timedelta(minutes=self.max_candles)  # 300 minutes

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self.session:
            await self.session.close()

    def _init_csv(self):
        ensure_parent(self.cfg.data_csv)
        if not self.cfg.data_csv.exists():
            pd.DataFrame(columns=[
                "open_time","open","high","low","close","volume","close_time"
            ]).to_csv(self.cfg.data_csv, index=False)

    def save_append(self, rows: List[List[Any]]):
        if not rows:
            return
        self._init_csv()
        with self.cfg.data_csv.open("a", newline="") as f:
            w = csv.writer(f)
            for r in rows:
                w.writerow(r)

    def load_csv(self) -> pd.DataFrame:
        if self.cfg.data_csv.exists():
            df = pd.read_csv(self.cfg.data_csv)
            for c in ["open_time","close_time"]:
                df[c] = pd.to_datetime(df[c], unit="ms", utc=True)
            for c in ["open","high","low","close","volume"]:
                df[c] = df[c].astype(float)
            df = df.sort_values("open_time").drop_duplicates(subset=["open_time"], keep="last").reset_index(drop=True)
            self.logger.info(f"Loaded {len(df)} rows from {self.cfg.data_csv}")
            return df
        return pd.DataFrame()

    async def fetch_candles(self, start: Optional[datetime], end: Optional[datetime]) -> List[List[Any]]:
        assert self.session is not None
        params = {"granularity": self.granularity}
        if start and end:
            params["start"] = start.replace(tzinfo=UTC).isoformat()
            params["end"] = end.replace(tzinfo=UTC).isoformat()
        url = f"{self.base}/products/{self.product_id}/candles"
        async with self.session.get(url, params=params, timeout=20) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"Coinbase candles error {resp.status}: {text}")
            data = await resp.json()
            rows: List[List[Any]] = []
            # Coinbase returns [ time, low, high, open, close, volume ]
            for t, low, high, opn, close, vol in data:
                ot = int(t) * 1000
                ct = ot + 60_000 - 1
                rows.append([ot, opn, high, low, close, vol, ct])
            rows.sort(key=lambda r: r[0])
            return rows

    async def _fetch_chunked(self, start_dt: datetime, end_dt: datetime) -> List[List[Any]]:
        cursor = start_dt
        out: List[List[Any]] = []
        while cursor < end_dt:
            chunk_end = min(cursor + self.max_span, end_dt)
            batch = await self.fetch_candles(cursor, chunk_end)
            out.extend(batch)
            cursor = chunk_end + timedelta(seconds=1)
            await asyncio.sleep(0.15)
        return out

    async def warmup(self):
        self._init_csv()
        df = self.load_csv()
        now_dt = datetime.now(tz=UTC)
        months_back = now_dt - timedelta(days=30*self.cfg.months_history)
        if df.empty:
            self.logger.info("Bootstrapping OHLCV from Coinbase REST...")
            rows_all = await self._fetch_chunked(months_back, now_dt)
            self.save_append(rows_all)
            df = self.load_csv()
        else:
            last_close = int(df.iloc[-1]["close_time"].timestamp() * 1000)
            last_dt = datetime.fromtimestamp(last_close/1000, tz=UTC)
            self.logger.info(f"Incrementally updating from {last_dt}...")
            rows = await self._fetch_chunked(last_dt + timedelta(milliseconds=1), now_dt)
            self.save_append(rows)
            df = self.load_csv()
        self.df = df
        return df

    async def stream_like_minutes(self) -> AsyncIterator[Dict[str, Any]]:
        """
        Polls recent few minutes every 5 seconds, emits a kline-ish dict when a new minute closes.
        """
        assert self.session is not None
        latest_open_ms = None
        while True:
            try:
                now_dt = datetime.now(tz=UTC)
                since_dt = now_dt - timedelta(minutes=5)
                batch = await self.fetch_candles(since_dt, now_dt)
                appended = None
                for r in batch:
                    ot, opn, high, low, close, vol, ct = r
                    if latest_open_ms is None or ot > latest_open_ms:
                        if self.df.empty or ot > int(self.df.iloc[-1]["open_time"].timestamp()*1000):
                            self.save_append([r])
                            s = pd.Series({
                                "open_time": pd.to_datetime(ot, unit="ms", utc=True),
                                "open": float(opn),"high": float(high),"low": float(low),
                                "close": float(close),"volume": float(vol),
                                "close_time": pd.to_datetime(ct, unit="ms", utc=True),
                            })
                            self.df = pd.concat([self.df, s.to_frame().T], ignore_index=True)
                            appended = s
                        latest_open_ms = ot
                if appended is not None:
                    yield {"k": {"x": True, "t": int(appended["open_time"].timestamp()*1000),
                                 "T": int(appended["close_time"].timestamp()*1000),
                                 "o": str(appended["open"]), "h": str(appended["high"]),
                                 "l": str(appended["low"]), "c": str(appended["close"]),
                                 "v": str(appended["volume"]) }}
                await asyncio.sleep(5)
            except Exception as e:
                self.logger.error(f"Coinbase poll error: {e}; retrying in 5s")
                await asyncio.sleep(5)

# =========================
# Strategy
# =========================

class StrategyEngineBase:
    def on_new_candle(self, df: pd.DataFrame) -> Tuple[str, str]:
        raise NotImplementedError
    def update_account(self, *, balance_usd: float, position_qty: float):
        pass

class SMACrossover(StrategyEngineBase):
    def __init__(self, fast: int = 20, slow: int = 50):
        self.fast = fast
        self.slow = slow
        self.balance_usd = 0.0
        self.position_qty = 0.0

    def update_account(self, *, balance_usd: float, position_qty: float):
        self.balance_usd = balance_usd
        self.position_qty = position_qty

    def on_new_candle(self, df: pd.DataFrame) -> Tuple[str, str]:
        if len(df) < max(self.fast, self.slow) + 2:
            return "HOLD", "warming_up"
        closes = df["close"].astype(float)
        sma_f = closes.rolling(self.fast).mean()
        sma_s = closes.rolling(self.slow).mean()
        prev_fast = sma_f.iloc[-2]; prev_slow = sma_s.iloc[-2]
        cur_fast = sma_f.iloc[-1];  cur_slow = sma_s.iloc[-1]
        if math.isnan(prev_fast) or math.isnan(prev_slow):
            return "HOLD", "insufficient_ma"
        if prev_fast <= prev_slow and cur_fast > cur_slow:
            return "BUY", f"SMA{self.fast} crossed above SMA{self.slow}"
        if prev_fast >= prev_slow and cur_fast < cur_slow:
            return "SELL", f"SMA{self.fast} crossed below SMA{self.slow}"
        return "HOLD", "no_cross"

# =========================
# Roostoo Client (hardened)
# =========================

class RoostooClient:
    def __init__(self, cfg: Config, session: aiohttp.ClientSession, logger: logging.Logger,
                 request_timeout: float = 30.0, max_retries: int = 5, retry_backoff: float = 1.6):
        self.cfg = cfg
        self.session = session
        self.logger = logger
        self.sim_balance_usd = 10_000.0
        self.sim_position_qty = 0.0
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

    async def _auth_headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.cfg.roostoo_api_key:
            h["X-API-KEY"] = self.cfg.roostoo_api_key
        return h

    async def _request_json(self, method: str, url: str, **kwargs) -> Dict[str, Any]:
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                async with self.session.request(
                    method, url,
                    headers=await self._auth_headers(),
                    timeout=self.request_timeout,
                    **kwargs
                ) as resp:
                    txt = await resp.text()
                    if 200 <= resp.status < 300:
                        try:
                            return json.loads(txt) if txt else {}
                        except json.JSONDecodeError:
                            return {}
                    if resp.status in (408, 425, 429, 500, 502, 503, 504):
                        delay = (self.retry_backoff ** attempt) + random.uniform(0, 0.25)
                        self.logger.warning(f"Roostoo {method} {url} -> {resp.status}; retrying in {delay:.2f}s")
                        await asyncio.sleep(delay)
                        continue
                    raise RuntimeError(f"Roostoo {method} {url} error {resp.status}: {txt}")
            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                last_err = e
                delay = (self.retry_backoff ** attempt) + random.uniform(0, 0.25)
                self.logger.warning(f"Roostoo {method} {url} client error {e!r}; retrying in {delay:.2f}s")
                await asyncio.sleep(delay)
        if last_err:
            raise last_err
        raise RuntimeError("Roostoo request failed")

    async def get_balance(self) -> Dict[str, float]:
        if self.cfg.dry_run:
            return {"USD": self.sim_balance_usd}
        url = f"{self.cfg.roostoo_base}/v1/balance"
        j = await self._request_json("GET", url)
        return {"USD": float(j.get("USD,USD", j.get("USD", 0.0)))}  # allow both keys if API differs

    async def place_order(self, *, symbol: str, side: str, qty: float, price: float) -> Dict[str, Any]:
        if self.cfg.dry_run:
            if side == "BUY":
                fill = min(self.sim_balance_usd / price, qty)
                self.sim_balance_usd -= fill * price
                self.sim_position_qty += fill
            else:
                fill = min(self.sim_position_qty, qty)
                self.sim_balance_usd += fill * price
                self.sim_position_qty -= fill
            return {
                "orderId": f"SIM-{int(time.time()*1000)}",
                "status": "FILLED",
                "filledQty": fill,
                "avgPrice": price,
                "balanceUSD": self.sim_balance_usd,
                "positionQty": self.sim_position_qty,
            }
        url = f"{self.cfg.roostoo_base}/v1/order"
        payload = {"symbol": symbol, "side": side, "type": "MARKET", "quantity": qty}
        return await self._request_json("POST", url, json=payload)

# =========================
# Trader
# =========================

class Trader:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.stop_event = asyncio.Event()
        self.trade_logger = TradeLogger(cfg.trades_csv)

    async def run(self, strategy: StrategyEngineBase):
        async with CoinbaseDataHandler(self.cfg, self.trade_logger.log) as data:
            await data.warmup()

            async with aiohttp.ClientSession() as sess:
                roostoo = RoostooClient(
                    self.cfg, sess, self.trade_logger.log,
                    request_timeout=self.cfg.roostoo_timeout,
                    max_retries=self.cfg.roostoo_retries,
                )

                # Initial balance (resilient)
                try:
                    bal = await roostoo.get_balance()
                    usd = bal.get("USD", 0.0)
                except Exception as e:
                    self.trade_logger.log.warning(f"Initial balance fetch failed: {e!r}; continuing")
                    usd = 0.0
                strategy.update_account(balance_usd=usd, position_qty=0.0)

                # ===== STARTUP DEMO: Execute $100 BUY → $100 SELL =====
                try:
                    self.trade_logger.log.info("Startup demo: $100 BUY then $100 SELL")
                    last_price = float(data.df.iloc[-1]["close"])
                    demo_notional = 100.0
                    demo_qty = max(demo_notional / last_price, 0.00000001)

                    # BUY
                    order_buy = await roostoo.place_order(
                        symbol=self.cfg.roostoo_symbol or roostoo_symbol_from(self.cfg.symbol),
                        side="BUY",
                        qty=demo_qty,
                        price=last_price,
                    )
                    buy_filled = float(order_buy.get("filledQty", demo_qty))
                    buy_avg_price = float(order_buy.get("avgPrice", last_price))

                    bal_after_buy = await roostoo.get_balance()
                    bal_usd_after_buy = bal_after_buy.get("USD", usd)
                    pos_after_buy = getattr(roostoo, "sim_position_qty", 0.0)

                    self.trade_logger.record(
                        timestamp=datetime.now(UTC),
                        symbol=self.cfg.roostoo_symbol or roostoo_symbol_from(self.cfg.symbol),
                        side="BUY",
                        price=buy_avg_price,
                        qty=buy_filled,
                        reason="startup_demo_buy",
                        balance_before=usd,
                        balance_after=bal_usd_after_buy,
                        position_before=0.0,
                        position_after=pos_after_buy,
                        order_id=str(order_buy.get("orderId", "")),
                        status=order_buy.get("status", "FILLED"),
                    )

                    # SELL immediately
                    order_sell = await roostoo.place_order(
                        symbol=self.cfg.roostoo_symbol or roostoo_symbol_from(self.cfg.symbol),
                        side="SELL",
                        qty=pos_after_buy,
                        price=last_price,
                    )
                    sell_filled = float(order_sell.get("filledQty", pos_after_buy))
                    sell_avg_price = float(order_sell.get("avgPrice", last_price))

                    bal_after_sell = await roostoo.get_balance()
                    bal_usd_after_sell = bal_after_sell.get("USD", bal_usd_after_buy)
                    pos_after_sell = getattr(roostoo, "sim_position_qty", pos_after_buy)

                    self.trade_logger.record(
                        timestamp=datetime.now(UTC),
                        symbol=self.cfg.roostoo_symbol or roostoo_symbol_from(self.cfg.symbol),
                        side="SELL",
                        price=sell_avg_price,
                        qty=sell_filled,
                        reason="startup_demo_sell",
                        balance_before=bal_usd_after_buy,
                        balance_after=bal_usd_after_sell,
                        position_before=pos_after_buy,
                        position_after=pos_after_sell,
                        order_id=str(order_sell.get("orderId", "")),
                        status=order_sell.get("status", "FILLED"),
                    )

                    strategy.update_account(
                        balance_usd=bal_usd_after_sell,
                        position_qty=pos_after_sell
                    )
                    self.trade_logger.log.info("Startup demo completed.")
                except Exception as e:
                    self.trade_logger.log.error(f"Startup demo failed: {e}")
                # ===== END STARTUP DEMO =====

                # Live loop
                async for msg in data.stream_like_minutes():
                    k = msg.get("k", {})
                    if not k:
                        continue

                    signal, reason = strategy.on_new_candle(data.df)
                    last_close = float(data.df.iloc[-1]["close"])
                    self.trade_logger.log.info(f"Signal={signal} reason={reason} close={last_close:.2f}")
                    if signal == "HOLD":
                        if self.stop_event.is_set():
                            break
                        continue

                    # balance before
                    try:
                        bal = await roostoo.get_balance()
                        balance_before = bal.get("USD", 0.0)
                    except Exception as e:
                        self.trade_logger.log.warning(f"Balance fetch failed before order: {e!r}")
                        balance_before = 0.0

                    px = last_close
                    position_before = getattr(roostoo, "sim_position_qty", 0.0)

                    if signal == "BUY":
                        target_notional = min(self.cfg.max_position_usd, balance_before * self.cfg.order_notional_fraction)
                        qty = max(0.0, target_notional / px)
                        side = "BUY"
                    else:
                        qty = max(0.0, getattr(roostoo, "sim_position_qty", 0.0))
                        side = "SELL"

                    if qty <= 0:
                        self.trade_logger.log.info("Qty=0; skipping order")
                        if self.stop_event.is_set():
                            break
                        continue

                    try:
                        order = await roostoo.place_order(
                            symbol=self.cfg.roostoo_symbol or roostoo_symbol_from(self.cfg.symbol),
                            side=side, qty=qty, price=px
                        )
                        status = order.get("status", "UNKNOWN")
                        filled = float(order.get("filledQty", qty))
                        avg_price = float(order.get("avgPrice", px))

                        try:
                            bal_after = await roostoo.get_balance()
                            balance_after = bal_after.get("USD", balance_before)
                        except Exception:
                            balance_after = balance_before

                        position_after = getattr(roostoo, "sim_position_qty", position_before)
                        strategy.update_account(balance_usd=balance_after, position_qty=position_after)

                        self.trade_logger.record(
                            timestamp=datetime.now(UTC),
                            symbol=self.cfg.roostoo_symbol or roostoo_symbol_from(self.cfg.symbol),
                            side=side,
                            price=avg_price,
                            qty=filled,
                            reason=reason,
                            balance_before=balance_before,
                            balance_after=balance_after,
                            position_before=position_before,
                            position_after=position_after,
                            order_id=str(order.get("orderId", "")),
                            status=status
                        )
                    except Exception as e:
                        self.trade_logger.log.error(f"Order failed: {e}")

                    if self.stop_event.is_set():
                        break

    def request_stop(self, *_):
        self.trade_logger.log.info("Stop requested")
        self.stop_event.set()

# =========================
# Entrypoint / CLI
# =========================

def parse_args(argv: Optional[List[str]] = None) -> Config:
    p = argparse.ArgumentParser(description="Live trading orchestrator (Coinbase data)")
    p.add_argument("--symbol", default=os.environ.get("SYMBOL", "BTC-USD"))  # Coinbase format
    p.add_argument("--interval", default=os.environ.get("INTERVAL", "1m"))
    p.add_argument("--months", type=int, default=int(os.environ.get("MONTHS", 3)))
    p.add_argument("--data-csv", default=os.environ.get("DATA_CSV", "data_1m.csv"))
    p.add_argument("--trades-csv", default=os.environ.get("TRADES_CSV", "trades.csv"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-position-usd", type=float, default=float(os.environ.get("MAX_POS_USD", 1000)))
    p.add_argument("--order-fraction", type=float, default=float(os.environ.get("ORDER_FRAC", 0.5)))
    p.add_argument("--roostoo-base", default=os.environ.get("ROOSTOO_BASE", "https://api.roostoo.com"))
    p.add_argument("--roostoo-timeout", type=float, default=float(os.environ.get("ROOSTOO_TIMEOUT", 30)))
    p.add_argument("--roostoo-retries", type=int, default=int(os.environ.get("ROOSTOO_RETRIES", 5)))
    p.add_argument("--roostoo-symbol", default=os.environ.get("ROOSTOO_SYMBOL", ""))  # optional, e.g., BTCUSD
    args = p.parse_args(argv)

    cfg = Config(
        symbol=args.symbol,
        interval=args.interval,
        months_history=args.months,
        data_csv=Path(args.data_csv),
        trades_csv=Path(args.trades_csv),
        dry_run=args.dry_run,
        max_position_usd=args.max_position_usd,
        order_notional_fraction=args.order_fraction,
        roostoo_base=args.roostoo_base,
        roostoo_timeout=args.roostoo_timeout,
        roostoo_retries=args.roostoo_retries,
        roostoo_symbol=args.roostoo_symbol or None,
    )
    cfg.finalize()
    return cfg

async def main_async(cfg: Config):
    trader = Trader(cfg)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, trader.request_stop)
        except NotImplementedError:
            pass
    strategy = SMACrossover(fast=20, slow=50)
    await trader.run(strategy)

def main():
    cfg = parse_args()
    asyncio.run(main_async(cfg))

if __name__ == "__main__":
    main()
