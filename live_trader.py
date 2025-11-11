from __future__ import annotations
import asyncio
import csv
import dataclasses as dc
import json
import logging
import math
import os
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple
import aiohttp
import pandas as pd
from dateutil import tz

@dc.dataclass
class Config:
    symbol: str = "BTCUSDT"
    interval: str = "1m"
    months_history: int = 3
    provider: str = "binance"
    rest_base: str = "https://api.binance.com"
    ws_base: str = "wss://stream.binance.com:9443/ws"
    data_csv: Path = Path("data_1m.csv")
    trades_csv: Path = Path("trades.csv")
    dry_run: bool = False
    roostoo_base: str = os.environ.get("ROOSTOO_BASE", "https://api.roostoo.com")
    roostoo_api_key: Optional[str] = os.environ.get("ROOSTOO_API_KEY")
    roostoo_api_secret: Optional[str] = os.environ.get("ROOSTOO_API_SECRET")
    max_position_usd: float = 1000.0
    order_notional_fraction: float = 0.5

UTC = timezone.utc
LOCAL_TZ = tz.tzlocal()

def utc_ms(dt_: datetime) -> int:
    return int(dt_.replace(tzinfo=UTC).timestamp() * 1000)

def now_ms() -> int:
    return int(time.time() * 1000)

def to_dt(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=UTC)

def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

def to_provider_symbol(provider: str, symbol: str) -> str:
    p = provider.lower()
    if p == "coinbase":
        if "-" in symbol:
            return symbol
        if symbol.endswith("USDT"):
            return symbol[:-4] + "-USD"
        if symbol.endswith("USD"):
            return symbol[:-3] + "-USD"
        return symbol + "-USD"
    return symbol

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
            timestamp.astimezone(UTC).isoformat(), symbol, side, f"{price:.8f}", f"{qty:.8f}", f"{price*qty:.2f}",
            reason, f"{balance_before:.2f}", f"{balance_after:.2f}", f"{position_before:.8f}", f"{position_after:.8f}",
            order_id, status
        ]
        with self.csv_path.open("a", newline="") as f:
            csv.writer(f).writerow(row)
        self.log.info(
            f"TRADE {side} {symbol} qty={qty:.6f} @ {price:.2f} | reason={reason} | bal {balance_before:.2f}->{balance_after:.2f} | pos {position_before:.6f}->{position_after:.6f} | status={status} | oid={order_id}"
        )

class BinanceDataHandler:
    def __init__(self, cfg: Config, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger
        self.df: pd.DataFrame = pd.DataFrame()
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self.session:
            await self.session.close()

    def _init_csv(self):
        ensure_parent(self.cfg.data_csv)
        if not self.cfg.data_csv.exists():
            pd.DataFrame(columns=["open_time","open","high","low","close","volume","close_time"]).to_csv(self.cfg.data_csv, index=False)

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
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
            float_cols = ["open","high","low","close","volume"]
            for c in float_cols:
                df[c] = df[c].astype(float)
            df = df.sort_values("open_time").drop_duplicates(subset=["open_time"], keep="last").reset_index(drop=True)
            self.logger.info(f"Loaded {len(df)} rows from {self.cfg.data_csv}")
            return df
        return pd.DataFrame()

    async def fetch_klines(self, start_ms: int, end_ms: int, limit: int = 1000) -> List[List[Any]]:
        assert self.session is not None
        params = {
            "symbol": self.cfg.symbol,
            "interval": self.cfg.interval,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": limit,
        }
        url = f"{self.cfg.rest_base}/api/v3/klines"
        async with self.session.get(url, params=params, timeout=20) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"REST klines error {resp.status}: {text}")
            data = await resp.json()
            return data

    async def warmup(self):
        self._init_csv()
        df = self.load_csv()
        now_dt = datetime.now(tz=UTC)
        months_back = now_dt - timedelta(days=30*self.cfg.months_history)
        if df.empty:
            self.logger.info("Bootstrapping OHLCV from REST...")
            start = utc_ms(months_back)
            end = now_ms()
            rows_all: List[List[Any]] = []
            cursor = start
            while cursor < end:
                batch = await self.fetch_klines(cursor, min(cursor + 1000*60*60, end))
                if not batch:
                    break
                rows = [
                    [b[0], b[1], b[2], b[3], b[4], b[5], b[6]] for b in batch
                ]
                rows_all.extend(rows)
                cursor = batch[-1][6] + 1
                await asyncio.sleep(0.2)
            self.save_append(rows_all)
            df = self.load_csv()
        else:
            last_close = int(df.iloc[-1]["close_time"].timestamp() * 1000)
            self.logger.info(f"Incrementally updating from {to_dt(last_close)}...")
            end = now_ms()
            if end - last_close > 60_000:
                batch = await self.fetch_klines(last_close + 1, end)
                rows = [
                    [b[0], b[1], b[2], b[3], b[4], b[5], b[6]] for b in batch
                ]
                self.save_append(rows)
                df = self.load_csv()
        self.df = df
        return df

    async def stream_klines(self) -> AsyncIterator[Dict[str, Any]]:
        assert self.session is not None
        stream = f"{self.cfg.symbol.lower()}@kline_{self.cfg.interval}"
        url = f"{self.cfg.ws_base}/{stream}"
        backoff = 1.0
        while True:
            try:
                self.logger.info(f"Connecting WS {url}")
                async with self.session.ws_connect(url, heartbeat=30) as ws:
                    backoff = 1.0
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            payload = json.loads(msg.data)
                            yield payload
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            raise ws.exception()
            except Exception as e:
                self.logger.error(f"WS error: {e}; reconnecting in {backoff:.1f}s")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)

    def apply_closed_kline(self, k: Dict[str, Any]) -> Optional[pd.Series]:
        kd = k.get("k", {})
        if not kd or not kd.get("x"):
            return None
        row = [
            kd["t"], kd["o"], kd["h"], kd["l"], kd["c"], kd["v"], kd["T"],
        ]
        self.save_append([row])
        s = pd.Series({
            "open_time": pd.to_datetime(kd["t"], unit="ms", utc=True),
            "open": float(kd["o"]),
            "high": float(kd["h"]),
            "low": float(kd["l"]),
            "close": float(kd["c"]),
            "volume": float(kd["v"]),
            "close_time": pd.to_datetime(kd["T"], unit="ms", utc=True),
        })
        self.df = pd.concat([self.df, s.to_frame().T], ignore_index=True)
        return s

class CoinbaseDataHandler:
    def __init__(self, cfg: Config, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger
        self.df: pd.DataFrame = pd.DataFrame()
        self.session: Optional[aiohttp.ClientSession] = None
        self.product_id = to_provider_symbol("coinbase", cfg.symbol)
        self.base = "https://api.exchange.coinbase.com"

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self.session:
            await self.session.close()

    def _init_csv(self):
        ensure_parent(self.cfg.data_csv)
        if not self.cfg.data_csv.exists():
            pd.DataFrame(columns=["open_time","open","high","low","close","volume","close_time"]).to_csv(self.cfg.data_csv, index=False)

    def save_append(self, rows):
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

    async def fetch_candles(self, start: Optional[datetime], end: Optional[datetime], granularity: int = 60):
        assert self.session is not None
        params = {"granularity": granularity}
        if start and end:
            params["start"] = start.replace(tzinfo=UTC).isoformat()
            params["end"] = end.replace(tzinfo=UTC).isoformat()
        url = f"{self.base}/products/{self.product_id}/candles"
        async with self.session.get(url, params=params, timeout=20) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"Coinbase candles error {resp.status}: {text}")
            data = await resp.json()
            rows = []
            for t, low, high, opn, close, vol in data:
                ot = int(t) * 1000
                ct = ot + 60_000 - 1
                rows.append([ot, opn, high, low, close, vol, ct])
            rows.sort(key=lambda r: r[0])
            return rows

    async def warmup(self):
        self._init_csv()
        df = self.load_csv()
        now_dt = datetime.now(tz=UTC)
        months_back = now_dt - timedelta(days=30*self.cfg.months_history)
        if df.empty:
            self.logger.info("Bootstrapping OHLCV from Coinbase REST...")
            cursor = months_back
            rows_all = []
            while cursor < now_dt:
                chunk_end = min(cursor + timedelta(days=5), now_dt)
                batch = await self.fetch_candles(cursor, chunk_end, 60)
                rows_all.extend(batch)
                cursor = chunk_end + timedelta(seconds=1)
                await asyncio.sleep(0.2)
            self.save_append(rows_all)
            df = self.load_csv()
        else:
            last_close = int(df.iloc[-1]["close_time"].timestamp() * 1000)
            last_dt = datetime.fromtimestamp(last_close/1000, tz=UTC)
            self.logger.info(f"Incrementally updating from {last_dt}...")
            batch = await self.fetch_candles(last_dt + timedelta(milliseconds=1), now_dt, 60)
            self.save_append(batch)
            df = self.load_csv()
        self.df = df
        return df

    async def stream_like_minutes(self):
        assert self.session is not None
        while True:
            try:
                now_dt = datetime.now(tz=UTC)
                since_dt = now_dt - timedelta(minutes=5)
                batch = await self.fetch_candles(since_dt, now_dt, 60)
                appended = None
                for r in batch:
                    ot, opn, high, low, close, vol, ct = r
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
                if appended is not None:
                    yield {"k": {"x": True, "t": int(appended["open_time"].timestamp()*1000), "T": int(appended["close_time"].timestamp()*1000), "o": str(appended["open"]), "h": str(appended["high"]), "l": str(appended["low"]), "c": str(appended["close"]), "v": str(appended["volume"]) }}
                await asyncio.sleep(5)
            except Exception as e:
                self.logger.error(f"Coinbase poll error: {e}; retrying in 5s")
                await asyncio.sleep(5)

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
        self.last_signal = "HOLD"

    def update_account(self, *, balance_usd: float, position_qty: float):
        self.balance_usd = balance_usd
        self.position_qty = position_qty

    def on_new_candle(self, df: pd.DataFrame) -> Tuple[str, str]:
        if len(df) < max(self.fast, self.slow) + 2:
            return "HOLD", "warming_up"
        closes = df["close"].astype(float)
        sma_f = closes.rolling(self.fast).mean()
        sma_s = closes.rolling(self.slow).mean()
        prev_fast = sma_f.iloc[-2]
        prev_slow = sma_s.iloc[-2]
        cur_fast = sma_f.iloc[-1]
        cur_slow = sma_s.iloc[-1]
        if math.isnan(prev_fast) or math.isnan(prev_slow):
            return "HOLD", "insufficient_ma"
        if prev_fast <= prev_slow and cur_fast > cur_slow:
            return "BUY", f"SMA{self.fast} crossed above SMA{self.slow}"
        if prev_fast >= prev_slow and cur_fast < cur_slow:
            return "SELL", f"SMA{self.fast} crossed below SMA{self.slow}"
        return "HOLD", "no_cross"

class RoostooClient:
    def __init__(self, cfg: Config, session: aiohttp.ClientSession, logger: logging.Logger):
        self.cfg = cfg
        self.session = session
        self.logger = logger
        self.sim_balance_usd = 10_000.0
        self.sim_position_qty = 0.0

    async def _auth_headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.cfg.roostoo_api_key:
            h["X-API-KEY"] = self.cfg.roostoo_api_key
        return h

    async def get_balance(self) -> Dict[str, float]:
        if self.cfg.dry_run:
            return {"USD": self.sim_balance_usd}
        url = f"{self.cfg.roostoo_base}/v1/balance"
        async with self.session.get(url, headers=await self._auth_headers(), timeout=15) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Roostoo balance error {resp.status}: {await resp.text()}")
            j = await resp.json()
            return {"USD": float(j.get("USD", 0.0))}

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
        async with self.session.post(url, headers=await self._auth_headers(), json=payload, timeout=20) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Roostoo order error {resp.status}: {await resp.text()}")
            return await resp.json()

class Trader:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.logger = logging.getLogger("trader")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            self.logger.addHandler(logging.StreamHandler(sys.stdout))
        self.stop_event = asyncio.Event()
        self.trade_logger = TradeLogger(cfg.trades_csv)

    async def run(self, strategy: StrategyEngineBase):
        provider = self.cfg.provider.lower()
        if provider == "coinbase":
            data_ctx = CoinbaseDataHandler(self.cfg, self.trade_logger.log)
        else:
            data_ctx = BinanceDataHandler(self.cfg, self.trade_logger.log)
        async with data_ctx as data:
            await data.warmup()
            async with aiohttp.ClientSession() as sess:
                roostoo = RoostooClient(self.cfg, sess, self.trade_logger.log)
                bal = await roostoo.get_balance()
                strategy.update_account(balance_usd=bal.get("USD", 0.0), position_qty=getattr(roostoo, "sim_position_qty", 0.0))
                if provider == "coinbase":
                    async for msg in data.stream_like_minutes():
                        kd = msg.get("k", {})
                        if not kd: continue
                        appended = data.apply_closed_kline(msg) if hasattr(data, "apply_closed_kline") else None
                        if appended is None and kd.get("x"):
                            appended = pd.Series({
                                "open_time": pd.to_datetime(kd["t"], unit="ms", utc=True),
                                "close_time": pd.to_datetime(kd["T"], unit="ms", utc=True),
                                "open": float(kd["o"]), "high": float(kd["h"]), "low": float(kd["l"]), "close": float(kd["c"]), "volume": float(kd["v"])
                            })
                        signal, reason = strategy.on_new_candle(data.df)
                        self.trade_logger.log.info(f"Signal={signal} reason={reason} close={data.df.iloc[-1]['close']:.2f}")
                        if signal == "HOLD": continue
                        bal = await roostoo.get_balance()
                        balance_before = bal.get("USD", 0.0)
                        position_before = getattr(roostoo, "sim_position_qty", 0.0)
                        px = float(data.df.iloc[-1]["close"])
                        if signal == "BUY":
                            target_notional = min(self.cfg.max_position_usd, balance_before * self.cfg.order_notional_fraction)
                            qty = max(0.0, target_notional / px)
                            side = "BUY"
                        else:
                            qty = max(0.0, getattr(roostoo, "sim_position_qty", 0.0))
                            side = "SELL"
                        if qty <= 0:
                            self.trade_logger.log.info("Qty=0; skipping order")
                            continue
                        try:
                            order = await roostoo.place_order(symbol=to_provider_symbol(provider, self.cfg.symbol), side=side, qty=qty, price=px)
                            status = order.get("status", "UNKNOWN")
                            filled = float(order.get("filledQty", qty))
                            avg_price = float(order.get("avgPrice", px))
                            bal_after = await roostoo.get_balance()
                            balance_after = bal_after.get("USD", balance_before)
                            position_after = getattr(roostoo, "sim_position_qty", position_before)
                            strategy.update_account(balance_usd=balance_after, position_qty=position_after)
                            self.trade_logger.record(
                                timestamp=datetime.now(UTC), symbol=to_provider_symbol(provider, self.cfg.symbol), side=side, price=avg_price, qty=filled,
                                reason=reason, balance_before=balance_before, balance_after=balance_after,
                                position_before=position_before, position_after=position_after,
                                order_id=str(order.get("orderId", "")), status=status
                            )
                        except Exception as e:
                            self.trade_logger.log.error(f"Order failed: {e}")
                        if self.stop_event.is_set(): break
                else:
                    async for msg in data.stream_klines():
                        k = msg.get("k", {})
                        if not k: continue
                        appended = data.apply_closed_kline(msg)
                        if appended is None: continue
                        signal, reason = strategy.on_new_candle(data.df)
                        self.trade_logger.log.info(f"Signal={signal} reason={reason} close={appended['close']:.2f}")
                        if signal == "HOLD": continue
                        bal = await roostoo.get_balance()
                        balance_before = bal.get("USD", 0.0)
                        position_before = getattr(roostoo, "sim_position_qty", 0.0)
                        px = float(appended["close"])
                        if signal == "BUY":
                            target_notional = min(self.cfg.max_position_usd, balance_before * self.cfg.order_notional_fraction)
                            qty = max(0.0, target_notional / px)
                            side = "BUY"
                        else:
                            qty = max(0.0, getattr(roostoo, "sim_position_qty", 0.0))
                            side = "SELL"
                        if qty <= 0:
                            self.trade_logger.log.info("Qty=0; skipping order")
                            continue
                        try:
                            order = await roostoo.place_order(symbol=self.cfg.symbol, side=side, qty=qty, price=px)
                            status = order.get("status", "UNKNOWN")
                            filled = float(order.get("filledQty", qty))
                            avg_price = float(order.get("avgPrice", px))
                            bal_after = await roostoo.get_balance()
                            balance_after = bal_after.get("USD", balance_before)
                            position_after = getattr(roostoo, "sim_position_qty", position_before)
                            strategy.update_account(balance_usd=balance_after, position_qty=position_after)
                            self.trade_logger.record(
                                timestamp=datetime.now(UTC), symbol=self.cfg.symbol, side=side, price=avg_price, qty=filled,
                                reason=reason, balance_before=balance_before, balance_after=balance_after,
                                position_before=position_before, position_after=position_after,
                                order_id=str(order.get("orderId", "")), status=status
                            )
                        except Exception as e:
                            self.trade_logger.log.error(f"Order failed: {e}")
                        if self.stop_event.is_set(): break

    def request_stop(self, *_):
        self.trade_logger.log.info("Stop requested")
        self.stop_event.set()

import argparse

def parse_args(argv: Optional[List[str]] = None) -> Config:
    p = argparse.ArgumentParser(description="Live trading orchestrator")
    p.add_argument("--symbol", default=os.environ.get("SYMBOL", "BTCUSDT"))
    p.add_argument("--interval", default=os.environ.get("INTERVAL", "1m"))
    p.add_argument("--months", type=int, default=int(os.environ.get("MONTHS", 3)))
    p.add_argument("--provider", default=os.environ.get("PROVIDER", "binance"))
    p.add_argument("--data-csv", default=os.environ.get("DATA_CSV", "data_1m.csv"))
    p.add_argument("--trades-csv", default=os.environ.get("TRADES_CSV", "trades.csv"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-position-usd", type=float, default=float(os.environ.get("MAX_POS_USD", 1000)))
    p.add_argument("--order-fraction", type=float, default=float(os.environ.get("ORDER_FRAC", 0.5)))
    p.add_argument("--roostoo-base", default=os.environ.get("ROOSTOO_BASE", "https://api.roostoo.com"))
    args = p.parse_args(argv)
    return Config(
        symbol=args.symbol,
        interval=args.interval,
        months_history=args.months,
        provider=args.provider,
        data_csv=Path(args.data_csv),
        trades_csv=Path(args.trades_csv),
        dry_run=args.dry_run,
        max_position_usd=args.max_position_usd,
        order_notional_fraction=args.order_fraction,
        roostoo_base=args.roostoo_base,
    )

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
