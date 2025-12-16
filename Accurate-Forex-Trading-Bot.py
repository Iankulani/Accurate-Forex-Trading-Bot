"""
===============================================
ACCURATE FOREX TRADING BOT 
Version: 0.0.2
Author: Ian Carter Kulani
E-mail:iancarterkulani@gmail.com
Phone:+265(0)988061969
===============================================
"""

import sys
import os
import json
import time
import datetime
import logging
import threading
import queue
import hashlib
import pickle
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from enum import Enum
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import talib
import telebot
from telebot import types
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog, Menu
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc
plt.style.use('seaborn-v0_8-darkgrid')
warnings.filterwarnings('ignore')

# ==============================================
# CONFIGURATION & CONSTANTS
# ==============================================

class TradingConfig:
    """Global trading configuration"""
    VERSION = "3.0.0"
    APP_NAME = "Forex Trading Bot Pro"
    
    # MT5 Configuration
    MT5_TIMEOUT = 60000
    MT5_RETRY_ATTEMPTS = 3
    MT5_RETRY_DELAY = 2
    
    # Trading Parameters
    DEFAULT_SYMBOL = "EURUSD"
    DEFAULT_TIMEFRAME = mt5.TIMEFRAME_M15
    DEFAULT_VOLUME = 0.1
    MAX_SPREAD = 20
    SLIPPAGE = 3
    
    # Risk Management
    MAX_RISK_PERCENT = 2.0
    MAX_DAILY_LOSS = 5.0
    MAX_POSITIONS = 5
    
    # Technical Indicators
    SAR_STEP = 0.02
    SAR_MAX = 0.2
    STOCH_K = 14
    STOCH_D = 3
    STOCH_SLOW = 3
    RSI_PERIOD = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    
    # Pattern Detection
    MARUBOZU_BODY_RATIO = 0.9
    DOJI_MAX_BODY = 0.1
    HAMMER_RATIO = 2.0
    ENGULFING_RATIO = 1.5
    
    # Machine Learning
    ML_LOOKBACK = 100
    ML_TRAIN_SIZE = 0.7
    ML_RETRAIN_HOURS = 24
    
    # Telegram
    TELEGRAM_POLLING_INTERVAL = 1
    ALERT_COOLDOWN_SECONDS = 60
    
    # GUI
    GUI_UPDATE_INTERVAL_MS = 1000
    GUI_MAX_LOG_LINES = 1000
    CHART_CANDLES_COUNT = 100
    
    # File Paths
    CONFIG_FILE = "config/bot_config.json"
    MODEL_FILE = "models/trading_model.pkl"
    LOG_FILE = "logs/trading_bot.log"
    TELEGRAM_CONFIG = "config/telegram_config.json"
    MT5_CONFIG = "config/mt5_config.json"


class TimeFrame(Enum):
    """MT5 timeframe mapping"""
    M1 = mt5.TIMEFRAME_M1
    M5 = mt5.TIMEFRAME_M5
    M15 = mt5.TIMEFRAME_M15
    M30 = mt5.TIMEFRAME_M30
    H1 = mt5.TIMEFRAME_H1
    H4 = mt5.TIMEFRAME_H4
    D1 = mt5.TIMEFRAME_D1
    W1 = mt5.TIMEFRAME_W1
    MN1 = mt5.TIMEFRAME_MN1


class TradeSignal(Enum):
    """Trading signals"""
    STRONG_BUY = 2
    BUY = 1
    NEUTRAL = 0
    SELL = -1
    STRONG_SELL = -2


class PatternType(Enum):
    """Candlestick pattern types"""
    MARUBOZU_BULLISH = "Marubozu Bullish"
    MARUBOZU_BEARISH = "Marubozu Bearish"
    DOJI = "Doji"
    HAMMER = "Hammer"
    SHOOTING_STAR = "Shooting Star"
    ENGULFING_BULLISH = "Bullish Engulfing"
    ENGULFING_BEARISH = "Bearish Engulfing"
    MORNING_STAR = "Morning Star"
    EVENING_STAR = "Evening Star"
    THREE_WHITE_SOLDIERS = "Three White Soldiers"
    THREE_BLACK_CROWS = "Three Black Crows"


class OrderType(Enum):
    """Order types"""
    BUY = mt5.ORDER_TYPE_BUY
    SELL = mt5.ORDER_TYPE_SELL
    BUY_LIMIT = mt5.ORDER_TYPE_BUY_LIMIT
    SELL_LIMIT = mt5.ORDER_TYPE_SELL_LIMIT
    BUY_STOP = mt5.ORDER_TYPE_BUY_STOP
    SELL_STOP = mt5.ORDER_TYPE_SELL_STOP


# ==============================================
# DATA MODELS
# ==============================================

@dataclass
class Candle:
    """Candlestick data model"""
    timestamp: datetime.datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    spread: int
    
    @property
    def body(self) -> float:
        return abs(self.close - self.open)
    
    @property
    def upper_shadow(self) -> float:
        return self.high - max(self.open, self.close)
    
    @property
    def lower_shadow(self) -> float:
        return min(self.open, self.close) - self.low
    
    @property
    def total_range(self) -> float:
        return self.high - self.low
    
    @property
    def is_bullish(self) -> bool:
        return self.close > self.open
    
    @property
    def is_bearish(self) -> bool:
        return self.close < self.open


@dataclass
class Pattern:
    """Detected pattern model"""
    pattern_type: PatternType
    confidence: float
    candle_index: int
    description: str
    implications: List[str]


@dataclass
class IndicatorValues:
    """Technical indicator values"""
    sar: float
    sar_trend: str  # "bullish" or "bearish"
    stoch_k: float
    stoch_d: float
    stoch_overbought: bool
    stoch_oversold: bool
    rsi: float
    macd: float
    macd_signal: float
    macd_histogram: float
    sma_20: float
    sma_50: float
    sma_200: float
    bollinger_upper: float
    bollinger_lower: float
    atr: float


@dataclass
class TradingSignal:
    """Complete trading signal"""
    symbol: str
    timeframe: TimeFrame
    signal: TradeSignal
    strength: float
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    patterns: List[Pattern]
    indicators: IndicatorValues
    timestamp: datetime.datetime
    recommended_volume: float


@dataclass
class Position:
    """Open position model"""
    ticket: int
    symbol: str
    order_type: OrderType
    volume: float
    entry_price: float
    current_price: float
    stop_loss: float
    take_profit: float
    profit: float
    swap: float
    commission: float
    timestamp: datetime.datetime
    comment: str


@dataclass
class TelegramConfig:
    """Telegram configuration"""
    token: str = ""
    chat_id: str = ""
    enabled: bool = False
    send_alerts: bool = True
    send_errors: bool = True
    send_positions: bool = True


@dataclass
class MT5Config:
    """MT5 configuration"""
    login: int = 0
    password: str = ""
    server: str = ""
    path: str = ""
    timeout: int = TradingConfig.MT5_TIMEOUT


# ==============================================
# LOGGING SYSTEM
# ==============================================

class TradingLogger:
    """Advanced logging system for trading bot"""
    
    def __init__(self):
        self.logger = logging.getLogger('TradingBot')
        self.logger.setLevel(logging.DEBUG)
        
        # Create logs directory
        os.makedirs('logs', exist_ok=True)
        
        # File handler
        file_handler = logging.FileHandler(
            TradingConfig.LOG_FILE,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.message_queue = queue.Queue()
    
    def log(self, level: str, message: str, **kwargs):
        """Log message with additional context"""
        context = ' '.join([f'{k}={v}' for k, v in kwargs.items()])
        full_message = f"{message} | {context}" if context else message
        
        if level == 'debug':
            self.logger.debug(full_message)
        elif level == 'info':
            self.logger.info(full_message)
        elif level == 'warning':
            self.logger.warning(full_message)
        elif level == 'error':
            self.logger.error(full_message)
        elif level == 'critical':
            self.logger.critical(full_message)
        
        self.message_queue.put(f"{level.upper()}: {message}")
    
    def get_log_messages(self, max_messages: int = 10) -> List[str]:
        """Get recent log messages"""
        messages = []
        while not self.message_queue.empty() and len(messages) < max_messages:
            messages.append(self.message_queue.get())
        return messages


# ==============================================
# CONFIGURATION MANAGER
# ==============================================

class ConfigManager:
    """Manages configuration files"""
    
    def __init__(self):
        self.logger = TradingLogger()
        self.config_dir = "config"
        self.models_dir = "models"
        self.ensure_directories()
        self.config = self.load_config()
    
    def ensure_directories(self):
        """Create necessary directories"""
        directories = [
            self.config_dir,
            self.models_dir,
            "logs",
            "data",
            "exports"
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def load_config(self) -> Dict:
        """Load main configuration"""
        config_path = TradingConfig.CONFIG_FILE
        default_config = {
            "symbol": TradingConfig.DEFAULT_SYMBOL,
            "timeframe": "M15",
            "volume": TradingConfig.DEFAULT_VOLUME,
            "max_spread": TradingConfig.MAX_SPREAD,
            "max_risk_percent": TradingConfig.MAX_RISK_PERCENT,
            "max_daily_loss": TradingConfig.MAX_DAILY_LOSS,
            "max_positions": TradingConfig.MAX_POSITIONS,
            "telegram_enabled": False,
            "auto_trading": False,
            "risk_free_mode": True,
            "ml_enabled": True
        }
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    # Merge with default config for missing keys
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                    return config
            else:
                self.save_config(default_config)
                return default_config
        except Exception as e:
            self.logger.log('error', f"Failed to load config: {str(e)}")
            return default_config
    
    def save_config(self, config: Dict):
        """Save configuration"""
        try:
            with open(TradingConfig.CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=4)
            self.config = config
            self.logger.log('info', "Configuration saved successfully")
            return True
        except Exception as e:
            self.logger.log('error', f"Failed to save config: {str(e)}")
            return False
    
    def get_telegram_config(self) -> TelegramConfig:
        """Load Telegram configuration"""
        config_path = TradingConfig.TELEGRAM_CONFIG
        default_config = {
            "token": "",
            "chat_id": "",
            "enabled": False,
            "send_alerts": True,
            "send_errors": True,
            "send_positions": True
        }
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    return TelegramConfig(**config)
            else:
                self.save_telegram_config(TelegramConfig())
                return TelegramConfig()
        except Exception as e:
            self.logger.log('error', f"Failed to load Telegram config: {str(e)}")
            return TelegramConfig()
    
    def save_telegram_config(self, config: TelegramConfig):
        """Save Telegram configuration"""
        try:
            config_dict = {
                "token": config.token,
                "chat_id": config.chat_id,
                "enabled": config.enabled,
                "send_alerts": config.send_alerts,
                "send_errors": config.send_errors,
                "send_positions": config.send_positions
            }
            with open(TradingConfig.TELEGRAM_CONFIG, 'w') as f:
                json.dump(config_dict, f, indent=4)
            self.logger.log('info', "Telegram configuration saved")
            return True
        except Exception as e:
            self.logger.log('error', f"Failed to save Telegram config: {str(e)}")
            return False
    
    def get_mt5_config(self) -> MT5Config:
        """Load MT5 configuration"""
        config_path = TradingConfig.MT5_CONFIG
        default_config = {
            "login": 0,
            "password": "",
            "server": "",
            "path": "",
            "timeout": TradingConfig.MT5_TIMEOUT
        }
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    return MT5Config(**config)
            else:
                self.save_mt5_config(MT5Config())
                return MT5Config()
        except Exception as e:
            self.logger.log('error', f"Failed to load MT5 config: {str(e)}")
            return MT5Config()
    
    def save_mt5_config(self, config: MT5Config):
        """Save MT5 configuration"""
        try:
            config_dict = {
                "login": config.login,
                "password": config.password,
                "server": config.server,
                "path": config.path,
                "timeout": config.timeout
            }
            with open(TradingConfig.MT5_CONFIG, 'w') as f:
                json.dump(config_dict, f, indent=4)
            self.logger.log('info', "MT5 configuration saved")
            return True
        except Exception as e:
            self.logger.log('error', f"Failed to save MT5 config: {str(e)}")
            return False


# ==============================================
# MT5 CONNECTION MANAGER
# ==============================================

class MT5Manager:
    """Manages MetaTrader 5 connection and operations"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.logger = TradingLogger()
        self.mt5_config = config_manager.get_mt5_config()
        self.connected = False
        self.account_info = None
        
    def connect(self) -> bool:
        """Connect to MT5 terminal"""
        try:
            if not mt5.initialize(
                path=self.mt5_config.path,
                login=self.mt5_config.login,
                password=self.mt5_config.password,
                server=self.mt5_config.server,
                timeout=self.mt5_config.timeout
            ):
                error = mt5.last_error()
                self.logger.log('error', f"MT5 initialization failed: {error}")
                return False
            
            self.connected = True
            self.account_info = mt5.account_info()
            
            self.logger.log('info', 
                f"Connected to MT5 | "
                f"Account: {self.account_info.login} | "
                f"Balance: {self.account_info.balance} | "
                f"Leverage: 1:{self.account_info.leverage}"
            )
            return True
            
        except Exception as e:
            self.logger.log('error', f"MT5 connection error: {str(e)}")
            return False
    
    def disconnect(self):
        """Disconnect from MT5"""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            self.logger.log('info', "Disconnected from MT5")
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get symbol information"""
        try:
            info = mt5.symbol_info(symbol)
            if info is None:
                self.logger.log('error', f"Symbol {symbol} not found")
                return None
            
            return {
                'name': symbol,
                'bid': info.bid,
                'ask': info.ask,
                'spread': info.spread,
                'digits': info.digits,
                'point': info.point,
                'trade_mode': info.trade_mode,
                'trade_contract_size': info.trade_contract_size,
                'trade_mode': info.trade_mode,
                'swap_mode': info.swap_mode,
                'swap_long': info.swap_long,
                'swap_short': info.swap_short,
                'margin_initial': info.margin_initial,
                'margin_maintenance': info.margin_maintenance
            }
        except Exception as e:
            self.logger.log('error', f"Failed to get symbol info: {str(e)}")
            return None
    
    def get_historical_data(self, symbol: str, timeframe: TimeFrame, 
                          count: int = 500) -> Optional[List[Candle]]:
        """Get historical candlestick data"""
        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe.value, 0, count)
            if rates is None:
                self.logger.log('error', f"No historical data for {symbol}")
                return None
            
            candles = []
            for rate in rates:
                candle = Candle(
                    timestamp=datetime.datetime.fromtimestamp(rate['time']),
                    open=rate['open'],
                    high=rate['high'],
                    low=rate['low'],
                    close=rate['close'],
                    volume=rate['tick_volume'],
                    spread=rate['spread']
                )
                candles.append(candle)
            
            self.logger.log('debug', 
                f"Retrieved {len(candles)} candles for {symbol} "
                f"on timeframe {timeframe.name}"
            )
            return candles
            
        except Exception as e:
            self.logger.log('error', f"Failed to get historical data: {str(e)}")
            return None
    
    def get_current_price(self, symbol: str) -> Optional[Dict]:
        """Get current bid/ask prices"""
        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return None
            
            return {
                'bid': tick.bid,
                'ask': tick.ask,
                'last': tick.last,
                'volume': tick.volume,
                'time': datetime.datetime.fromtimestamp(tick.time)
            }
        except Exception as e:
            self.logger.log('error', f"Failed to get current price: {str(e)}")
            return None
    
    def calculate_position_size(self, symbol: str, risk_percent: float, 
                              stop_loss_pips: float) -> float:
        """Calculate position size based on risk"""
        try:
            account_balance = self.account_info.balance
            risk_amount = account_balance * (risk_percent / 100)
            
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return TradingConfig.DEFAULT_VOLUME
            
            pip_value = symbol_info.trade_tick_value * 10
            if symbol_info.digits == 3 or symbol_info.digits == 5:
                pip_value = symbol_info.trade_tick_value
            
            position_size = risk_amount / (stop_loss_pips * pip_value)
            
            # Normalize to lot size
            min_lot = symbol_info.volume_min
            max_lot = symbol_info.volume_max
            lot_step = symbol_info.volume_step
            
            position_size = max(min_lot, min(position_size, max_lot))
            position_size = round(position_size / lot_step) * lot_step
            
            return position_size
            
        except Exception as e:
            self.logger.log('error', f"Failed to calculate position size: {str(e)}")
            return TradingConfig.DEFAULT_VOLUME
    
    def place_order(self, symbol: str, order_type: OrderType, volume: float,
                   stop_loss: float = 0.0, take_profit: float = 0.0,
                   comment: str = "") -> Optional[int]:
        """Place a trade order"""
        try:
            price = mt5.symbol_info_tick(symbol).ask if order_type == OrderType.BUY \
                    else mt5.symbol_info_tick(symbol).bid
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": order_type.value,
                "price": price,
                "sl": stop_loss,
                "tp": take_profit,
                "deviation": TradingConfig.SLIPPAGE,
                "magic": 234000,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.log('error', 
                    f"Order failed: {result.retcode} - {result.comment}"
                )
                return None
            
            self.logger.log('info',
                f"Order placed: {order_type.name} {volume} {symbol} | "
                f"Price: {price} | SL: {stop_loss} | TP: {take_profit}"
            )
            return result.order
            
        except Exception as e:
            self.logger.log('error', f"Failed to place order: {str(e)}")
            return None
    
    def get_open_positions(self) -> List[Position]:
        """Get all open positions"""
        try:
            positions = mt5.positions_get()
            if positions is None:
                return []
            
            open_positions = []
            for position in positions:
                pos = Position(
                    ticket=position.ticket,
                    symbol=position.symbol,
                    order_type=OrderType(position.type),
                    volume=position.volume,
                    entry_price=position.price_open,
                    current_price=position.price_current,
                    stop_loss=position.sl,
                    take_profit=position.tp,
                    profit=position.profit,
                    swap=position.swap,
                    commission=position.commission,
                    timestamp=datetime.datetime.fromtimestamp(position.time),
                    comment=position.comment
                )
                open_positions.append(pos)
            
            return open_positions
            
        except Exception as e:
            self.logger.log('error', f"Failed to get positions: {str(e)}")
            return []
    
    def close_position(self, ticket: int) -> bool:
        """Close a position by ticket"""
        try:
            positions = mt5.positions_get(ticket=ticket)
            if not positions:
                self.logger.log('error', f"Position {ticket} not found")
                return False
            
            position = positions[0]
            order_type = OrderType.BUY if position.type == mt5.ORDER_TYPE_SELL \
                        else OrderType.SELL
            price = mt5.symbol_info_tick(position.symbol).bid if order_type == OrderType.BUY \
                    else mt5.symbol_info_tick(position.symbol).ask
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": order_type.value,
                "position": position.ticket,
                "price": price,
                "deviation": TradingConfig.SLIPPAGE,
                "magic": 234000,
                "comment": "Closed by bot",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.log('error',
                    f"Close order failed: {result.retcode} - {result.comment}"
                )
                return False
            
            self.logger.log('info',
                f"Position closed: Ticket {ticket} | "
                f"Profit: {position.profit}"
            )
            return True
            
        except Exception as e:
            self.logger.log('error', f"Failed to close position: {str(e)}")
            return False


# ==============================================
# PATTERN DETECTOR
# ==============================================

class PatternDetector:
    """Detects candlestick patterns"""
    
    def __init__(self):
        self.logger = TradingLogger()
    
    def detect_all_patterns(self, candles: List[Candle]) -> List[Pattern]:
        """Detect all candlestick patterns"""
        patterns = []
        
        for i in range(len(candles)):
            # Skip if we don't have enough candles for multi-candle patterns
            if i < 2:
                continue
            
            # Single candle patterns
            patterns.extend(self.detect_single_candle_patterns(candles[i]))
            
            # Two candle patterns
            if i >= 1:
                patterns.extend(self.detect_two_candle_patterns(candles[i-1], candles[i]))
            
            # Three candle patterns
            if i >= 2:
                patterns.extend(
                    self.detect_three_candle_patterns(candles[i-2], candles[i-1], candles[i])
                )
        
        return patterns
    
    def detect_single_candle_patterns(self, candle: Candle) -> List[Pattern]:
        """Detect single candle patterns"""
        patterns = []
        
        # Marubozu Pattern
        marubozu_pattern = self.detect_marubozu(candle)
        if marubozu_pattern:
            patterns.append(marubozu_pattern)
        
        # Doji Pattern
        doji_pattern = self.detect_doji(candle)
        if doji_pattern:
            patterns.append(doji_pattern)
        
        # Hammer Pattern
        hammer_pattern = self.detect_hammer(candle)
        if hammer_pattern:
            patterns.append(hammer_pattern)
        
        # Shooting Star Pattern
        shooting_star_pattern = self.detect_shooting_star(candle)
        if shooting_star_pattern:
            patterns.append(shooting_star_pattern)
        
        return patterns
    
    def detect_marubozu(self, candle: Candle) -> Optional[Pattern]:
        """Detect Marubozu pattern"""
        if candle.total_range == 0:
            return None
        
        body_ratio = candle.body / candle.total_range
        
        if body_ratio >= TradingConfig.MARUBOZU_BODY_RATIO:
            if candle.is_bullish:
                return Pattern(
                    pattern_type=PatternType.MARUBOZU_BULLISH,
                    confidence=min(0.9, body_ratio),
                    candle_index=-1,
                    description="Bullish Marubozu: Strong buying pressure",
                    implications=[
                        "Strong bullish momentum",
                        "Potential continuation or reversal",
                        "Consider entering long position"
                    ]
                )
            else:
                return Pattern(
                    pattern_type=PatternType.MARUBOZU_BEARISH,
                    confidence=min(0.9, body_ratio),
                    candle_index=-1,
                    description="Bearish Marubozu: Strong selling pressure",
                    implications=[
                        "Strong bearish momentum",
                        "Potential continuation or reversal",
                        "Consider entering short position"
                    ]
                )
        return None
    
    def detect_doji(self, candle: Candle) -> Optional[Pattern]:
        """Detect Doji pattern"""
        if candle.total_range == 0:
            return None
        
        body_ratio = candle.body / candle.total_range
        
        if body_ratio <= TradingConfig.DOJI_MAX_BODY:
            return Pattern(
                pattern_type=PatternType.DOJI,
                confidence=1.0 - body_ratio,
                candle_index=-1,
                description="Doji: Market indecision",
                implications=[
                    "Potential trend reversal",
                    "Market indecision",
                    "Wait for confirmation"
                ]
            )
        return None
    
    def detect_hammer(self, candle: Candle) -> Optional[Pattern]:
        """Detect Hammer pattern"""
        if candle.total_range == 0:
            return None
        
        body_ratio = candle.body / candle.total_range
        lower_shadow_ratio = candle.lower_shadow / candle.total_range
        upper_shadow_ratio = candle.upper_shadow / candle.total_range
        
        # Hammer criteria
        is_small_body = body_ratio <= 0.3
        is_long_lower_shadow = lower_shadow_ratio >= 0.6
        is_small_upper_shadow = upper_shadow_ratio <= 0.1
        
        if is_small_body and is_long_lower_shadow and is_small_upper_shadow:
            if candle.is_bullish:
                return Pattern(
                    pattern_type=PatternType.HAMMER,
                    confidence=0.7,
                    candle_index=-1,
                    description="Hammer: Potential bullish reversal",
                    implications=[
                        "Bullish reversal signal",
                        "Appears at bottom of downtrend",
                        "Consider long entry with confirmation"
                    ]
                )
        
        # Shooting Star (inverse hammer)
        if is_small_body and is_small_upper_shadow and lower_shadow_ratio >= 0.6:
            if candle.is_bearish:
                return Pattern(
                    pattern_type=PatternType.SHOOTING_STAR,
                    confidence=0.7,
                    candle_index=-1,
                    description="Shooting Star: Potential bearish reversal",
                    implications=[
                        "Bearish reversal signal",
                        "Appears at top of uptrend",
                        "Consider short entry with confirmation"
                    ]
                )
        
        return None
    
    def detect_shooting_star(self, candle: Candle) -> Optional[Pattern]:
        """Detect Shooting Star pattern"""
        # Already covered in detect_hammer
        return None
    
    def detect_two_candle_patterns(self, first: Candle, second: Candle) -> List[Pattern]:
        """Detect two-candle patterns"""
        patterns = []
        
        # Engulfing Pattern
        engulfing_pattern = self.detect_engulfing(first, second)
        if engulfing_pattern:
            patterns.append(engulfing_pattern)
        
        return patterns
    
    def detect_engulfing(self, first: Candle, second: Candle) -> Optional[Pattern]:
        """Detect Engulfing pattern"""
        # Bullish Engulfing
        if first.is_bearish and second.is_bullish:
            if (second.open < first.close and 
                second.close > first.open):
                body_ratio = second.body / first.body
                if body_ratio >= TradingConfig.ENGULFING_RATIO:
                    return Pattern(
                        pattern_type=PatternType.ENGULFING_BULLISH,
                        confidence=min(0.8, body_ratio / 2),
                        candle_index=-2,
                        description="Bullish Engulfing: Strong reversal signal",
                        implications=[
                            "Strong bullish reversal",
                            "Buying pressure overcomes selling",
                            "Consider entering long position"
                        ]
                    )
        
        # Bearish Engulfing
        if first.is_bullish and second.is_bearish:
            if (second.open > first.close and 
                second.close < first.open):
                body_ratio = second.body / first.body
                if body_ratio >= TradingConfig.ENGULFING_RATIO:
                    return Pattern(
                        pattern_type=PatternType.ENGULFING_BEARISH,
                        confidence=min(0.8, body_ratio / 2),
                        candle_index=-2,
                        description="Bearish Engulfing: Strong reversal signal",
                        implications=[
                            "Strong bearish reversal",
                            "Selling pressure overcomes buying",
                            "Consider entering short position"
                        ]
                    )
        
        return None
    
    def detect_three_candle_patterns(self, first: Candle, second: Candle, 
                                   third: Candle) -> List[Pattern]:
        """Detect three-candle patterns"""
        patterns = []
        
        # Morning Star
        morning_star = self.detect_morning_star(first, second, third)
        if morning_star:
            patterns.append(morning_star)
        
        # Evening Star
        evening_star = self.detect_evening_star(first, second, third)
        if evening_star:
            patterns.append(evening_star)
        
        return patterns
    
    def detect_morning_star(self, first: Candle, second: Candle, 
                          third: Candle) -> Optional[Pattern]:
        """Detect Morning Star pattern"""
        # First candle is bearish
        # Second candle is small body (doji or spinning top)
        # Third candle is bullish and closes into first candle's body
        if (first.is_bearish and 
            second.body < first.body * 0.5 and
            third.is_bullish and
            third.close > (first.open + first.close) / 2):
            
            return Pattern(
                pattern_type=PatternType.MORNING_STAR,
                confidence=0.75,
                candle_index=-3,
                description="Morning Star: Strong bullish reversal",
                implications=[
                    "Major bullish reversal pattern",
                    "End of downtrend likely",
                    "Strong buy signal"
                ]
            )
        return None
    
    def detect_evening_star(self, first: Candle, second: Candle, 
                          third: Candle) -> Optional[Pattern]:
        """Detect Evening Star pattern"""
        # First candle is bullish
        # Second candle is small body
        # Third candle is bearish and closes into first candle's body
        if (first.is_bullish and 
            second.body < first.body * 0.5 and
            third.is_bearish and
            third.close < (first.open + first.close) / 2):
            
            return Pattern(
                pattern_type=PatternType.EVENING_STAR,
                confidence=0.75,
                candle_index=-3,
                description="Evening Star: Strong bearish reversal",
                implications=[
                    "Major bearish reversal pattern",
                    "End of uptrend likely",
                    "Strong sell signal"
                ]
            )
        return None


# ==============================================
# TECHNICAL INDICATORS
# ==============================================

class TechnicalIndicators:
    """Calculates technical indicators"""
    
    def __init__(self):
        self.logger = TradingLogger()
    
    def calculate_all_indicators(self, candles: List[Candle]) -> IndicatorValues:
        """Calculate all technical indicators"""
        try:
            closes = np.array([c.close for c in candles])
            highs = np.array([c.high for c in candles])
            lows = np.array([c.low for c in candles])
            
            # Parabolic SAR
            sar = talib.SAR(highs, lows, 
                           acceleration=TradingConfig.SAR_STEP,
                           maximum=TradingConfig.SAR_MAX)
            current_sar = sar[-1] if not np.isnan(sar[-1]) else closes[-1]
            sar_trend = "bullish" if closes[-1] > current_sar else "bearish"
            
            # Stochastic
            stoch_k, stoch_d = talib.STOCH(
                highs, lows, closes,
                fastk_period=TradingConfig.STOCH_K,
                slowk_period=TradingConfig.STOCH_D,
                slowd_period=TradingConfig.STOCH_SLOW
            )
            current_stoch_k = stoch_k[-1] if not np.isnan(stoch_k[-1]) else 50
            current_stoch_d = stoch_d[-1] if not np.isnan(stoch_d[-1]) else 50
            
            # RSI
            rsi = talib.RSI(closes, timeperiod=TradingConfig.RSI_PERIOD)
            current_rsi = rsi[-1] if not np.isnan(rsi[-1]) else 50
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(
                closes,
                fastperiod=TradingConfig.MACD_FAST,
                slowperiod=TradingConfig.MACD_SLOW,
                signalperiod=TradingConfig.MACD_SIGNAL
            )
            current_macd = macd[-1] if not np.isnan(macd[-1]) else 0
            current_macd_signal = macd_signal[-1] if not np.isnan(macd_signal[-1]) else 0
            current_macd_hist = macd_hist[-1] if not np.isnan(macd_hist[-1]) else 0
            
            # Moving Averages
            sma_20 = talib.SMA(closes, timeperiod=20)
            sma_50 = talib.SMA(closes, timeperiod=50)
            sma_200 = talib.SMA(closes, timeperiod=200)
            
            current_sma_20 = sma_20[-1] if not np.isnan(sma_20[-1]) else closes[-1]
            current_sma_50 = sma_50[-1] if not np.isnan(sma_50[-1]) else closes[-1]
            current_sma_200 = sma_200[-1] if not np.isnan(sma_200[-1]) else closes[-1]
            
            # Bollinger Bands
            upper_bb, middle_bb, lower_bb = talib.BBANDS(
                closes, timeperiod=20, nbdevup=2, nbdevdn=2
            )
            current_upper_bb = upper_bb[-1] if not np.isnan(upper_bb[-1]) else closes[-1]
            current_lower_bb = lower_bb[-1] if not np.isnan(lower_bb[-1]) else closes[-1]
            
            # ATR
            atr = talib.ATR(highs, lows, closes, timeperiod=14)
            current_atr = atr[-1] if not np.isnan(atr[-1]) else 0
            
            return IndicatorValues(
                sar=current_sar,
                sar_trend=sar_trend,
                stoch_k=current_stoch_k,
                stoch_d=current_stoch_d,
                stoch_overbought=current_stoch_k > 80,
                stoch_oversold=current_stoch_k < 20,
                rsi=current_rsi,
                macd=current_macd,
                macd_signal=current_macd_signal,
                macd_histogram=current_macd_hist,
                sma_20=current_sma_20,
                sma_50=current_sma_50,
                sma_200=current_sma_200,
                bollinger_upper=current_upper_bb,
                bollinger_lower=current_lower_bb,
                atr=current_atr
            )
            
        except Exception as e:
            self.logger.log('error', f"Failed to calculate indicators: {str(e)}")
            # Return default values
            return IndicatorValues(
                sar=closes[-1],
                sar_trend="neutral",
                stoch_k=50,
                stoch_d=50,
                stoch_overbought=False,
                stoch_oversold=False,
                rsi=50,
                macd=0,
                macd_signal=0,
                macd_histogram=0,
                sma_20=closes[-1],
                sma_50=closes[-1],
                sma_200=closes[-1],
                bollinger_upper=closes[-1],
                bollinger_lower=closes[-1],
                atr=0
            )
    
    def calculate_trend_strength(self, candles: List[Candle]) -> float:
        """Calculate trend strength (0-100)"""
        try:
            closes = np.array([c.close for c in candles])
            
            # Calculate multiple trend indicators
            sma_20 = talib.SMA(closes, timeperiod=20)
            sma_50 = talib.SMA(closes, timeperiod=50)
            
            # ADX for trend strength
            adx = talib.ADX(
                np.array([c.high for c in candles]),
                np.array([c.low for c in candles]),
                closes,
                timeperiod=14
            )
            
            # Current ADX value
            current_adx = adx[-1] if not np.isnan(adx[-1]) else 0
            
            # Normalize to 0-100
            trend_strength = min(100, current_adx * 2)
            
            return trend_strength
            
        except Exception as e:
            self.logger.log('error', f"Failed to calculate trend strength: {str(e)}")
            return 0.0


# ==============================================
# SIGNAL GENERATOR
# ==============================================

class SignalGenerator:
    """Generates trading signals based on patterns and indicators"""
    
    def __init__(self, pattern_detector: PatternDetector, indicators_calculator: TechnicalIndicators):
        self.pattern_detector = pattern_detector
        self.indicators_calculator = indicators_calculator
        self.logger = TradingLogger()
    
    def generate_signal(self, symbol: str, timeframe: TimeFrame, 
                       candles: List[Candle]) -> TradingSignal:
        """Generate complete trading signal"""
        try:
            if len(candles) < 50:
                self.logger.log('warning', 
                    f"Insufficient candles for signal generation: {len(candles)}"
                )
                return self._neutral_signal(symbol, timeframe, candles)
            
            # Detect patterns
            patterns = self.pattern_detector.detect_all_patterns(candles[-5:])
            
            # Calculate indicators
            indicators = self.indicators_calculator.calculate_all_indicators(candles)
            
            # Generate signal score
            signal_score = self._calculate_signal_score(
                patterns, indicators, candles
            )
            
            # Determine signal type
            signal, strength = self._score_to_signal(signal_score)
            
            # Calculate entry levels
            current_price = candles[-1].close
            stop_loss, take_profit = self._calculate_levels(
                signal, current_price, indicators.atr, candles
            )
            
            # Calculate confidence
            confidence = self._calculate_confidence(
                patterns, indicators
            )
            
            # Calculate recommended volume (simplified)
            recommended_volume = TradingConfig.DEFAULT_VOLUME
            
            return TradingSignal(
                symbol=symbol,
                timeframe=timeframe,
                signal=signal,
                strength=strength,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=confidence,
                patterns=patterns,
                indicators=indicators,
                timestamp=datetime.datetime.now(),
                recommended_volume=recommended_volume
            )
            
        except Exception as e:
            self.logger.log('error', f"Failed to generate signal: {str(e)}")
            return self._neutral_signal(symbol, timeframe, candles)
    
    def _calculate_signal_score(self, patterns: List[Pattern], 
                               indicators: IndicatorValues,
                               candles: List[Candle]) -> float:
        """Calculate signal score from -100 to 100"""
        score = 0.0
        
        # Pattern scoring
        pattern_score = 0.0
        for pattern in patterns:
            if pattern.pattern_type in [
                PatternType.MARUBOZU_BULLISH,
                PatternType.ENGULFING_BULLISH,
                PatternType.MORNING_STAR,
                PatternType.HAMMER
            ]:
                pattern_score += pattern.confidence * 20
            elif pattern.pattern_type in [
                PatternType.MARUBOZU_BEARISH,
                PatternType.ENGULFING_BEARISH,
                PatternType.EVENING_STAR,
                PatternType.SHOOTING_STAR
            ]:
                pattern_score -= pattern.confidence * 20
        
        score += pattern_score
        
        # Indicator scoring
        # RSI
        if indicators.rsi > 70:
            score -= 15
        elif indicators.rsi < 30:
            score += 15
        
        # Stochastic
        if indicators.stoch_overbought:
            score -= 10
        elif indicators.stoch_oversold:
            score += 10
        
        # MACD
        if indicators.macd > indicators.macd_signal:
            score += 10
        else:
            score -= 10
        
        # SAR trend
        if indicators.sar_trend == "bullish":
            score += 5
        else:
            score -= 5
        
        # Moving averages alignment
        if (indicators.sma_20 > indicators.sma_50 > indicators.sma_200 and
            candles[-1].close > indicators.sma_20):
            score += 20
        elif (indicators.sma_20 < indicators.sma_50 < indicators.sma_200 and
              candles[-1].close < indicators.sma_20):
            score -= 20
        
        # Normalize score
        return max(-100, min(100, score))
    
    def _score_to_signal(self, score: float) -> Tuple[TradeSignal, float]:
        """Convert score to signal type and strength"""
        strength = abs(score) / 100
        
        if score >= 50:
            return TradeSignal.STRONG_BUY, strength
        elif score >= 20:
            return TradeSignal.BUY, strength
        elif score <= -50:
            return TradeSignal.STRONG_SELL, strength
        elif score <= -20:
            return TradeSignal.SELL, strength
        else:
            return TradeSignal.NEUTRAL, strength
    
    def _calculate_levels(self, signal: TradeSignal, current_price: float,
                         atr: float, candles: List[Candle]) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels"""
        # Use ATR for dynamic levels
        if atr == 0:
            atr = np.mean([c.high - c.low for c in candles[-14:]])
        
        if signal in [TradeSignal.BUY, TradeSignal.STRONG_BUY]:
            stop_loss = current_price - (atr * 1.5)
            take_profit = current_price + (atr * 2.5)
        elif signal in [TradeSignal.SELL, TradeSignal.STRONG_SELL]:
            stop_loss = current_price + (atr * 1.5)
            take_profit = current_price - (atr * 2.5)
        else:
            stop_loss = 0.0
            take_profit = 0.0
        
        return stop_loss, take_profit
    
    def _calculate_confidence(self, patterns: List[Pattern],
                            indicators: IndicatorValues) -> float:
        """Calculate overall confidence score"""
        confidence_scores = []
        
        # Pattern confidence
        if patterns:
            pattern_confidence = np.mean([p.confidence for p in patterns])
            confidence_scores.append(pattern_confidence)
        
        # Indicator alignment confidence
        indicator_confidence = 0.0
        bullish_signals = 0
        total_signals = 0
        
        if indicators.rsi < 30:
            bullish_signals += 1
        elif indicators.rsi > 70:
            bullish_signals -= 1
        total_signals += 1
        
        if indicators.stoch_oversold:
            bullish_signals += 1
        elif indicators.stoch_overbought:
            bullish_signals -= 1
        total_signals += 1
        
        if indicators.macd > indicators.macd_signal:
            bullish_signals += 1
        else:
            bullish_signals -= 1
        total_signals += 1
        
        indicator_confidence = abs(bullish_signals) / total_signals
        confidence_scores.append(indicator_confidence)
        
        # Average all confidence scores
        return np.mean(confidence_scores) if confidence_scores else 0.5
    
    def _neutral_signal(self, symbol: str, timeframe: TimeFrame,
                       candles: List[Candle]) -> TradingSignal:
        """Create neutral signal when analysis fails"""
        return TradingSignal(
            symbol=symbol,
            timeframe=timeframe,
            signal=TradeSignal.NEUTRAL,
            strength=0.0,
            entry_price=candles[-1].close if candles else 0.0,
            stop_loss=0.0,
            take_profit=0.0,
            confidence=0.0,
            patterns=[],
            indicators=self.indicators_calculator.calculate_all_indicators(candles),
            timestamp=datetime.datetime.now(),
            recommended_volume=0.0
        )


# ==============================================
# TELEGRAM BOT MANAGER
# ==============================================

class TelegramManager:
    """Manages Telegram bot for alerts and notifications"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.logger = TradingLogger()
        self.bot = None
        self.chat_id = None
        self.enabled = False
        self.last_alert_time = {}
        
        self.load_config()
    
    def load_config(self):
        """Load Telegram configuration"""
        config = self.config_manager.get_telegram_config()
        self.token = config.token
        self.chat_id = config.chat_id
        self.enabled = config.enabled
        self.send_alerts = config.send_alerts
        self.send_errors = config.send_errors
        self.send_positions = config.send_positions
    
    def initialize(self) -> bool:
        """Initialize Telegram bot"""
        try:
            if not self.enabled or not self.token:
                self.logger.log('info', "Telegram bot disabled or no token")
                return False
            
            self.bot = telebot.TeleBot(self.token)
            self.logger.log('info', "Telegram bot initialized")
            
            # Setup command handlers
            @self.bot.message_handler(commands=['start'])
            def send_welcome(message):
                self.bot.reply_to(message, 
                    f"Welcome to Forex Trading Bot!\n"
                    f"Available commands:\n"
                    f"/status - Check bot status\n"
                    f"/positions - Show open positions\n"
                    f"/signal - Get current signal\n"
                    f"/help - Show all commands"
                )
            
            @self.bot.message_handler(commands=['status'])
            def send_status(message):
                self.bot.reply_to(message, "Bot is running and monitoring markets.")
            
            @self.bot.message_handler(commands=['signal'])
            def send_signal(message):
                self.bot.reply_to(message, "Generating current signal...")
            
            self.start_polling()
            return True
            
        except Exception as e:
            self.logger.log('error', f"Failed to initialize Telegram bot: {str(e)}")
            return False
    
    def start_polling(self):
        """Start Telegram bot polling in background thread"""
        if not self.enabled or not self.bot:
            return
        
        def polling_thread():
            try:
                self.bot.polling(
                    none_stop=True,
                    interval=TradingConfig.TELEGRAM_POLLING_INTERVAL
                )
            except Exception as e:
                self.logger.log('error', f"Telegram polling error: {str(e)}")
        
        thread = threading.Thread(target=polling_thread, daemon=True)
        thread.start()
        self.logger.log('info', "Telegram polling started")
    
    def send_message(self, message: str, alert_type: str = "info") -> bool:
        """Send message to Telegram"""
        try:
            if not self.enabled or not self.bot or not self.chat_id:
                return False
            
            # Check cooldown for alert types
            if alert_type in self.last_alert_time:
                elapsed = (datetime.datetime.now() - 
                          self.last_alert_time[alert_type]).seconds
                if elapsed < TradingConfig.ALERT_COOLDOWN_SECONDS:
                    return False
            
            # Format message based on type
            if alert_type == "error":
                formatted_message = f"🚨 *ERROR*\n{message}"
            elif alert_type == "warning":
                formatted_message = f"⚠️ *WARNING*\n{message}"
            elif alert_type == "success":
                formatted_message = f"✅ *SUCCESS*\n{message}"
            elif alert_type == "signal":
                formatted_message = f"📈 *TRADING SIGNAL*\n{message}"
            elif alert_type == "position":
                formatted_message = f"💰 *POSITION UPDATE*\n{message}"
            else:
                formatted_message = f"ℹ️ *INFO*\n{message}"
            
            # Send message
            self.bot.send_message(
                chat_id=self.chat_id,
                text=formatted_message,
                parse_mode='Markdown'
            )
            
            self.last_alert_time[alert_type] = datetime.datetime.now()
            self.logger.log('debug', f"Telegram message sent: {alert_type}")
            return True
            
        except Exception as e:
            self.logger.log('error', f"Failed to send Telegram message: {str(e)}")
            return False
    
    def send_signal_alert(self, signal: TradingSignal):
        """Send trading signal alert"""
        if not self.send_alerts:
            return
        
        # Create signal emoji
        if signal.signal == TradeSignal.STRONG_BUY:
            emoji = "🟢🟢"
        elif signal.signal == TradeSignal.BUY:
            emoji = "🟢"
        elif signal.signal == TradeSignal.STRONG_SELL:
            emoji = "🔴🔴"
        elif signal.signal == TradeSignal.SELL:
            emoji = "🔴"
        else:
            emoji = "⚪"
        
        message = (
            f"{emoji} *NEW TRADING SIGNAL* {emoji}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 *Symbol:* {signal.symbol}\n"
            f"⏰ *Timeframe:* {signal.timeframe.name}\n"
            f"📈 *Signal:* {signal.signal.name}\n"
            f"💪 *Strength:* {signal.strength:.2%}\n"
            f"🎯 *Confidence:* {signal.confidence:.2%}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 *Entry Price:* {signal.entry_price:.5f}\n"
            f"🛑 *Stop Loss:* {signal.stop_loss:.5f}\n"
            f"🎯 *Take Profit:* {signal.take_profit:.5f}\n"
            f"📦 *Volume:* {signal.recommended_volume}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📅 *Time:* {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        if signal.patterns:
            patterns_text = ", ".join([p.pattern_type.value for p in signal.patterns[:3]])
            message += f"\n*🎯 Patterns:* {patterns_text}"
        
        # Add indicator summary
        message += f"\n*📊 Indicators:* RSI: {signal.indicators.rsi:.1f}, MACD: {signal.indicators.macd:.4f}"
        
        self.send_message(message, "signal")
    
    def send_position_alert(self, position: Position, action: str):
        """Send position update alert"""
        if not self.send_positions:
            return
        
        profit_color = "🟢" if position.profit >= 0 else "🔴"
        profit_emoji = "📈" if position.profit >= 0 else "📉"
        
        message = (
            f"{profit_emoji} *POSITION {action.upper()}*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🎫 *Ticket:* {position.ticket}\n"
            f"📊 *Symbol:* {position.symbol}\n"
            f"📈 *Type:* {position.order_type.name}\n"
            f"📦 *Volume:* {position.volume}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 *Entry Price:* {position.entry_price:.5f}\n"
            f"📊 *Current Price:* {position.current_price:.5f}\n"
            f"🛑 *Stop Loss:* {position.stop_loss:.5f}\n"
            f"🎯 *Take Profit:* {position.take_profit:.5f}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💵 *Profit:* {profit_color} {position.profit:.2f}\n"
            f"📅 *Time:* {position.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        self.send_message(message, "position")
    
    def send_error_alert(self, error_message: str):
        """Send error alert"""
        if not self.send_errors:
            return
        
        message = (
            f"🚨 *SYSTEM ERROR*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{error_message}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📅 *Time:* {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        self.send_message(message, "error")


# ==============================================
# CHART MANAGER
# ==============================================

class ChartManager:
    """Manages candlestick charts"""
    
    def __init__(self):
        self.logger = TradingLogger()
    
    def create_candlestick_chart(self, candles: List[Candle], signal: Optional[TradingSignal] = None) -> Figure:
        """Create candlestick chart with indicators"""
        try:
            # Prepare data for candlestick chart
            dates = [c.timestamp for c in candles]
            opens = [c.open for c in candles]
            highs = [c.high for c in candles]
            lows = [c.low for c in candles]
            closes = [c.close for c in candles]
            
            # Create figure with subplots
            fig = plt.figure(figsize=(12, 8), facecolor='#2b2b2b')
            gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1], hspace=0.1)
            
            # Main price chart
            ax1 = fig.add_subplot(gs[0])
            ax1.set_facecolor('#2b2b2b')
            
            # Plot candlesticks
            for i in range(len(candles)):
                color = 'green' if candles[i].close > candles[i].open else 'red'
                
                # Plot body
                ax1.add_patch(Rectangle(
                    (i - 0.3, min(candles[i].open, candles[i].close)),
                    0.6, abs(candles[i].close - candles[i].open),
                    facecolor=color, edgecolor=color
                ))
                
                # Plot wicks
                ax1.plot([i, i], [candles[i].low, candles[i].high], color=color, linewidth=1)
            
            # Add indicators if available
            if signal:
                # Add moving averages
                if hasattr(signal.indicators, 'sma_20'):
                    sma_20 = [signal.indicators.sma_20] * len(candles)
                    ax1.plot(range(len(candles)), sma_20, 'yellow', label='SMA 20', linewidth=1, alpha=0.7)
                
                if hasattr(signal.indicators, 'sma_50'):
                    sma_50 = [signal.indicators.sma_50] * len(candles)
                    ax1.plot(range(len(candles)), sma_50, 'orange', label='SMA 50', linewidth=1, alpha=0.7)
                
                # Add entry price line
                if signal.entry_price > 0:
                    ax1.axhline(y=signal.entry_price, color='blue', linestyle='--', 
                               label=f'Entry: {signal.entry_price:.5f}', alpha=0.5)
                
                # Add stop loss and take profit lines
                if signal.stop_loss > 0:
                    ax1.axhline(y=signal.stop_loss, color='red', linestyle='--', 
                               label=f'SL: {signal.stop_loss:.5f}', alpha=0.5)
                
                if signal.take_profit > 0:
                    ax1.axhline(y=signal.take_profit, color='green', linestyle='--', 
                               label=f'TP: {signal.take_profit:.5f}', alpha=0.5)
                
                # Add signal annotation
                signal_text = f"{signal.signal.name} (Confidence: {signal.confidence:.2%})"
                ax1.text(0.02, 0.98, signal_text, transform=ax1.transAxes,
                        fontsize=12, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Customize main chart
            ax1.set_ylabel('Price', color='white')
            ax1.tick_params(axis='x', colors='white')
            ax1.tick_params(axis='y', colors='white')
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper left', facecolor='#2b2b2b', edgecolor='white', 
                      labelcolor='white', fontsize=8)
            
            # Volume chart
            ax2 = fig.add_subplot(gs[1], sharex=ax1)
            ax2.set_facecolor('#2b2b2b')
            
            volumes = [c.volume for c in candles]
            colors = ['green' if c.close > c.open else 'red' for c in candles]
            ax2.bar(range(len(candles)), volumes, color=colors, alpha=0.5)
            
            ax2.set_ylabel('Volume', color='white')
            ax2.tick_params(axis='x', colors='white')
            ax2.tick_params(axis='y', colors='white')
            ax2.grid(True, alpha=0.3)
            
            # RSI chart
            ax3 = fig.add_subplot(gs[2], sharex=ax1)
            ax3.set_facecolor('#2b2b2b')
            
            # Calculate RSI
            if len(closes) >= 14:
                rsi = talib.RSI(np.array(closes), timeperiod=14)
                ax3.plot(range(len(candles)), rsi, 'purple', linewidth=1)
                ax3.axhline(y=70, color='red', linestyle='--', alpha=0.3)
                ax3.axhline(y=30, color='green', linestyle='--', alpha=0.3)
                ax3.fill_between(range(len(candles)), 70, 30, alpha=0.1, color='gray')
            
            ax3.set_ylabel('RSI', color='white')
            ax3.set_xlabel('Candles', color='white')
            ax3.tick_params(axis='x', colors='white')
            ax3.tick_params(axis='y', colors='white')
            ax3.grid(True, alpha=0.3)
            
            # Set x-axis ticks
            if len(candles) > 20:
                step = len(candles) // 10
                indices = list(range(0, len(candles), step))
                dates_labels = [dates[i].strftime('%m-%d %H:%M') for i in indices]
                ax3.set_xticks(indices)
                ax3.set_xticklabels(dates_labels, rotation=45)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.logger.log('error', f"Failed to create chart: {str(e)}")
            # Return empty figure on error
            fig = plt.figure(figsize=(12, 8), facecolor='#2b2b2b')
            return fig


# ==============================================
# TRADING BOT CORE
# ==============================================

class TradingBot:
    """Main trading bot class"""
    
    def __init__(self):
        # Initialize components
        self.config_manager = ConfigManager()
        self.logger = TradingLogger()
        self.mt5_manager = MT5Manager(self.config_manager)
        self.pattern_detector = PatternDetector()
        self.indicators_calculator = TechnicalIndicators()
        self.signal_generator = SignalGenerator(self.pattern_detector, self.indicators_calculator)
        self.telegram_manager = TelegramManager(self.config_manager)
        self.chart_manager = ChartManager()
        
        # State
        self.running = False
        self.auto_trading = False
        self.risk_free_mode = True
        self.current_symbol = TradingConfig.DEFAULT_SYMBOL
        self.current_timeframe = TimeFrame.M15
        self.current_signal = None
        self.last_analysis_time = None
        
        # Load configuration
        self.load_config()
        
        self.logger.log('info', 
            f"Trading Bot v{TradingConfig.VERSION} initialized"
        )
    
    def load_config(self):
        """Load bot configuration"""
        config = self.config_manager.config
        self.current_symbol = config.get('symbol', TradingConfig.DEFAULT_SYMBOL)
        
        timeframe_str = config.get('timeframe', 'M15')
        self.current_timeframe = getattr(TimeFrame, timeframe_str, TimeFrame.M15)
        
        self.auto_trading = config.get('auto_trading', False)
        self.risk_free_mode = config.get('risk_free_mode', True)
        
        # Initialize Telegram
        if config.get('telegram_enabled', False):
            self.telegram_manager.initialize()
    
    def connect_mt5(self) -> bool:
        """Connect to MT5"""
        try:
            success = self.mt5_manager.connect()
            if success and self.telegram_manager.enabled:
                self.telegram_manager.send_message(
                    "✅ Connected to MetaTrader 5",
                    "success"
                )
            return success
        except Exception as e:
            self.logger.log('error', f"MT5 connection failed: {str(e)}")
            return False
    
    def analyze_market(self) -> Optional[TradingSignal]:
        """Analyze current market and generate signal"""
        try:
            # Get historical data
            candles = self.mt5_manager.get_historical_data(
                self.current_symbol,
                self.current_timeframe,
                count=TradingConfig.CHART_CANDLES_COUNT
            )
            
            if not candles or len(candles) < 50:
                self.logger.log('warning', "Insufficient data for analysis")
                return None
            
            # Generate signal
            signal = self.signal_generator.generate_signal(
                self.current_symbol,
                self.current_timeframe,
                candles
            )
            
            self.current_signal = signal
            self.last_analysis_time = datetime.datetime.now()
            
            # Log signal
            self.logger.log('info',
                f"Signal generated: {signal.signal.name} | "
                f"Strength: {signal.strength:.2%} | "
                f"Confidence: {signal.confidence:.2%}"
            )
            
            # Send Telegram alert if enabled
            if (self.telegram_manager.enabled and 
                signal.signal != TradeSignal.NEUTRAL and
                signal.confidence > 0.6):
                self.telegram_manager.send_signal_alert(signal)
            
            return signal
            
        except Exception as e:
            self.logger.log('error', f"Market analysis failed: {str(e)}")
            return None
    
    def execute_trade(self, signal: TradingSignal) -> Optional[int]:
        """Execute trade based on signal"""
        try:
            if self.risk_free_mode:
                self.logger.log('info', 
                    f"Risk-free mode: Signal {signal.signal.name} received "
                    f"(Entry: {signal.entry_price:.5f}, "
                    f"SL: {signal.stop_loss:.5f}, "
                    f"TP: {signal.take_profit:.5f})"
                )
                return None
            
            if not self.auto_trading:
                self.logger.log('info', "Auto-trading disabled")
                return None
            
            # Check if we should trade based on signal strength and confidence
            if (signal.signal == TradeSignal.NEUTRAL or 
                signal.confidence < 0.6 or
                signal.strength < 0.3):
                self.logger.log('info', "Signal too weak for trading")
                return None
            
            # Check current positions
            positions = self.mt5_manager.get_open_positions()
            if len(positions) >= TradingConfig.MAX_POSITIONS:
                self.logger.log('warning', 
                    f"Maximum positions ({TradingConfig.MAX_POSITIONS}) reached"
                )
                return None
            
            # Calculate position size
            stop_loss_pips = abs(signal.entry_price - signal.stop_loss)
            if self.current_symbol.endswith('JPY'):
                stop_loss_pips *= 100
            else:
                stop_loss_pips *= 10000
            
            volume = self.mt5_manager.calculate_position_size(
                self.current_symbol,
                TradingConfig.MAX_RISK_PERCENT,
                stop_loss_pips
            )
            
            # Determine order type
            if signal.signal in [TradeSignal.BUY, TradeSignal.STRONG_BUY]:
                order_type = OrderType.BUY
            else:
                order_type = OrderType.SELL
            
            # Place order
            order_id = self.mt5_manager.place_order(
                symbol=self.current_symbol,
                order_type=order_type,
                volume=volume,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                comment=f"Bot: {signal.signal.name} | Confidence: {signal.confidence:.2%}"
            )
            
            if order_id:
                self.logger.log('info', 
                    f"Trade executed: {order_type.name} {volume} {self.current_symbol} | "
                    f"Ticket: {order_id}"
                )
                
                if self.telegram_manager.enabled:
                    # Get the position to send alert
                    positions = self.mt5_manager.get_open_positions()
                    for position in positions:
                        if position.ticket == order_id:
                            self.telegram_manager.send_position_alert(
                                position, "opened"
                            )
                            break
            
            return order_id
            
        except Exception as e:
            self.logger.log('error', f"Trade execution failed: {str(e)}")
            return None
    
    def monitor_positions(self):
        """Monitor and manage open positions"""
        try:
            positions = self.mt5_manager.get_open_positions()
            
            for position in positions:
                # Check if position needs management
                # (e.g., trailing stop, breakeven, etc.)
                # This is a simplified version
                
                current_time = datetime.datetime.now()
                position_age = current_time - position.timestamp
                
                # Log position status
                self.logger.log('debug',
                    f"Position {position.ticket}: "
                    f"Profit: {position.profit:.2f} | "
                    f"Age: {position_age}"
                )
            
            return positions
            
        except Exception as e:
            self.logger.log('error', f"Position monitoring failed: {str(e)}")
            return []
    
    def get_market_data(self) -> Optional[List[Candle]]:
        """Get current market data"""
        try:
            return self.mt5_manager.get_historical_data(
                self.current_symbol,
                self.current_timeframe,
                count=TradingConfig.CHART_CANDLES_COUNT
            )
        except Exception as e:
            self.logger.log('error', f"Failed to get market data: {str(e)}")
            return None
    
    def start(self):
        """Start the trading bot"""
        try:
            self.running = True
            
            # Connect to MT5
            if not self.connect_mt5():
                self.logger.log('error', "Failed to connect to MT5")
                return False
            
            self.logger.log('info', "Trading bot started")
            
            if self.telegram_manager.enabled:
                self.telegram_manager.send_message(
                    "🚀 Trading bot started successfully!",
                    "success"
                )
            
            return True
            
        except Exception as e:
            self.logger.log('error', f"Failed to start bot: {str(e)}")
            return False
    
    def stop(self):
        """Stop the trading bot"""
        try:
            self.running = False
            
            # Close all positions if in risk-free mode
            if self.risk_free_mode:
                positions = self.mt5_manager.get_open_positions()
                for position in positions:
                    self.mt5_manager.close_position(position.ticket)
            
            # Disconnect from MT5
            self.mt5_manager.disconnect()
            
            self.logger.log('info', "Trading bot stopped")
            
            if self.telegram_manager.enabled:
                self.telegram_manager.send_message(
                    "🛑 Trading bot stopped",
                    "info"
                )
            
            return True
            
        except Exception as e:
            self.logger.log('error', f"Failed to stop bot: {str(e)}")
            return False


# ==============================================
# Tkinter GUI Application
# ==============================================

class TradingGUI:
    """Tkinter GUI for the trading bot"""
    
    def __init__(self, trading_bot: TradingBot):
        self.bot = trading_bot
        self.root = tk.Tk()
        self.root.title(f"{TradingConfig.APP_NAME} v{TradingConfig.VERSION}")
        self.root.geometry("1400x900")
        
        # Set window icon (you can add an icon file if available)
        try:
            self.root.iconbitmap('icon.ico')
        except:
            pass
        
        # Configure styles
        self.setup_styles()
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create GUI components
        self.create_widgets()
        
        # Start update loop
        self.update_interval = TradingConfig.GUI_UPDATE_INTERVAL_MS
        self.schedule_updates()
        
    def setup_styles(self):
        """Configure ttk styles"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        self.bg_color = '#2b2b2b'
        self.fg_color = '#ffffff'
        self.accent_color = '#4CAF50'
        self.warning_color = '#FF9800'
        self.error_color = '#F44336'
        
        style.configure('TFrame', background=self.bg_color)
        style.configure('TLabel', background=self.bg_color, foreground=self.fg_color)
        style.configure('TButton', background=self.accent_color, foreground=self.fg_color)
        style.configure('TLabelframe', background=self.bg_color, foreground=self.fg_color)
        style.configure('TLabelframe.Label', background=self.bg_color, foreground=self.fg_color)
        
        self.root.configure(bg=self.bg_color)
    
    def create_menu_bar(self):
        """Create menu bar with File, View, Help, and Settings"""
        menubar = Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Save Configuration", command=self.save_configuration)
        file_menu.add_command(label="Load Configuration", command=self.load_configuration)
        file_menu.add_separator()
        file_menu.add_command(label="Export Data", command=self.export_data)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # View menu
        view_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Refresh Chart", command=self.refresh_chart)
        view_menu.add_command(label="Toggle Dark/Light Mode", command=self.toggle_theme)
        view_menu.add_separator()
        view_menu.add_command(label="Show Indicators", command=self.toggle_indicators)
        view_menu.add_command(label="Show Grid", command=self.toggle_grid)
        
        # Settings menu
        settings_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Settings", menu=settings_menu)
        settings_menu.add_command(label="MT5 Configuration", command=self.configure_mt5)
        settings_menu.add_command(label="Telegram Configuration", command=self.configure_telegram)
        settings_menu.add_command(label="Trading Parameters", command=self.configure_trading)
        settings_menu.add_separator()
        settings_menu.add_command(label="Risk Management", command=self.configure_risk)
        
        # Help menu
        help_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="User Guide", command=self.show_user_guide)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_separator()
        help_menu.add_command(label="Check for Updates", command=self.check_updates)
    
    def create_widgets(self):
        """Create all GUI widgets"""
        # Create main container with paned window
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Controls
        left_panel = ttk.Frame(main_paned)
        main_paned.add(left_panel, weight=1)
        
        # Right panel - Charts and Logs
        right_panel = ttk.Frame(main_paned)
        main_paned.add(right_panel, weight=3)
        
        # Create control widgets
        self.create_control_widgets(left_panel)
        
        # Create chart and log widgets in right panel
        self.create_chart_widgets(right_panel)
    
    def create_control_widgets(self, parent):
        """Create control widgets"""
        # Create notebook for tabs
        control_notebook = ttk.Notebook(parent)
        control_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Connection tab
        conn_frame = ttk.Frame(control_notebook)
        control_notebook.add(conn_frame, text="Connection")
        self.create_connection_tab(conn_frame)
        
        # Trading tab
        trading_frame = ttk.Frame(control_notebook)
        control_notebook.add(trading_frame, text="Trading")
        self.create_trading_tab(trading_frame)
        
        # Positions tab
        positions_frame = ttk.Frame(control_notebook)
        control_notebook.add(positions_frame, text="Positions")
        self.create_positions_tab(positions_frame)
        
        # Analysis tab
        analysis_frame = ttk.Frame(control_notebook)
        control_notebook.add(analysis_frame, text="Analysis")
        self.create_analysis_tab(analysis_frame)
    
    def create_connection_tab(self, parent):
        """Create connection tab widgets"""
        # MT5 Connection
        mt5_frame = ttk.LabelFrame(parent, text="MetaTrader 5", padding=10)
        mt5_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(mt5_frame, text="Status:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.mt5_status_label = ttk.Label(mt5_frame, text="Disconnected", foreground="red")
        self.mt5_status_label.grid(row=0, column=1, sticky=tk.W, pady=5)
        
        self.connect_btn = ttk.Button(mt5_frame, text="Connect MT5", 
                                     command=self.connect_mt5)
        self.connect_btn.grid(row=1, column=0, pady=5, padx=5)
        
        self.disconnect_btn = ttk.Button(mt5_frame, text="Disconnect MT5", 
                                        command=self.disconnect_mt5, state=tk.DISABLED)
        self.disconnect_btn.grid(row=1, column=1, pady=5, padx=5)
        
        # Telegram Connection
        telegram_frame = ttk.LabelFrame(parent, text="Telegram", padding=10)
        telegram_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(telegram_frame, text="Status:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.telegram_status_label = ttk.Label(telegram_frame, text="Disabled", foreground="orange")
        self.telegram_status_label.grid(row=0, column=1, sticky=tk.W, pady=5)
        
        self.telegram_btn = ttk.Button(telegram_frame, text="Configure Telegram",
                                      command=self.configure_telegram)
        self.telegram_btn.grid(row=1, column=0, columnspan=2, pady=5)
        
        # Account Info
        account_frame = ttk.LabelFrame(parent, text="Account Information", padding=10)
        account_frame.pack(fill=tk.X)
        
        self.account_labels = {}
        info_fields = [
            ("Balance:", "balance"),
            ("Equity:", "equity"),
            ("Margin:", "margin"),
            ("Free Margin:", "free_margin"),
            ("Leverage:", "leverage"),
        ]
        
        for i, (label_text, key) in enumerate(info_fields):
            ttk.Label(account_frame, text=label_text).grid(row=i, column=0, sticky=tk.W, pady=2)
            label = ttk.Label(account_frame, text="N/A")
            label.grid(row=i, column=1, sticky=tk.W, pady=2)
            self.account_labels[key] = label
    
    def create_trading_tab(self, parent):
        """Create trading tab widgets"""
        # Symbol selection
        symbol_frame = ttk.LabelFrame(parent, text="Trading Symbol", padding=10)
        symbol_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(symbol_frame, text="Symbol:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.symbol_var = tk.StringVar(value=self.bot.current_symbol)
        symbol_combo = ttk.Combobox(symbol_frame, textvariable=self.symbol_var,
                                   values=["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "BTCUSD"],
                                   state="readonly", width=15)
        symbol_combo.grid(row=0, column=1, pady=5)
        symbol_combo.bind('<<ComboboxSelected>>', self.change_symbol)
        
        # Timeframe selection
        ttk.Label(symbol_frame, text="Timeframe:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.timeframe_var = tk.StringVar(value=self.bot.current_timeframe.name)
        timeframe_combo = ttk.Combobox(symbol_frame, textvariable=self.timeframe_var,
                                      values=[tf.name for tf in TimeFrame], 
                                      state="readonly", width=15)
        timeframe_combo.grid(row=1, column=1, pady=5)
        timeframe_combo.bind('<<ComboboxSelected>>', self.change_timeframe)
        
        # Trading controls
        control_frame = ttk.LabelFrame(parent, text="Trading Controls", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Auto-trading checkbox
        self.auto_trading_var = tk.BooleanVar(value=self.bot.auto_trading)
        auto_trading_cb = ttk.Checkbutton(control_frame, text="Auto Trading",
                                         variable=self.auto_trading_var,
                                         command=self.toggle_auto_trading)
        auto_trading_cb.grid(row=0, column=0, sticky=tk.W, pady=5)
        
        # Risk-free mode checkbox
        self.risk_free_var = tk.BooleanVar(value=self.bot.risk_free_mode)
        risk_free_cb = ttk.Checkbutton(control_frame, text="Risk-Free Mode",
                                      variable=self.risk_free_var,
                                      command=self.toggle_risk_free)
        risk_free_cb.grid(row=1, column=0, sticky=tk.W, pady=5)
        
        # Start/Stop buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=10)
        
        self.start_btn = ttk.Button(button_frame, text="▶ Start Bot", 
                                   command=self.start_bot, width=12)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(button_frame, text="⏹ Stop Bot", 
                                  command=self.stop_bot, width=12, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        # Manual trading
        manual_frame = ttk.LabelFrame(parent, text="Manual Trading", padding=10)
        manual_frame.pack(fill=tk.X)
        
        ttk.Label(manual_frame, text="Volume:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.volume_var = tk.StringVar(value=str(TradingConfig.DEFAULT_VOLUME))
        ttk.Entry(manual_frame, textvariable=self.volume_var, width=10).grid(row=0, column=1, pady=5)
        
        self.buy_btn = ttk.Button(manual_frame, text="BUY", 
                                 command=lambda: self.manual_trade("BUY"),
                                 width=8)
        self.buy_btn.grid(row=1, column=0, pady=5, padx=5)
        
        self.sell_btn = ttk.Button(manual_frame, text="SELL", 
                                  command=lambda: self.manual_trade("SELL"),
                                  width=8)
        self.sell_btn.grid(row=1, column=1, pady=5, padx=5)
    
    def create_positions_tab(self, parent):
        """Create positions tab widgets"""
        # Positions treeview
        columns = ('Ticket', 'Symbol', 'Type', 'Volume', 'Entry', 'Current', 'Profit', 'Status')
        self.positions_tree = ttk.Treeview(parent, columns=columns, show='headings', height=15)
        
        # Define column headings
        for col in columns:
            self.positions_tree.heading(col, text=col)
            self.positions_tree.column(col, width=80)
        
        # Configure column widths
        self.positions_tree.column('Ticket', width=60)
        self.positions_tree.column('Symbol', width=70)
        self.positions_tree.column('Type', width=60)
        self.positions_tree.column('Volume', width=60)
        self.positions_tree.column('Entry', width=80)
        self.positions_tree.column('Current', width=80)
        self.positions_tree.column('Profit', width=80)
        self.positions_tree.column('Status', width=80)
        
        # Add scrollbars
        scrollbar_y = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.positions_tree.yview)
        scrollbar_x = ttk.Scrollbar(parent, orient=tk.HORIZONTAL, command=self.positions_tree.xview)
        self.positions_tree.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        
        # Layout
        self.positions_tree.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Control buttons
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(button_frame, text="Refresh", command=self.update_positions).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Close Selected", command=self.close_selected_position).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Close All", command=self.close_all_positions).pack(side=tk.LEFT, padx=5)
    
    def create_analysis_tab(self, parent):
        """Create analysis tab widgets"""
        # Signal display
        signal_frame = ttk.LabelFrame(parent, text="Current Signal", padding=10)
        signal_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.signal_label = ttk.Label(signal_frame, text="No Signal", 
                                     font=('Arial', 16, 'bold'))
        self.signal_label.pack(pady=5)
        
        # Signal details
        details_frame = ttk.Frame(signal_frame)
        details_frame.pack(fill=tk.X, pady=5)
        
        self.signal_details = {}
        detail_fields = [
            ("Symbol:", "symbol"),
            ("Timeframe:", "timeframe"),
            ("Strength:", "strength"),
            ("Confidence:", "confidence"),
            ("Entry Price:", "entry"),
            ("Stop Loss:", "sl"),
            ("Take Profit:", "tp"),
        ]
        
        for i, (label_text, key) in enumerate(detail_fields):
            frame = ttk.Frame(details_frame)
            frame.pack(fill=tk.X, pady=2)
            ttk.Label(frame, text=label_text, width=15, anchor=tk.W).pack(side=tk.LEFT)
            label = ttk.Label(frame, text="N/A")
            label.pack(side=tk.LEFT)
            self.signal_details[key] = label
        
        # Analysis controls
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, pady=10)
        
        self.analyze_btn = ttk.Button(control_frame, text="Analyze Market",
                                     command=self.analyze_market, width=15)
        self.analyze_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Send to Telegram",
                  command=self.send_signal_to_telegram, width=15).pack(side=tk.LEFT, padx=5)
        
        # Patterns display
        patterns_frame = ttk.LabelFrame(parent, text="Detected Patterns", padding=10)
        patterns_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        self.patterns_text = scrolledtext.ScrolledText(
            patterns_frame,
            height=8,
            bg='#1e1e1e',
            fg=self.fg_color,
            insertbackground=self.fg_color,
            wrap=tk.WORD
        )
        self.patterns_text.pack(fill=tk.BOTH, expand=True)
    
    def create_chart_widgets(self, parent):
        """Create chart and log widgets"""
        # Chart frame
        chart_frame = ttk.LabelFrame(parent, text="Price Chart", padding=10)
        chart_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(10, 6), dpi=100, facecolor=self.bg_color)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(self.bg_color)
        
        # Customize axes
        self.ax.tick_params(colors=self.fg_color)
        self.ax.xaxis.label.set_color(self.fg_color)
        self.ax.yaxis.label.set_color(self.fg_color)
        self.ax.title.set_color(self.fg_color)
        
        # Create canvas
        self.chart_canvas = FigureCanvasTkAgg(self.fig, chart_frame)
        self.chart_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        toolbar = NavigationToolbar2Tk(self.chart_canvas, chart_frame)
        toolbar.update()
        self.chart_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Log frame
        log_frame = ttk.LabelFrame(parent, text="Log Output", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=False, pady=(0, 10))
        
        # Create scrolled text widget for logs
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            height=8,
            bg='#1e1e1e',
            fg=self.fg_color,
            insertbackground=self.fg_color,
            wrap=tk.WORD
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Log control buttons
        log_button_frame = ttk.Frame(log_frame)
        log_button_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(log_button_frame, text="Clear Log", 
                  command=self.clear_log).pack(side=tk.LEFT, padx=5)
        ttk.Button(log_button_frame, text="Save Log", 
                  command=self.save_log).pack(side=tk.LEFT, padx=5)
    
    def schedule_updates(self):
        """Schedule GUI updates"""
        self.update_gui()
        self.root.after(self.update_interval, self.schedule_updates)
    
    def update_gui(self):
        """Update GUI elements"""
        try:
            # Update MT5 status
            if self.bot.mt5_manager.connected:
                self.mt5_status_label.config(text="Connected", foreground="green")
                self.connect_btn.config(state=tk.DISABLED)
                self.disconnect_btn.config(state=tk.NORMAL)
                
                # Update account info
                if self.bot.mt5_manager.account_info:
                    account = self.bot.mt5_manager.account_info
                    self.account_labels['balance'].config(text=f"{account.balance:.2f}")
                    self.account_labels['equity'].config(text=f"{account.equity:.2f}")
                    self.account_labels['margin'].config(text=f"{account.margin:.2f}")
                    self.account_labels['free_margin'].config(text=f"{account.margin_free:.2f}")
                    self.account_labels['leverage'].config(text=f"1:{account.leverage}")
            else:
                self.mt5_status_label.config(text="Disconnected", foreground="red")
                self.connect_btn.config(state=tk.NORMAL)
                self.disconnect_btn.config(state=tk.DISABLED)
            
            # Update Telegram status
            if self.bot.telegram_manager.enabled:
                self.telegram_status_label.config(text="Connected", foreground="green")
            else:
                self.telegram_status_label.config(text="Disabled", foreground="orange")
            
            # Update bot controls
            if self.bot.running:
                self.start_btn.config(state=tk.DISABLED)
                self.stop_btn.config(state=tk.NORMAL)
                self.analyze_btn.config(state=tk.DISABLED)
                self.buy_btn.config(state=tk.DISABLED)
                self.sell_btn.config(state=tk.DISABLED)
            else:
                self.start_btn.config(state=tk.NORMAL)
                self.stop_btn.config(state=tk.DISABLED)
                self.analyze_btn.config(state=tk.NORMAL)
                self.buy_btn.config(state=tk.NORMAL)
                self.sell_btn.config(state=tk.NORMAL)
            
            # Update positions
            self.update_positions()
            
            # Update logs
            self.update_logs()
            
            # Update chart periodically
            if self.bot.running and self.bot.mt5_manager.connected:
                self.update_chart()
            
        except Exception as e:
            print(f"GUI update error: {e}")
    
    def update_chart(self):
        """Update the candlestick chart"""
        try:
            # Get market data
            candles = self.bot.get_market_data()
            if not candles or len(candles) < 10:
                return
            
            # Clear previous chart
            self.ax.clear()
            
            # Create new chart
            fig = self.bot.chart_manager.create_candlestick_chart(candles, self.bot.current_signal)
            
            # Update the canvas
            self.chart_canvas.figure = fig
            self.chart_canvas.draw()
            
        except Exception as e:
            print(f"Chart update error: {e}")
    
    def update_positions(self):
        """Update positions treeview"""
        try:
            # Clear existing items
            for item in self.positions_tree.get_children():
                self.positions_tree.delete(item)
            
            # Get current positions
            positions = self.bot.mt5_manager.get_open_positions()
            
            # Add positions to treeview
            for position in positions:
                profit = position.profit
                status = "Profit" if profit >= 0 else "Loss"
                profit_color = "green" if profit >= 0 else "red"
                
                self.positions_tree.insert('', tk.END, values=(
                    position.ticket,
                    position.symbol,
                    position.order_type.name,
                    position.volume,
                    f"{position.entry_price:.5f}",
                    f"{position.current_price:.5f}",
                    f"{profit:.2f}",
                    status
                ), tags=(status,))
            
            # Configure tags for colors
            self.positions_tree.tag_configure('Profit', foreground='green')
            self.positions_tree.tag_configure('Loss', foreground='red')
            
        except Exception as e:
            print(f"Position update error: {e}")
    
    def update_logs(self):
        """Update log display"""
        try:
            # Get recent log messages
            messages = self.bot.logger.get_log_messages(10)
            
            # Add new messages to log text
            for message in messages:
                if message not in self.log_text.get(1.0, tk.END):
                    self.log_text.insert(tk.END, message + '\n')
            
            # Limit log size
            lines = self.log_text.get(1.0, tk.END).split('\n')
            if len(lines) > TradingConfig.GUI_MAX_LOG_LINES:
                self.log_text.delete(1.0, f"{len(lines)-TradingConfig.GUI_MAX_LOG_LINES}.0")
            
            # Auto-scroll to bottom
            self.log_text.see(tk.END)
            
        except Exception as e:
            print(f"Log update error: {e}")
    
    # Menu command handlers
    def save_configuration(self):
        """Save current configuration"""
        try:
            self.bot.config_manager.save_config(self.bot.config_manager.config)
            messagebox.showinfo("Success", "Configuration saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration: {e}")
    
    def load_configuration(self):
        """Load configuration from file"""
        try:
            file_path = filedialog.askopenfilename(
                title="Load Configuration",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if file_path:
                with open(file_path, 'r') as f:
                    config = json.load(f)
                self.bot.config_manager.save_config(config)
                self.bot.load_config()
                messagebox.showinfo("Success", "Configuration loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load configuration: {e}")
    
    def export_data(self):
        """Export trading data"""
        try:
            file_path = filedialog.asksaveasfilename(
                title="Export Data",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            if file_path:
                # Export positions
                positions = self.bot.mt5_manager.get_open_positions()
                if positions:
                    df = pd.DataFrame([vars(p) for p in positions])
                    df.to_csv(file_path, index=False)
                    messagebox.showinfo("Success", f"Data exported to {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export data: {e}")
    
    def refresh_chart(self):
        """Refresh the chart"""
        self.update_chart()
    
    def toggle_theme(self):
        """Toggle between dark and light theme"""
        messagebox.showinfo("Info", "Theme toggle feature coming soon!")
    
    def toggle_indicators(self):
        """Toggle indicator display"""
        messagebox.showinfo("Info", "Indicator toggle feature coming soon!")
    
    def toggle_grid(self):
        """Toggle grid display"""
        self.ax.grid(not self.ax.get_grid())
        self.chart_canvas.draw()
    
    def configure_mt5(self):
        """Open MT5 configuration dialog"""
        dialog = MT5ConfigDialog(self.root, self.bot.config_manager)
        self.root.wait_window(dialog)
    
    def configure_telegram(self):
        """Open Telegram configuration dialog"""
        dialog = TelegramConfigDialog(self.root, self.bot.config_manager)
        self.root.wait_window(dialog)
    
    def configure_trading(self):
        """Open trading parameters dialog"""
        dialog = TradingConfigDialog(self.root, self.bot.config_manager)
        self.root.wait_window(dialog)
    
    def configure_risk(self):
        """Open risk management dialog"""
        dialog = RiskConfigDialog(self.root, self.bot.config_manager)
        self.root.wait_window(dialog)
    
    def show_user_guide(self):
        """Show user guide"""
        guide_text = """
        Forex Trading Bot User Guide
        
        1. Connection:
           - Configure MT5 connection in Settings > MT5 Configuration
           - Configure Telegram for alerts in Settings > Telegram Configuration
        
        2. Trading:
           - Select symbol and timeframe
           - Enable Auto Trading for automatic execution
           - Use Risk-Free Mode for testing
        
        3. Analysis:
           - Click 'Analyze Market' to generate signals
           - View signals and patterns in Analysis tab
        
        4. Positions:
           - Monitor open positions in Positions tab
           - Manually close positions if needed
        
        For more information, visit our documentation.
        """
        messagebox.showinfo("User Guide", guide_text)
    
    def show_about(self):
        """Show about dialog"""
        about_text = f"""
        {TradingConfig.APP_NAME}
        Version: {TradingConfig.VERSION}
        
        Advanced Forex Trading Bot with:
        - MT5 Integration
        - Pattern Recognition
        - Technical Analysis
        - Telegram Alerts
        - Risk Management
        
        Created for professional forex trading.
       
        """
        messagebox.showinfo("About", about_text)
    
    def check_updates(self):
        """Check for updates"""
        messagebox.showinfo("Check Updates", "Update check feature coming soon!")
    
    # Button command handlers
    def connect_mt5(self):
        """Connect to MT5"""
        if self.bot.connect_mt5():
            messagebox.showinfo("Success", "Connected to MetaTrader 5")
        else:
            messagebox.showerror("Error", "Failed to connect to MetaTrader 5")
    
    def disconnect_mt5(self):
        """Disconnect from MT5"""
        self.bot.mt5_manager.disconnect()
        messagebox.showinfo("Info", "Disconnected from MetaTrader 5")
    
    def start_bot(self):
        """Start trading bot"""
        if self.bot.start():
            messagebox.showinfo("Success", "Trading bot started")
        else:
            messagebox.showerror("Error", "Failed to start trading bot")
    
    def stop_bot(self):
        """Stop trading bot"""
        if self.bot.stop():
            messagebox.showinfo("Info", "Trading bot stopped")
    
    def analyze_market(self):
        """Analyze market and display signal"""
        signal = self.bot.analyze_market()
        
        if signal:
            # Update signal display
            if signal.signal == TradeSignal.STRONG_BUY:
                signal_text = "🟢🟢 STRONG BUY 🟢🟢"
                color = "green"
            elif signal.signal == TradeSignal.BUY:
                signal_text = "🟢 BUY 🟢"
                color = "green"
            elif signal.signal == TradeSignal.STRONG_SELL:
                signal_text = "🔴🔴 STRONG SELL 🔴🔴"
                color = "red"
            elif signal.signal == TradeSignal.SELL:
                signal_text = "🔴 SELL 🔴"
                color = "red"
            else:
                signal_text = "⚪ NEUTRAL ⚪"
                color = "gray"
            
            self.signal_label.config(text=signal_text, foreground=color)
            
            # Update signal details
            self.signal_details['symbol'].config(text=signal.symbol)
            self.signal_details['timeframe'].config(text=signal.timeframe.name)
            self.signal_details['strength'].config(text=f"{signal.strength:.2%}")
            self.signal_details['confidence'].config(text=f"{signal.confidence:.2%}")
            self.signal_details['entry'].config(text=f"{signal.entry_price:.5f}")
            self.signal_details['sl'].config(text=f"{signal.stop_loss:.5f}")
            self.signal_details['tp'].config(text=f"{signal.take_profit:.5f}")
            
            # Update patterns
            self.patterns_text.delete(1.0, tk.END)
            if signal.patterns:
                for pattern in signal.patterns:
                    self.patterns_text.insert(tk.END, 
                        f"• {pattern.pattern_type.value} "
                        f"(Confidence: {pattern.confidence:.2%})\n"
                        f"  {pattern.description}\n\n"
                    )
            else:
                self.patterns_text.insert(tk.END, "No patterns detected\n")
            
            # Update chart
            self.update_chart()
    
    def send_signal_to_telegram(self):
        """Send current signal to Telegram"""
        if self.bot.current_signal and self.bot.telegram_manager.enabled:
            self.bot.telegram_manager.send_signal_alert(self.bot.current_signal)
            messagebox.showinfo("Success", "Signal sent to Telegram")
        else:
            messagebox.showwarning("Warning", 
                "No signal available or Telegram not configured"
            )
    
    def manual_trade(self, trade_type: str):
        """Execute manual trade"""
        try:
            if not self.bot.mt5_manager.connected:
                messagebox.showerror("Error", "Not connected to MT5")
                return
            
            volume = float(self.volume_var.get())
            symbol = self.symbol_var.get()
            
            if trade_type == "BUY":
                order_type = OrderType.BUY
            else:
                order_type = OrderType.SELL
            
            order_id = self.bot.mt5_manager.place_order(
                symbol=symbol,
                order_type=order_type,
                volume=volume,
                comment="Manual trade"
            )
            
            if order_id:
                messagebox.showinfo("Success", f"Trade executed: {order_id}")
            else:
                messagebox.showerror("Error", "Failed to execute trade")
                
        except ValueError:
            messagebox.showerror("Error", "Invalid volume")
        except Exception as e:
            messagebox.showerror("Error", f"Trade failed: {e}")
    
    def change_symbol(self, event=None):
        """Change trading symbol"""
        self.bot.current_symbol = self.symbol_var.get()
        self.bot.config_manager.config['symbol'] = self.bot.current_symbol
        self.bot.config_manager.save_config(self.bot.config_manager.config)
        self.update_chart()
    
    def change_timeframe(self, event=None):
        """Change timeframe"""
        timeframe = getattr(TimeFrame, self.timeframe_var.get())
        self.bot.current_timeframe = timeframe
        self.bot.config_manager.config['timeframe'] = timeframe.name
        self.bot.config_manager.save_config(self.bot.config_manager.config)
        self.update_chart()
    
    def toggle_auto_trading(self):
        """Toggle auto-trading mode"""
        self.bot.auto_trading = self.auto_trading_var.get()
        self.bot.config_manager.config['auto_trading'] = self.bot.auto_trading
        self.bot.config_manager.save_config(self.bot.config_manager.config)
    
    def toggle_risk_free(self):
        """Toggle risk-free mode"""
        self.bot.risk_free_mode = self.risk_free_var.get()
        self.bot.config_manager.config['risk_free_mode'] = self.bot.risk_free_mode
        self.bot.config_manager.save_config(self.bot.config_manager.config)
    
    def close_selected_position(self):
        """Close selected position"""
        try:
            selection = self.positions_tree.selection()
            if not selection:
                messagebox.showwarning("Warning", "No position selected")
                return
            
            item = self.positions_tree.item(selection[0])
            ticket = int(item['values'][0])
            
            if self.bot.mt5_manager.close_position(ticket):
                messagebox.showinfo("Success", f"Position {ticket} closed")
                self.update_positions()
            else:
                messagebox.showerror("Error", "Failed to close position")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to close position: {e}")
    
    def close_all_positions(self):
        """Close all open positions"""
        try:
            if messagebox.askyesno("Confirm", "Close all positions?"):
                positions = self.bot.mt5_manager.get_open_positions()
                for position in positions:
                    self.bot.mt5_manager.close_position(position.ticket)
                messagebox.showinfo("Success", "All positions closed")
                self.update_positions()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to close positions: {e}")
    
    def clear_log(self):
        """Clear log display"""
        self.log_text.delete(1.0, tk.END)
    
    def save_log(self):
        """Save log to file"""
        try:
            file_path = filedialog.asksaveasfilename(
                title="Save Log",
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            if file_path:
                with open(file_path, 'w') as f:
                    f.write(self.log_text.get(1.0, tk.END))
                messagebox.showinfo("Success", f"Log saved to {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save log: {e}")
    
    def run(self):
        """Run the GUI application"""
        self.root.mainloop()


# ==============================================
# CONFIGURATION DIALOGS
# ==============================================

class MT5ConfigDialog(tk.Toplevel):
    """MT5 Configuration Dialog"""
    
    def __init__(self, parent, config_manager):
        super().__init__(parent)
        self.config_manager = config_manager
        self.mt5_config = config_manager.get_mt5_config()
        
        self.title("MT5 Configuration")
        self.geometry("400x300")
        self.resizable(False, False)
        
        # Make dialog modal
        self.transient(parent)
        self.grab_set()
        
        self.create_widgets()
    
    def create_widgets(self):
        """Create dialog widgets"""
        # Login
        ttk.Label(self, text="Account Number:").grid(row=0, column=0, sticky=tk.W, padx=10, pady=10)
        self.login_var = tk.StringVar(value=str(self.mt5_config.login))
        ttk.Entry(self, textvariable=self.login_var, width=30).grid(row=0, column=1, padx=10, pady=10)
        
        # Password
        ttk.Label(self, text="Password:").grid(row=1, column=0, sticky=tk.W, padx=10, pady=10)
        self.password_var = tk.StringVar(value=self.mt5_config.password)
        ttk.Entry(self, textvariable=self.password_var, width=30, show="*").grid(row=1, column=1, padx=10, pady=10)
        
        # Server
        ttk.Label(self, text="Server:").grid(row=2, column=0, sticky=tk.W, padx=10, pady=10)
        self.server_var = tk.StringVar(value=self.mt5_config.server)
        ttk.Entry(self, textvariable=self.server_var, width=30).grid(row=2, column=1, padx=10, pady=10)
        
        # MT5 Path
        ttk.Label(self, text="MT5 Path:").grid(row=3, column=0, sticky=tk.W, padx=10, pady=10)
        self.path_var = tk.StringVar(value=self.mt5_config.path)
        ttk.Entry(self, textvariable=self.path_var, width=25).grid(row=3, column=1, padx=10, pady=10)
        ttk.Button(self, text="Browse", command=self.browse_path, width=8).grid(row=3, column=2, padx=5)
        
        # Buttons
        button_frame = ttk.Frame(self)
        button_frame.grid(row=4, column=0, columnspan=3, pady=20)
        
        ttk.Button(button_frame, text="Save", command=self.save).pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="Cancel", command=self.destroy).pack(side=tk.LEFT, padx=10)
    
    def browse_path(self):
        """Browse for MT5 installation path"""
        path = filedialog.askdirectory(title="Select MT5 Installation Directory")
        if path:
            self.path_var.set(path)
    
    def save(self):
        """Save configuration"""
        try:
            config = MT5Config(
                login=int(self.login_var.get()),
                password=self.password_var.get(),
                server=self.server_var.get(),
                path=self.path_var.get(),
                timeout=TradingConfig.MT5_TIMEOUT
            )
            
            if self.config_manager.save_mt5_config(config):
                messagebox.showinfo("Success", "MT5 configuration saved")
                self.destroy()
            else:
                messagebox.showerror("Error", "Failed to save configuration")
                
        except ValueError:
            messagebox.showerror("Error", "Invalid account number")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {e}")


class TelegramConfigDialog(tk.Toplevel):
    """Telegram Configuration Dialog"""
    
    def __init__(self, parent, config_manager):
        super().__init__(parent)
        self.config_manager = config_manager
        self.telegram_config = config_manager.get_telegram_config()
        
        self.title("Telegram Configuration")
        self.geometry("400x350")
        self.resizable(False, False)
        
        self.transient(parent)
        self.grab_set()
        
        self.create_widgets()
    
    def create_widgets(self):
        """Create dialog widgets"""
        # Enable Telegram
        self.enabled_var = tk.BooleanVar(value=self.telegram_config.enabled)
        ttk.Checkbutton(self, text="Enable Telegram Bot", 
                       variable=self.enabled_var).grid(row=0, column=0, columnspan=2, 
                                                      sticky=tk.W, padx=10, pady=10)
        
        # Bot Token
        ttk.Label(self, text="Bot Token:").grid(row=1, column=0, sticky=tk.W, padx=10, pady=5)
        self.token_var = tk.StringVar(value=self.telegram_config.token)
        ttk.Entry(self, textvariable=self.token_var, width=40).grid(row=1, column=1, padx=10, pady=5)
        
        # Chat ID
        ttk.Label(self, text="Chat ID:").grid(row=2, column=0, sticky=tk.W, padx=10, pady=5)
        self.chat_id_var = tk.StringVar(value=self.telegram_config.chat_id)
        ttk.Entry(self, textvariable=self.chat_id_var, width=40).grid(row=2, column=1, padx=10, pady=5)
        
        # Notification settings
        ttk.Label(self, text="Notifications:").grid(row=3, column=0, sticky=tk.W, padx=10, pady=10)
        
        self.alerts_var = tk.BooleanVar(value=self.telegram_config.send_alerts)
        ttk.Checkbutton(self, text="Send Trading Alerts", 
                       variable=self.alerts_var).grid(row=4, column=0, columnspan=2, 
                                                     sticky=tk.W, padx=30, pady=2)
        
        self.errors_var = tk.BooleanVar(value=self.telegram_config.send_errors)
        ttk.Checkbutton(self, text="Send Error Alerts", 
                       variable=self.errors_var).grid(row=5, column=0, columnspan=2, 
                                                     sticky=tk.W, padx=30, pady=2)
        
        self.positions_var = tk.BooleanVar(value=self.telegram_config.send_positions)
        ttk.Checkbutton(self, text="Send Position Updates", 
                       variable=self.positions_var).grid(row=6, column=0, columnspan=2, 
                                                        sticky=tk.W, padx=30, pady=2)
        
        # Buttons
        button_frame = ttk.Frame(self)
        button_frame.grid(row=7, column=0, columnspan=2, pady=20)
        
        ttk.Button(button_frame, text="Save", command=self.save).pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="Cancel", command=self.destroy).pack(side=tk.LEFT, padx=10)
    
    def save(self):
        """Save configuration"""
        try:
            config = TelegramConfig(
                token=self.token_var.get(),
                chat_id=self.chat_id_var.get(),
                enabled=self.enabled_var.get(),
                send_alerts=self.alerts_var.get(),
                send_errors=self.errors_var.get(),
                send_positions=self.positions_var.get()
            )
            
            if self.config_manager.save_telegram_config(config):
                messagebox.showinfo("Success", "Telegram configuration saved")
                self.destroy()
            else:
                messagebox.showerror("Error", "Failed to save configuration")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {e}")


class TradingConfigDialog(tk.Toplevel):
    """Trading Configuration Dialog"""
    
    def __init__(self, parent, config_manager):
        super().__init__(parent)
        self.config_manager = config_manager
        self.config = config_manager.config
        
        self.title("Trading Configuration")
        self.geometry("400x400")
        self.resizable(False, False)
        
        self.transient(parent)
        self.grab_set()
        
        self.create_widgets()
    
    def create_widgets(self):
        """Create dialog widgets"""
        # Symbol
        ttk.Label(self, text="Default Symbol:").grid(row=0, column=0, sticky=tk.W, padx=10, pady=10)
        self.symbol_var = tk.StringVar(value=self.config.get('symbol', TradingConfig.DEFAULT_SYMBOL))
        symbol_combo = ttk.Combobox(self, textvariable=self.symbol_var,
                                   values=["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "BTCUSD"],
                                   state="readonly", width=15)
        symbol_combo.grid(row=0, column=1, padx=10, pady=10)
        
        # Timeframe
        ttk.Label(self, text="Default Timeframe:").grid(row=1, column=0, sticky=tk.W, padx=10, pady=10)
        self.timeframe_var = tk.StringVar(value=self.config.get('timeframe', 'M15'))
        timeframe_combo = ttk.Combobox(self, textvariable=self.timeframe_var,
                                      values=[tf.name for tf in TimeFrame], 
                                      state="readonly", width=15)
        timeframe_combo.grid(row=1, column=1, padx=10, pady=10)
        
        # Default Volume
        ttk.Label(self, text="Default Volume:").grid(row=2, column=0, sticky=tk.W, padx=10, pady=10)
        self.volume_var = tk.StringVar(value=str(self.config.get('volume', TradingConfig.DEFAULT_VOLUME)))
        ttk.Entry(self, textvariable=self.volume_var, width=15).grid(row=2, column=1, padx=10, pady=10)
        
        # Max Spread
        ttk.Label(self, text="Max Spread (pips):").grid(row=3, column=0, sticky=tk.W, padx=10, pady=10)
        self.spread_var = tk.StringVar(value=str(self.config.get('max_spread', TradingConfig.MAX_SPREAD)))
        ttk.Entry(self, textvariable=self.spread_var, width=15).grid(row=3, column=1, padx=10, pady=10)
        
        # Max Positions
        ttk.Label(self, text="Max Positions:").grid(row=4, column=0, sticky=tk.W, padx=10, pady=10)
        self.positions_var = tk.StringVar(value=str(self.config.get('max_positions', TradingConfig.MAX_POSITIONS)))
        ttk.Entry(self, textvariable=self.positions_var, width=15).grid(row=4, column=1, padx=10, pady=10)
        
        # Buttons
        button_frame = ttk.Frame(self)
        button_frame.grid(row=5, column=0, columnspan=2, pady=20)
        
        ttk.Button(button_frame, text="Save", command=self.save).pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="Cancel", command=self.destroy).pack(side=tk.LEFT, padx=10)
    
    def save(self):
        """Save configuration"""
        try:
            self.config['symbol'] = self.symbol_var.get()
            self.config['timeframe'] = self.timeframe_var.get()
            self.config['volume'] = float(self.volume_var.get())
            self.config['max_spread'] = int(self.spread_var.get())
            self.config['max_positions'] = int(self.positions_var.get())
            
            if self.config_manager.save_config(self.config):
                messagebox.showinfo("Success", "Trading configuration saved")
                self.destroy()
            else:
                messagebox.showerror("Error", "Failed to save configuration")
                
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid value: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {e}")


class RiskConfigDialog(tk.Toplevel):
    """Risk Management Configuration Dialog"""
    
    def __init__(self, parent, config_manager):
        super().__init__(parent)
        self.config_manager = config_manager
        self.config = config_manager.config
        
        self.title("Risk Management Configuration")
        self.geometry("400x300")
        self.resizable(False, False)
        
        self.transient(parent)
        self.grab_set()
        
        self.create_widgets()
    
    def create_widgets(self):
        """Create dialog widgets"""
        # Max Risk Percent
        ttk.Label(self, text="Max Risk per Trade (%):").grid(row=0, column=0, sticky=tk.W, padx=10, pady=10)
        self.risk_var = tk.StringVar(value=str(self.config.get('max_risk_percent', TradingConfig.MAX_RISK_PERCENT)))
        ttk.Entry(self, textvariable=self.risk_var, width=15).grid(row=0, column=1, padx=10, pady=10)
        
        # Max Daily Loss
        ttk.Label(self, text="Max Daily Loss (%):").grid(row=1, column=0, sticky=tk.W, padx=10, pady=10)
        self.daily_loss_var = tk.StringVar(value=str(self.config.get('max_daily_loss', TradingConfig.MAX_DAILY_LOSS)))
        ttk.Entry(self, textvariable=self.daily_loss_var, width=15).grid(row=1, column=1, padx=10, pady=10)
        
        # Slippage
        ttk.Label(self, text="Slippage (pips):").grid(row=2, column=0, sticky=tk.W, padx=10, pady=10)
        self.slippage_var = tk.StringVar(value=str(TradingConfig.SLIPPAGE))
        ttk.Entry(self, textvariable=self.slippage_var, width=15).grid(row=2, column=1, padx=10, pady=10)
        
        # Risk-Free Mode
        self.risk_free_var = tk.BooleanVar(value=self.config.get('risk_free_mode', True))
        ttk.Checkbutton(self, text="Enable Risk-Free Mode (no real trades)", 
                       variable=self.risk_free_var).grid(row=3, column=0, columnspan=2, 
                                                        sticky=tk.W, padx=10, pady=10)
        
        # Buttons
        button_frame = ttk.Frame(self)
        button_frame.grid(row=4, column=0, columnspan=2, pady=20)
        
        ttk.Button(button_frame, text="Save", command=self.save).pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="Cancel", command=self.destroy).pack(side=tk.LEFT, padx=10)
    
    def save(self):
        """Save configuration"""
        try:
            self.config['max_risk_percent'] = float(self.risk_var.get())
            self.config['max_daily_loss'] = float(self.daily_loss_var.get())
            self.config['risk_free_mode'] = self.risk_free_var.get()
            
            if self.config_manager.save_config(self.config):
                messagebox.showinfo("Success", "Risk configuration saved")
                self.destroy()
            else:
                messagebox.showerror("Error", "Failed to save configuration")
                
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid value: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {e}")


# ==============================================
# MAIN APPLICATION
# ==============================================

def main():
    """Main application entry point"""
    print(f"\n{'='*60}")
    print(f"{TradingConfig.APP_NAME} v{TradingConfig.VERSION}")
    print(f"{'='*60}\n")
    
    try:
        # Create trading bot instance
        bot = TradingBot()
        
        # Create and run GUI
        gui = TradingGUI(bot)
        gui.run()
        
    except KeyboardInterrupt:
        print("\n\nBot stopped by user")
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nShutting down...")


if __name__ == "__main__":
    main()