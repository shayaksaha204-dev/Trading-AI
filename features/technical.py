"""
Technical Features — 50+ Technical Indicators
================================================
Computes trend, momentum, volatility, volume, and pattern indicators.
"""

import numpy as np
import pandas as pd
from typing import List
from loguru import logger
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import cfg


class TechnicalFeatures:
    """Computes 50+ technical indicators on OHLCV data."""

    def __init__(self):
        self.cfg = cfg.features

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        required = ["Open", "High", "Low", "Close"]
        if not all(c in df.columns for c in required):
            logger.error(f"Missing columns. Have: {df.columns.tolist()}")
            return df
        # Check for meaningful volume data (forex has Volume column but all zeros)
        has_volume = "Volume" in df.columns and (df["Volume"] > 0).any()
        df = self._trend(df)
        df = self._momentum(df, has_volume=has_volume)
        df = self._volatility(df)
        if has_volume:
            df = self._volume(df)
        df = self._patterns(df)
        df = self._custom(df)
        df = self._rolling_stats(df)
        return df

    def _trend(self, df):
        c = df["Close"]; h = df["High"]; l = df["Low"]
        for w in self.cfg.sma_windows:
            df[f"SMA_{w}"] = c.rolling(w).mean()
            df[f"Close_SMA_{w}_ratio"] = c / df[f"SMA_{w}"]
        for w in self.cfg.ema_windows:
            df[f"EMA_{w}"] = c.ewm(span=w, adjust=False).mean()
            df[f"Close_EMA_{w}_ratio"] = c / df[f"EMA_{w}"]
        ema1 = c.ewm(span=20, adjust=False).mean()
        ema2 = ema1.ewm(span=20, adjust=False).mean()
        ema3 = ema2.ewm(span=20, adjust=False).mean()
        df["DEMA_20"] = 2*ema1 - ema2
        df["TEMA_20"] = 3*ema1 - 3*ema2 + ema3
        nh = h.rolling(9).max(); nl = l.rolling(9).min()
        df["Ichimoku_Tenkan"] = (nh+nl)/2
        nh26 = h.rolling(26).max(); nl26 = l.rolling(26).min()
        df["Ichimoku_Kijun"] = (nh26+nl26)/2
        df["Ichimoku_Senkou_A"] = ((df["Ichimoku_Tenkan"]+df["Ichimoku_Kijun"])/2).shift(26)
        nh52 = h.rolling(52).max(); nl52 = l.rolling(52).min()
        df["Ichimoku_Senkou_B"] = ((nh52+nl52)/2).shift(26)
        df = self._adx(df)
        atr = self._atr(df)
        hl2 = (h+l)/2
        df["Supertrend_upper"] = hl2 + 3.0*atr
        df["Supertrend_lower"] = hl2 - 3.0*atr
        ap = 25
        df["Aroon_Up"] = h.rolling(ap+1).apply(lambda x: x.argmax()/ap*100, raw=True)
        df["Aroon_Down"] = l.rolling(ap+1).apply(lambda x: x.argmin()/ap*100, raw=True)
        df["Aroon_Osc"] = df["Aroon_Up"] - df["Aroon_Down"]
        if "SMA_5" in df.columns and "SMA_20" in df.columns:
            df["MA_Cross_5_20"] = (df["SMA_5"]>df["SMA_20"]).astype(int)
        if "SMA_20" in df.columns and "SMA_50" in df.columns:
            df["MA_Cross_20_50"] = (df["SMA_20"]>df["SMA_50"]).astype(int)
        return df

    def _momentum(self, df, has_volume=True):
        c = df["Close"]; h = df["High"]; l = df["Low"]
        delta = c.diff()
        gain = delta.where(delta>0, 0.0); loss = (-delta).where(delta<0, 0.0)
        for w in [7, 14, 21]:
            ag = gain.rolling(w).mean(); al = loss.rolling(w).mean()
            df[f"RSI_{w}"] = 100 - 100/(1 + ag/(al+1e-10))
        ef = c.ewm(span=self.cfg.macd_fast, adjust=False).mean()
        es = c.ewm(span=self.cfg.macd_slow, adjust=False).mean()
        df["MACD"] = ef - es
        df["MACD_Signal"] = df["MACD"].ewm(span=self.cfg.macd_signal, adjust=False).mean()
        df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
        ln = l.rolling(self.cfg.stoch_window).min()
        hn = h.rolling(self.cfg.stoch_window).max()
        df["Stoch_K"] = 100*(c-ln)/(hn-ln+1e-10)
        df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()
        df["Williams_R"] = -100*(hn-c)/(hn-ln+1e-10)
        tp = (h+l+c)/3; ma_tp = tp.rolling(self.cfg.cci_window).mean()
        md_tp = tp.rolling(self.cfg.cci_window).apply(lambda x: np.abs(x-x.mean()).mean(), raw=True)
        df["CCI"] = (tp-ma_tp)/(0.015*md_tp+1e-10)
        for p in [5,10,20]:
            df[f"ROC_{p}"] = (c/c.shift(p)-1)*100
        if has_volume:
            mf = tp*df["Volume"]
            pmf = mf.where(tp>tp.shift(1),0); nmf = mf.where(tp<tp.shift(1),0)
            df["MFI"] = 100 - 100/(1 + pmf.rolling(14).sum()/(nmf.rolling(14).sum()+1e-10))
        pc = c.diff()
        dsp = pc.ewm(span=25).mean().ewm(span=13).mean()
        dsa = pc.abs().ewm(span=25).mean().ewm(span=13).mean()
        df["TSI"] = 100*dsp/(dsa+1e-10)
        df["Momentum_10"] = c - c.shift(10)
        df["Momentum_20"] = c - c.shift(20)
        return df

    def _volatility(self, df):
        c = df["Close"]; h = df["High"]; l = df["Low"]
        sma = c.rolling(self.cfg.bb_window).mean()
        std = c.rolling(self.cfg.bb_window).std()
        df["BB_Upper"] = sma + self.cfg.bb_std*std
        df["BB_Lower"] = sma - self.cfg.bb_std*std
        df["BB_Width"] = (df["BB_Upper"]-df["BB_Lower"])/(sma+1e-10)
        df["BB_Percent"] = (c-df["BB_Lower"])/(df["BB_Upper"]-df["BB_Lower"]+1e-10)
        atr = self._atr(df)
        df["ATR"] = atr; df["ATR_Percent"] = atr/(c+1e-10)*100
        ema20 = c.ewm(span=20, adjust=False).mean()
        df["Keltner_Upper"] = ema20 + 2.0*atr
        df["Keltner_Lower"] = ema20 - 2.0*atr
        df["Donchian_Upper"] = h.rolling(20).max()
        df["Donchian_Lower"] = l.rolling(20).min()
        ret = c.pct_change()
        for w in [10,20,60]:
            df[f"HV_{w}"] = ret.rolling(w).std()*np.sqrt(252)
        df["Vol_Ratio"] = df.get("HV_10",ret.rolling(10).std())/(df.get("HV_60",ret.rolling(60).std())+1e-10)
        df["Daily_Range"] = (h-l)/(c+1e-10)
        return df

    def _volume(self, df):
        c = df["Close"]; h = df["High"]; l = df["Low"]; v = df["Volume"]
        obv = (np.sign(c.diff())*v).fillna(0).cumsum()
        df["OBV"] = obv; df["OBV_SMA_20"] = obv.rolling(20).mean()
        tp = (h+l+c)/3
        df["VWAP"] = (tp*v).cumsum()/(v.cumsum()+1e-10)
        mfv = ((c-l)-(h-c))/(h-l+1e-10)*v
        df["CMF"] = mfv.rolling(20).sum()/(v.rolling(20).sum()+1e-10)
        df["AD_Line"] = (((c-l)-(h-c))/(h-l+1e-10)*v).cumsum()
        df["Force_Index"] = c.diff()*v
        df["Force_Index_13"] = df["Force_Index"].ewm(span=13, adjust=False).mean()
        df["Volume_ROC"] = (v/v.shift(10)-1)*100
        df["Volume_SMA_Ratio"] = v/(v.rolling(20).mean()+1e-10)
        return df

    def _patterns(self, df):
        o=df["Open"]; h=df["High"]; l=df["Low"]; c=df["Close"]
        body = c-o; ba = body.abs()
        us = h - pd.concat([o,c],axis=1).max(axis=1)
        ls = pd.concat([o,c],axis=1).min(axis=1) - l
        tr = h-l+1e-10
        df["Pattern_Doji"] = (ba/tr < 0.1).astype(int)
        df["Pattern_Hammer"] = ((ls>2*ba)&(us<ba*0.3)&(body>0)).astype(int)
        df["Pattern_BullEngulf"] = ((body>0)&(body.shift(1)<0)&(c>o.shift(1))&(o<c.shift(1))).astype(int)
        df["Pattern_BearEngulf"] = ((body<0)&(body.shift(1)>0)&(o>c.shift(1))&(c<o.shift(1))).astype(int)
        df["Pattern_MorningStar"] = ((body.shift(2)<0)&(ba.shift(1)<ba.shift(2)*0.3)&(body>0)).astype(int)
        df["Pattern_EveningStar"] = ((body.shift(2)>0)&(ba.shift(1)<ba.shift(2)*0.3)&(body<0)).astype(int)
        df["Pattern_ShootingStar"] = ((us>2*ba)&(ls<ba*0.3)&(body<0)).astype(int)
        df["Pattern_3WhiteSoldiers"] = ((body>0)&(body.shift(1)>0)&(body.shift(2)>0)&(c>c.shift(1))&(c.shift(1)>c.shift(2))).astype(int)
        df["Pattern_3BlackCrows"] = ((body<0)&(body.shift(1)<0)&(body.shift(2)<0)&(c<c.shift(1))&(c.shift(1)<c.shift(2))).astype(int)
        return df

    def _custom(self, df):
        c = df["Close"]; ret = c.pct_change()
        rh = c.rolling(52).max(); rl = c.rolling(52).min()
        df["Price_Position_52w"] = (c-rl)/(rh-rl+1e-10)
        df["Dist_52w_High"] = (c/rh-1)*100
        df["Dist_52w_Low"] = (c/rl-1)*100
        df["Gap"] = (df["Open"]/c.shift(1)-1)*100
        df["Intraday_Return"] = (c/df["Open"]-1)*100
        vol = ret.rolling(20).std()
        df["Vol_Adj_Momentum"] = ret.rolling(10).mean()/(vol+1e-10)
        sma20 = c.rolling(20).mean(); std20 = c.rolling(20).std()
        df["Z_Score"] = (c-sma20)/(std20+1e-10)
        df["Trend_Strength"] = (c.rolling(5).mean()/c.rolling(20).mean()-1)*100
        df["Higher_High"] = (df["High"]>df["High"].shift(1)).astype(int)
        df["Lower_Low"] = (df["Low"]<df["Low"].shift(1)).astype(int)
        return df

    def _rolling_stats(self, df):
        ret = df["Close"].pct_change()
        for w in self.cfg.rolling_windows:
            df[f"Ret_Mean_{w}"] = ret.rolling(w).mean()
            df[f"Ret_Std_{w}"] = ret.rolling(w).std()
            df[f"Ret_Skew_{w}"] = ret.rolling(w).skew()
            df[f"Ret_Kurt_{w}"] = ret.rolling(w).kurt()
        return df

    def _atr(self, df, window=None):
        window = window or self.cfg.atr_window
        tr = pd.concat([df["High"]-df["Low"], (df["High"]-df["Close"].shift(1)).abs(), (df["Low"]-df["Close"].shift(1)).abs()], axis=1).max(axis=1)
        return tr.rolling(window).mean()

    def _adx(self, df, window=None):
        window = window or self.cfg.adx_window
        pdm = df["High"].diff(); mdm = -df["Low"].diff()
        pdm = pdm.where((pdm>mdm)&(pdm>0),0)
        mdm = mdm.where((mdm>pdm)&(mdm>0),0)
        atr = self._atr(df, window)
        pdi = 100*(pdm.rolling(window).mean()/(atr+1e-10))
        mdi = 100*(mdm.rolling(window).mean()/(atr+1e-10))
        dx = 100*((pdi-mdi).abs()/(pdi+mdi+1e-10))
        df["ADX"] = dx.rolling(window).mean()
        df["Plus_DI"] = pdi; df["Minus_DI"] = mdi
        return df

    def get_feature_names(self) -> List[str]:
        dummy = pd.DataFrame({"Open": np.random.randn(200)+100, "High": np.random.randn(200)+101, "Low": np.random.randn(200)+99, "Close": np.random.randn(200)+100, "Volume": np.abs(np.random.randn(200))*1e6})
        result = self.compute_all(dummy)
        skip = {"Open","High","Low","Close","Volume","Returns","LogReturns","Ticker","Category"}
        return [c for c in result.columns if c not in skip]


if __name__ == "__main__":
    tf = TechnicalFeatures()
    names = tf.get_feature_names()
    print(f"\nTotal technical features: {len(names)}")
    for i, n in enumerate(names): print(f"  {i+1:3d}. {n}")
