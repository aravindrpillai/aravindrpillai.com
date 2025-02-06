import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBClassifier  # Advanced model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.volatility import BollingerBands
from datetime import datetime, timedelta
import json

class StockEngine:
    def __init__(self, ticker, days_back=365):
        self.ticker = ticker + ".NS"
        self.end_date = datetime.today().strftime('%Y-%m-%d')
        self.start_date = (datetime.today() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        self.data = self._fetch_data()
        self.features = self._calculate_technical_indicators()
        self.model, self.accuracy = self._train_model()

    def _fetch_data(self):
        """Fetch historical stock data from Yahoo Finance."""
        data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        if data.empty:
            raise ValueError(f"No data found for {self.ticker} between {self.start_date} and {self.end_date}.")
        return data

    def _calculate_technical_indicators(self):
        """Calculate technical indicators like RSI, MACD, Bollinger Bands, EMA, and SMA."""
        data = self.data.copy()

        # Ensure 'Close' is a 1D Pandas Series
        close_prices = data['Close'].squeeze()

        # RSI (Relative Strength Index)
        rsi = RSIIndicator(close_prices, window=14)
        data['rsi'] = rsi.rsi()

        # MACD (Moving Average Convergence Divergence)
        macd = MACD(close_prices, window_slow=26, window_fast=12, window_sign=9)
        data['macd'] = macd.macd()
        data['macd_signal'] = macd.macd_signal()
        data['macd_diff'] = macd.macd_diff()

        # Bollinger Bands
        bollinger = BollingerBands(close_prices, window=20, window_dev=2)
        data['bollinger_hband'] = bollinger.bollinger_hband()
        data['bollinger_lband'] = bollinger.bollinger_lband()

        # EMA (Exponential Moving Average)
        data['ema_20'] = EMAIndicator(close_prices, window=20).ema_indicator()

        # SMA (Simple Moving Average)
        data['sma_50'] = SMAIndicator(close_prices, window=50).sma_indicator()
        data['sma_200'] = SMAIndicator(close_prices, window=200).sma_indicator()

        # Drop NaN values created by indicators
        data = data.dropna()
        return data

    def _prepare_features(self):
        """Prepare features and target variable for the model."""
        features = self.features.copy()
        features['Target'] = np.where(features['Close'].shift(-1) > features['Close'], 1, 0)
        features = features.dropna()
        X = features[['rsi', 'macd', 'macd_signal', 'macd_diff', 'bollinger_hband', 'bollinger_lband', 'ema_20', 'sma_50', 'sma_200']]
        y = features['Target']
        return X, y

    def _train_model(self):
        """Train an XGBoost model and return both model and accuracy."""
        X, y = self._prepare_features()
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Insufficient data for training")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = XGBClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.2f}")
        return model, accuracy

    def predict_today(self):
        """Generate prediction with dynamic JSON output."""
        latest_data = self.features.iloc[-1:]
        current_price = float(latest_data['Close'].values[0])

        # Prepare JSON structure
        output = {
            "ticker": self.ticker,
            "status": True,
            "prediction_accuracy": float(self.accuracy),
            "current_price": current_price,
            "decision": "",
            "stop_loss": float(latest_data['bollinger_lband'].values[0]),
            "target_price": float(latest_data['bollinger_hband'].values[0]),
            "technical_indicators": [],
            "analysis": "",
            "detailed_description": ""
        }

        # Populate technical indicators
        indicators = {
            "rsi": {"current_value": float(latest_data['rsi'].values[0])},
            "macd": {"current_value": float(latest_data['macd'].values[0])},
            "macd_signal": {"current_value": float(latest_data['macd_signal'].values[0])},
            "macd_diff": {"current_value": float(latest_data['macd_diff'].values[0])},
            "bollinger_hband": {"current_value": float(latest_data['bollinger_hband'].values[0])},
            "bollinger_lband": {"current_value": float(latest_data['bollinger_lband'].values[0])},
            "ema_20": {"current_value": float(latest_data['ema_20'].values[0])},
            "sma_50": {"current_value": float(latest_data['sma_50'].values[0])},
            "sma_200": {"current_value": float(latest_data['sma_200'].values[0])},
        }

        for indicator, values in indicators.items():
            output["technical_indicators"].append({
                "indicator": indicator,
                "current_value": values["current_value"],
                "analysis": self._get_analysis(indicator, values["current_value"], current_price)
            })

        # Final decision
        prediction = self.model.predict(latest_data[['rsi', 'macd', 'macd_signal', 'macd_diff', 'bollinger_hband', 'bollinger_lband', 'ema_20', 'sma_50', 'sma_200']])
        decision = "Buy" if prediction[0] == 1 else "Sell"
        output["decision"] = decision

        # Generate analysis
        output["analysis"] = self._generate_analysis(output, decision, indicators, current_price)

        # Generate detailed description
        output["detailed_description"] = (
            f"Model suggests to {decision}, because:\n"
            + "\n".join([f"- {indicator['analysis']}" for indicator in output["technical_indicators"]])
            + f"\n\nPrediction Confidence: {self.accuracy:.2%}"
            + f"\nStop-loss: {output['stop_loss']:.2f}"
            + f"\nTarget: {output['target_price']:.2f}"
        )

        return output

    def _get_analysis(self, indicator, current_value, close_price):
        """Generate analysis for each indicator."""
        if indicator == "rsi":
            if current_value > 70:
                return "Overbought (Sell signal)"
            elif current_value < 30:
                return "Oversold (Buy signal)"
            else:
                return "Neutral"
        elif indicator == "macd":
            if current_value > 0:
                return "Bullish momentum (Buy signal)"
            else:
                return "Bearish momentum (Sell signal)"
        elif indicator == "bollinger_hband":
            if close_price > current_value:
                return "Price above upper Bollinger Band (Potential resistance)"
            else:
                return "Price below upper Bollinger Band"
        elif indicator == "bollinger_lband":
            if close_price < current_value:
                return "Price below lower Bollinger Band (Potential support)"
            else:
                return "Price above lower Bollinger Band"
        elif indicator == "ema_20":
            if close_price > current_value:
                return "Price above EMA 20 (Bullish)"
            else:
                return "Price below EMA 20 (Bearish)"
        elif indicator == "sma_50":
            if close_price > current_value:
                return "Price above SMA 50 (Bullish)"
            else:
                return "Price below SMA 50 (Bearish)"
        elif indicator == "sma_200":
            if close_price > current_value:
                return "Price above SMA 200 (Bullish)"
            else:
                return "Price below SMA 200 (Bearish)"
        return ""

    def _generate_analysis(self, output, decision, indicators, current_price):
        """Generate detailed analysis explaining the decision."""
        analysis = "Why the Model Suggests " + decision + ":\n"

        # Bollinger Bands Resistance
        if indicators["bollinger_hband"]["current_value"] < current_price:
            analysis += (
                "\nBollinger Bands Resistance:\n"
                "The price is near the upper Bollinger Band ({}), which often acts as a resistance level. "
                "This suggests that the stock might face selling pressure soon.\n"
            ).format(indicators["bollinger_hband"]["current_value"])

        # Model Decision Explanation
        analysis += (
            "\nXGBoost Model Decision:\n"
            "The model considers all indicators together, not just individual ones. "
            "While some indicators (like MACD and moving averages) are bullish, the combination of all features "
            "might have led the model to predict a '{}'. The model might be weighing the Bollinger Bands resistance "
            "more heavily than the bullish signals.\n"
        ).format(decision)

        # Prediction Confidence
        analysis += (
            "\nPrediction Confidence:\n"
            "The model's accuracy is {:.2%}, which means it is not 100% reliable. "
            "There is a chance the prediction could be incorrect.\n"
        ).format(output["prediction_accuracy"])

        return analysis

# Example usage
if __name__ == "__main__":
    ticker = "AARVEEDEN"  # Example: AARVEEDEN
    days_back = 365  # Example: 1 year
    predictor = StockEngine(ticker, days_back=days_back)
    result = predictor.predict_today()
    print(json.dumps(result, indent=4))