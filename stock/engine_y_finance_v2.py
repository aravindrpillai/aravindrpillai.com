import yfinance as yf
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.volatility import BollingerBands
from datetime import datetime, timedelta
from imblearn.over_sampling import SMOTE  # For oversampling

class StockEngine:
    def __init__(self, ticker, exchange, days_back=365):
        self.ticker = ticker + "."+ exchange
        self.end_date = datetime.today().strftime('%Y-%m-%d')  # Today's date
        self.start_date = (datetime.today() - timedelta(days=days_back)).strftime('%Y-%m-%d')  # Dynamic start date
        self.data = self._fetch_data()
        self.features = self._calculate_technical_indicators()
        self.model, self.accuracy = self._train_model()

    def _fetch_data(self):
        data = yf.download(self.ticker, start=self.start_date, end=self.end_date, progress=False)
        if data.empty:
            raise ValueError(f"No data found for {self.ticker} between {self.start_date} and {self.end_date}.")
        return data

    def _calculate_technical_indicators(self):
        data = self.data.copy()
        close_prices = data['Close'].squeeze()

        # RSI
        rsi = RSIIndicator(close_prices, window=14)
        data['rsi'] = rsi.rsi()

        # MACD
        macd = MACD(close_prices, window_slow=26, window_fast=12, window_sign=9)
        data['macd'] = macd.macd()
        data['macd_signal'] = macd.macd_signal()
        data['macd_diff'] = macd.macd_diff()

        # Bollinger Bands
        bollinger = BollingerBands(close_prices, window=20, window_dev=2)
        data['bollinger_hband'] = bollinger.bollinger_hband()
        data['bollinger_lband'] = bollinger.bollinger_lband()

        # EMA
        data['ema_20'] = EMAIndicator(close_prices, window=20).ema_indicator()

        # SMA
        data['sma_50'] = SMAIndicator(close_prices, window=50).sma_indicator()
        data['sma_200'] = SMAIndicator(close_prices, window=200).sma_indicator()

        # Drop NaN values
        data = data.dropna()
        return data

    def _prepare_features(self):
        features = self.features.copy()
        # Define target based on a 1% price increase
        features['Target'] = np.where(features['Close'].shift(-1) > features['Close'] * 1.01, 1, 0)
        features = features.dropna()
        X = features[['rsi', 'macd', 'macd_signal', 'macd_diff', 'bollinger_hband', 'bollinger_lband', 'ema_20', 'sma_50', 'sma_200']]
        y = features['Target']
        return X, y

    def _train_model(self):
        X, y = self._prepare_features()
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Not enough data to train the model. Please check the date range and stock ticker.")
        
        # Oversample the minority class using SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

        # Train the model
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return model, accuracy

    def _get_benchmark_and_analysis(self, indicator, current_value, close_price=None):
        benchmark = None
        analysis = ""

        if indicator == "rsi":
            benchmark = 50  # Neutral level
            if current_value > 70:
                analysis = "Overbought (Sell signal)"
            elif current_value < 40:  # Adjusted threshold for oversold
                analysis = "Oversold (Buy signal)"
            else:
                analysis = "Neutral"
        elif indicator == "macd":
            benchmark = 0  # MACD crossover level
            if current_value > 0:
                analysis = "Bullish momentum (Buy signal)"
            else:
                analysis = "Bearish momentum (Sell signal)"
        elif indicator == "bollinger_hband":
            benchmark = close_price  # Compare with current price
            if current_value > close_price:
                analysis = "Price near upper band (Potential resistance)"
            else:
                analysis = "Price below upper band"
        elif indicator == "bollinger_lband":
            benchmark = close_price  # Compare with current price
            if current_value < close_price:
                analysis = "Price near lower band (Potential support)"
            else:
                analysis = "Price above lower band"
        elif indicator == "ema_20":
            benchmark = close_price  # Compare with current price
            if current_value < close_price:
                analysis = "Price above EMA 20 (Bullish)"
            else:
                analysis = "Price below EMA 20 (Bearish)"
        elif indicator == "sma_50":
            benchmark = close_price  # Compare with current price
            if current_value < close_price:
                analysis = "Price above SMA 50 (Bullish)"
            else:
                analysis = "Price below SMA 50 (Bearish)"
        elif indicator == "sma_200":
            benchmark = close_price  # Compare with current price
            if current_value < close_price:
                analysis = "Price above SMA 200 (Bullish)"
            else:
                analysis = "Price below SMA 200 (Bearish)"

        return benchmark, analysis

    def predict_today(self):
        latest_data = self.features.iloc[-1:]
        current_price = float(latest_data['Close'].iloc[0,0])

        # Prepare JSON structure
        output = {
            "ticker": self.ticker,
            "status": True,
            "prediction_accuracy": float(self.accuracy),
            "current_price": current_price,
            "decision": "",
            "stop_loss": float(latest_data['bollinger_lband'].iloc[-1]),
            "target_price": float(latest_data['bollinger_hband'].iloc[-1]),
            "technical_indicators": [],
            "detailed_description": ""
        }

        # Populate technical indicators
        indicators = {
            "rsi": {"current_value": float(latest_data['rsi'].iloc[-1])},
            "macd": {"current_value": float(latest_data['macd'].iloc[-1])},
            "macd_signal": {"current_value": float(latest_data['macd_signal'].iloc[-1])},
            "macd_diff": {"current_value": float(latest_data['macd_diff'].iloc[-1])},
            "bollinger_hband": {"current_value": float(latest_data['bollinger_hband'].iloc[-1])},
            "bollinger_lband": {"current_value": float(latest_data['bollinger_lband'].iloc[-1])},
            "ema_20": {"current_value": float(latest_data['ema_20'].iloc[-1])},
            "sma_50": {"current_value": float(latest_data['sma_50'].iloc[-1])},
            "sma_200": {"current_value": float(latest_data['sma_200'].iloc[-1])},
        }

        for indicator, values in indicators.items():
            benchmark, analysis = self._get_benchmark_and_analysis(
                indicator, values["current_value"], current_price
            )
            output["technical_indicators"].append({
                "indicator": indicator,
                "current_value": values["current_value"],
                "benchmark": benchmark,
                "analysis": analysis
            })

        # Generate dynamic description
        reasons = []
        rsi = indicators['rsi']['current_value']
        macd = indicators['macd']['current_value']
        sma_200 = indicators['sma_200']['current_value']
        bollinger_hband = indicators['bollinger_hband']['current_value']
        bollinger_lband = indicators['bollinger_lband']['current_value']

        if rsi < 40:  # Adjusted threshold for oversold
            reasons.append(f"Oversold (RSI: {rsi:.2f} < 40)")
        elif rsi > 70:
            reasons.append(f"Overbought (RSI: {rsi:.2f} > 70)")

        if macd > 0:
            reasons.append(f"Bullish MACD crossover (MACD: {macd:.2f} > 0)")
        else:
            reasons.append(f"Bearish MACD crossover (MACD: {macd:.2f} < 0)")

        if current_price > sma_200:
            reasons.append(f"Price above 200-day SMA ({current_price:.2f} > {sma_200:.2f})")
        else:
            reasons.append(f"Price below 200-day SMA ({current_price:.2f} < {sma_200:.2f})")

        if current_price > bollinger_hband:
            reasons.append("Price above upper Bollinger Band")
        elif current_price < bollinger_lband:
            reasons.append("Price below lower Bollinger Band")

        # Final decision
        prediction = self.model.predict(latest_data[['rsi', 'macd', 'macd_signal', 'macd_diff', 'bollinger_hband', 'bollinger_lband', 'ema_20', 'sma_50', 'sma_200']])
        decision = "Buy" if prediction[0] == 1 else "Sell"
        output["decision"] = decision

        # Build detailed description
        output["detailed_description"] = (
            f"Model suggests to {decision}, because:\n"
            + "\n".join([f"- {reason}" for reason in reasons])
            + f"\n\nPrediction Confidence: {self.accuracy:.2%}"
            + f"\nStop-loss: {output['stop_loss']:.2f}"
            + f"\nTarget: {output['target_price']:.2f}"
        )

        return output


# Example Usage
if __name__ == "__main__":
    #ticker = "JINDALSTEL"  # Replace with any stock ticker
    ticker = "JINDALSTEL"  # Replace with any stock ticker
    engine = StockEngine(ticker, 'NS', 900)
    prediction = engine.predict_today()
    print(prediction["decision"])