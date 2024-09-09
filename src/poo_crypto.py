import logging
import time
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import requests
import numpy as np

# Définitions des niveaux de log personnalisés
LOG_BUY = 25
LOG_SELL = 24
LOG_GAINS = 23
LOG_LOST = 22

# Ajouter les nouveaux niveaux au logger
logging.addLevelName(LOG_BUY, 'BUY')
logging.addLevelName(LOG_SELL, 'SELL')
logging.addLevelName(LOG_GAINS, 'GAINS')
logging.addLevelName(LOG_LOST, 'LOST')

# Créer une classe personnalisée de logger avec les nouvelles méthodes
class CustomLogger(logging.Logger):
    def __init__(self, name, level=logging.INFO):
        super().__init__(name, level)

    def buy(self, message, *args, **kwargs):
        if self.isEnabledFor(LOG_BUY):
            self._log(LOG_BUY, message, args, **kwargs)

    def sell(self, message, *args, **kwargs):
        if self.isEnabledFor(LOG_SELL):
            self._log(LOG_SELL, message, args, **kwargs)

    def gains(self, message, *args, **kwargs):
        if self.isEnabledFor(LOG_GAINS):
            self._log(LOG_GAINS, message, args, **kwargs)

    def lost(self, message, *args, **kwargs):
        if self.isEnabledFor(LOG_LOST):
            self._log(LOG_LOST, message, args, **kwargs)

# Remplacer le logger par défaut par notre CustomLogger
logging.setLoggerClass(CustomLogger)

# Configuration des couleurs pour les niveaux
class CustomFormatter(logging.Formatter):
    COLORS = {
        'SELL': '\033[94m',  # Bleu
        'BUY': '\033[93m',   # Jaune
        'GAINS': '\033[92m', # Vert
        'LOST': '\033[91m',  # Rouge
        'INFO': '\033[97m',  # Blanc forcé
        'RESET': '\033[0m'   # Réinitialiser la couleur
    }

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        message = super().format(record)
        return f"{log_color}{message}{self.COLORS['RESET']}"

# Configuration du logger avec les niveaux et couleurs personnalisées
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',  # Inclure l'horodatage et le niveau de log
    datefmt='%Y-%m-%d %H:%M:%S',  # Format plus lisible de l'horodatage
    handlers=[
        logging.FileHandler("trading_simulator.log"),
        logging.StreamHandler()
    ]
)

# Appliquer la CustomFormatter avec des couleurs
for handler in logging.getLogger().handlers:
    handler.setFormatter(CustomFormatter())

# Initialisation du logger
logger = logging.getLogger(__name__)

# Classe CoinbaseAPI
class CoinbaseAPI:
    def __init__(self):
        self.url = 'https://api.coinbase.com/v2/prices/spot?currency=USD'

    def get_current_price(self):
        response = requests.get(self.url)
        data = response.json()
        return float(data['data']['amount'])

# Classe CoinGeckoAPI
class CoinGeckoAPI:
    def __init__(self, crypto_id='bitcoin'):
        self.crypto_id = crypto_id
        self.base_url = "https://api.coingecko.com/api/v3/coins"

    def get_historical_data(self, days=7):
        url = f"{self.base_url}/{self.crypto_id}/market_chart"
        params = {
            'vs_currency': 'usd',
            'days': days
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()

            data = response.json()

            if 'prices' not in data:
                raise KeyError(f"La clé 'prices' est absente dans la réponse de l'API : {data}")

            prices = data['prices']
            df = pd.DataFrame(prices, columns=['timestamp', 'price'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df

        except requests.exceptions.RequestException as e:
            logger.error(f"Erreur lors de la requête à l'API CoinGecko : {e}")
            return None

# Classe SimpleStrategyAI
class SimpleStrategyAI:
    def __init__(self, tolerance=0.002, rsi_window=14, macd_short_window=12, macd_long_window=26, macd_signal_window=9):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.tolerance = tolerance
        self.rsi_window = rsi_window
        self.macd_short_window = macd_short_window
        self.macd_long_window = macd_long_window
        self.macd_signal_window = macd_signal_window

    def train(self, market_data):
        market_data['time_index'] = np.arange(len(market_data))
        X = market_data['time_index'].values.reshape(-1, 1)
        y = market_data['price'].values
        self.model.fit(X, y)

    def predict(self, market_data, future_steps=1):
        future_index = np.array([[len(market_data) + future_steps]])
        predicted_price = self.model.predict(future_index)
        return predicted_price[0]

    def calculate_rsi(self, market_data):
        delta = market_data['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, market_data):
        exp1 = market_data['price'].ewm(span=self.macd_short_window, adjust=False).mean()
        exp2 = market_data['price'].ewm(span=self.macd_long_window, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=self.macd_signal_window, adjust=False).mean()
        return macd, signal

    def analyze(self, market_data):
        self.train(market_data)
        predicted_price = self.predict(market_data, future_steps=1)
        current_price = market_data['price'].iloc[-1]

        rsi = self.calculate_rsi(market_data).iloc[-1]
        macd, signal = self.calculate_macd(market_data)
        macd_value = macd.iloc[-1]
        signal_value = signal.iloc[-1]

        logger.info(f"Current Price : {current_price:.2f} USD")
        logger.info(f"Predicted Price : {predicted_price:.2f} USD")
        logger.info(f"RSI : {rsi:.2f}")
        logger.info(f"MACD : {macd_value:.2f}, Signal : {signal_value:.2f}")

        percentage_change = abs(predicted_price - current_price) / current_price

        if rsi < 40 and percentage_change > self.tolerance:
            if macd_value > signal_value:
                logger.buy("Buy (RSI indicates oversold and MACD confirms)")
                return 'buy'
            else:
                logger.info("Hold (RSI indicates oversold but MACD not confirming)")
                return 'hold'
        elif rsi > 60 and percentage_change > self.tolerance:
            if macd_value < signal_value:
                logger.sell("Sell (RSI indicates overbought and MACD confirms)")
                return 'sell'
            else:
                logger.info("Hold (RSI indicates overbought but MACD not confirming)")
                return 'hold'
        else:
            logger.info("Hold (No significant indicators for buy/sell)")
            return 'hold'

# Classe TradingSimulator
class TradingSimulator:
    def __init__(self, initial_balance=1000):
        self.balance = initial_balance
        self.bitcoin_balance = 0
        self.initial_balance = initial_balance
        self.trades = []

    def trade(self, decision, current_price, timestamp):
        if decision == 'buy' and self.balance > 0:
            btc_to_buy = self.balance / current_price
            self.bitcoin_balance += btc_to_buy
            self.trades.append(('buy', btc_to_buy, current_price, timestamp))
            self.balance = 0
            logger.buy(f"{timestamp}: Achat de {btc_to_buy:.6f} BTC à {current_price:.2f} USD.")
            logger.info(f"Montant global : {self.balance:.2f} USD")

        elif decision == 'sell' and self.bitcoin_balance > 0:
            usd_gained = self.bitcoin_balance * current_price
            self.trades.append(('sell', self.bitcoin_balance, current_price, timestamp))
            self.balance += usd_gained
            self.bitcoin_balance = 0
            logger.sell(f"{timestamp}: Vente de BTC à {current_price:.2f} USD.")
            logger.info(f"Montant global après la vente : {self.balance:.2f} USD")

        total_value = self.balance + self.bitcoin_balance * current_price
        gain_loss = total_value - self.initial_balance
        if gain_loss >= 0:
            logger.gains(f"Gains actuels : {gain_loss:.2f} USD.")
        else:
            logger.lost(f"Pertes actuelles : {gain_loss:.2f} USD.")

    def display_results(self, current_price):
        total_value = self.balance + self.bitcoin_balance * current_price
        gain_loss = total_value - self.initial_balance
        logger.info(f"\nValeur totale : {total_value:.2f} USD.")
        if gain_loss >= 0:
            logger.gains(f"Gains finaux : {gain_loss:.2f} USD.")
        else:
            logger.lost(f"Pertes finales : {gain_loss:.2f} USD.")

# Boucle infinie pour exécuter le programme en continu
if __name__ == '__main__':
    coingecko_api = CoinGeckoAPI()
    coinbase_api = CoinbaseAPI()

    historical_data = coingecko_api.get_historical_data(days=7)
    if historical_data is None:
        logger.error("Impossible de récupérer les données historiques.")
        exit()

    ai_strategy = SimpleStrategyAI(tolerance=0.002, rsi_window=14, macd_short_window=12, macd_long_window=26, macd_signal_window=9)
    simulator = TradingSimulator(initial_balance=1000)

    try:
        while True:
            current_price = coinbase_api.get_current_price()
            new_data = {
                'timestamp': datetime.now(),
                'price': current_price
            }
            historical_data = pd.concat([historical_data, pd.DataFrame([new_data])], ignore_index=True)

            decision = ai_strategy.analyze(historical_data)
            simulator.trade(decision, current_price, datetime.now())

            logger.info("Attente de 10 minutes avant la prochaine mise à jour...")
            time.sleep(6)  # Attente de 10 minutes

    except KeyboardInterrupt:
        logger.info("Programme interrompu. Affichage des résultats finaux.")
        simulator.display_results(current_price)
