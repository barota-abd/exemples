import os
import time
from datetime import datetime
import pandas as pd
import requests

class CoinbaseAPI:
    def __init__(self):
        self.url = 'https://api.coinbase.com/v2/prices/spot?currency=USD'

    def get_current_price(self):
        response = requests.get(self.url)
        data = response.json()
        return float(data['data']['amount'])

class CoinGeckoAPI:
    def __init__(self, crypto_id='bitcoin'):
        self.crypto_id = crypto_id
        self.base_url = "https://api.coingecko.com/api/v3/coins"

    def get_historical_data(self, days=7):
        """
        Récupère les données historiques du prix du Bitcoin sur les 7 derniers jours.
        L'intervalle est géré automatiquement par l'API si le nombre de jours est entre 2 et 90.
        """
        url = f"{self.base_url}/{self.crypto_id}/market_chart"
        params = {
            'vs_currency': 'usd',
            'days': days  # Récupérer les données des X derniers jours
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()  # Vérifie si la requête est réussie

            # Parse la réponse JSON
            data = response.json()

            # Vérifier si 'prices' est dans la réponse
            if 'prices' not in data:
                raise KeyError(f"La clé 'prices' est absente dans la réponse de l'API : {data}")

            # Récupérer les prix et les convertir en DataFrame
            prices = data['prices']
            df = pd.DataFrame(prices, columns=['timestamp', 'price'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')  # Convertir en datetime
            return df

        except requests.exceptions.RequestException as e:
            print(f"Erreur lors de la requête à l'API CoinGecko : {e}")
            return None

class SimpleStrategyAI:
    def analyze(self, market_data):
        # Moyenne mobile courte et longue
        short_window = market_data['price'].rolling(window=5).mean().iloc[-1]
        long_window = market_data['price'].rolling(window=15).mean().iloc[-1]

        if short_window > long_window:
            return 'buy'
        elif short_window < long_window:
            return 'sell'
        else:
            return 'hold'

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
            print(f"{timestamp}: Achat de {btc_to_buy:.6f} BTC à {current_price:.2f} USD.")

        elif decision == 'sell' and self.bitcoin_balance > 0:
            usd_gained = self.bitcoin_balance * current_price
            self.trades.append(('sell', self.bitcoin_balance, current_price, timestamp))
            self.balance += usd_gained
            self.bitcoin_balance = 0
            print(f"{timestamp}: Vente de BTC à {current_price:.2f} USD.")
            print(f"Montant global après la vente : {self.balance:.2f} USD")

        # Trace des gains ou pertes
        total_value = self.balance + self.bitcoin_balance * current_price
        gain_loss = total_value - self.initial_balance
        print(f"Gains/Pertes actuels : {gain_loss:.2f} USD.")

    def display_results(self, current_price):
        total_value = self.balance + self.bitcoin_balance * current_price
        gain_loss = total_value - self.initial_balance
        print(f"\nValeur totale : {total_value:.2f} USD.")
        print(f"Gains/Pertes finaux : {gain_loss:.2f} USD.")

if __name__ == '__main__':
    # Initialiser les APIs
    coingecko_api = CoinGeckoAPI()
    coinbase_api = CoinbaseAPI()

    # Récupérer les données historiques sur 7 jours
    historical_data = coingecko_api.get_historical_data(days=7)

    if historical_data is None:
        print("Impossible de récupérer les données historiques.")
        exit()

    ai_strategy = SimpleStrategyAI()
    simulator = TradingSimulator(initial_balance=1000)

    # Boucle infinie : mise à jour des données et prise de décision de trading toutes les 10 minutes
    try:
        while True:
            current_price = coinbase_api.get_current_price()
            new_data = {
                'timestamp': datetime.now(),
                'price': current_price
            }
            historical_data = pd.concat([historical_data, pd.DataFrame([new_data])], ignore_index=True)

            # Prendre une décision basée sur les données mises à jour
            decision = ai_strategy.analyze(historical_data)
            simulator.trade(decision, current_price, datetime.now())

            # Attendre 10 minutes avant de récupérer les nouvelles données
            print("Attente de 10 minutes avant la prochaine mise à jour...")
            time.sleep(60)  # 600 secondes = 10 minutes

    except KeyboardInterrupt:
        # Si l'utilisateur interrompt le programme (Ctrl+C), on affiche les résultats finaux
        print("\nProgramme interrompu. Affichage des résultats finaux.")
        simulator.display_results(current_price)

