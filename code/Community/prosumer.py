import numpy as np
from .useful_functions import generate_pv
from PvForecast.pvModel import PvModel

class Prosumer:
    def __init__(self, prosumer_id, pv_capacity, load_profile, battery_capacity, losses, neighbourhood, latitude, longitude, pv_model_path=None):
        """
        prosumer_id: unique identifier (1-100)
        pv_capacity: max PV generation in kW
        load_profile: list of 24 hour loads [kWh]
        battery_capacity: battery size in kWh
        losses:
        neighbourhood: neighbourhood ID
        latitude:
        longitude:
        """
        self.id = prosumer_id
        self.pv_capacity = pv_capacity # max PV generation in kW
        self.load_profile = load_profile # list of 24 hour loads [kWh]
        self.battery_capacity = battery_capacity # battery size in kWh
        self.losses = losses
        self.neighbourhood = neighbourhood # neighbourhood ID
        self.latitude = latitude
        self.longitude = longitude

        # Load the model for PV generation prediction
        self.pv_model = PvModel()
        if pv_model_path:
            self.pv_model.load_model(pv_model_path)

        self.battery_level = 0 # current battery level in kWh
        self.imbalance = 0 # current energy imbalance in kWh for the hour
        self.money_balance = 0 # money balance (earnings - costs)
        self.transactions = {hour: [] for hour in range(24)} # record of transactions per hour
        self.trading_price = 0 # price per kWh for trading in local market for the current hour

        self.bonus = 1.0  # total bonuses received from regulator
        self.penalty = 1.0  # total penalties paid to regulator
        self.p2p_exchanges = 0 #number of P2P exchanges made
        self.agg_exchanges = 0 #number of exchanges with the Aggregator made

    def get_load(self, hour):
        return self.load_profile[hour]

    def get_stats(self, date, hour):
        stats = {
            "id": self.id,
            "pv_capacity": self.pv_capacity,
            "battery_capacity": self.battery_capacity,
            "battery_level": self.battery_level,
            "imbalance": self.imbalance,
            "money_balance": self.money_balance,
            "trading_price": self.trading_price,
            "neighbourhood": self.neighbourhood,
            "load" : self.get_load(hour),
            "pv_generation" : self.generate_pv(date, hour),
            "bonus": self.bonus,
            "penalty": self.penalty,
            "p2p_exchanges": self.p2p_exchanges,
            "agg_exchanges": self.agg_exchanges,
            "transactions": self.transactions[hour]
        }
        return stats
    
    def get_params(self):
        params = {
            "id": self.id,
            "pv_capacity": self.pv_capacity,
            "battery_capacity": self.battery_capacity,
            "latitude": self.latitude,
            "longitude": self.longitude
        }
        return params
    
    def self_balance(self, date, hour):
        pv_generation = self.generate_pv(date, hour)
        load = self.get_load(hour)
        residual = load - pv_generation # POSITIVE IF DEFICIT (neeeds to buy), NEGATIVE IF SURPLUS (needs to sell)
        # self.imbalance = self.imbalance - residual # - because if residual is positive, imbalance should decrease
        # why adding to imbalance and not only looking at the residual?
        self.imbalance = residual # isn't it more correct?

        if self.imbalance > 0:  # deficit energy
            if self.battery_capacity > 0:
                needed_energy = self.imbalance
                energy_from_battery = min(needed_energy, self.battery_level)
                self.battery_level = self.battery_level - energy_from_battery
                self.imbalance = self.imbalance - energy_from_battery

        elif self.imbalance < 0:  # surplus energy
            if self.battery_capacity > 0:
                surplus_energy = abs(self.imbalance)
                available_storage = self.battery_capacity - self.battery_level
                energy_to_store = min(surplus_energy, available_storage)
                self.battery_level = self.battery_level + energy_to_store
                self.imbalance = self.imbalance + energy_to_store  # adding because imbalance is negative

    # NEW METHOD: Prosumer decides their price based on the market reference
    def calculate_trading_price(self, current_market_price):
        """
        Sets the trading price based on D-1 market price.
        Sellers want to undercut the market slightly to sell.
        Buyers are willing to pay up to the market price (or slightly more/less).
        """
        # Add some randomness to simulate different bidding strategies
        if self.imbalance < 0:  # Seller
            # Seller Strategy: Offer between 80% and 100% of market price
            self.trading_price = current_market_price * np.random.uniform(0.8, 1.0)
            
        elif self.imbalance > 0:  # Buyer
            # Buyer Strategy: Willing to pay 90% to 110% of market price
            self.trading_price = current_market_price * np.random.uniform(0.9, 1.1)
        else:
            self.trading_price = 0
    
    def generate_pv(self, date, hour):
        params = self.get_params()
        generation = self.pv_model.predict_single_point(params, date, hour)
        return generation