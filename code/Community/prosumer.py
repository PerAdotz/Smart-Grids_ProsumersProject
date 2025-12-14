import numpy as np
from PvForecast.pvModel import PvModel

class Prosumer:
    """
    Represents a Prosumer (producer/consumer) in a local energy market.

    A Prosumer has a load profile, a PV generation capacity, and optionally a battery.
    It manages its own energy balance and participates in energy trading.
    """

    def __init__(self, prosumer_id, pv_capacity, load_profile, battery_capacity, losses, neighbourhood, latitude, longitude, pv_model_path=None):
        """
        Initializes the Prosumer object.
        Args:
            prosumer_id (int): Unique identifier (0-NUMBER_PROSUMERS).
            pv_capacity (float): Max PV generation capacity in kW.
            load_profile (list): List of 24 hour loads [kWh] (index corresponds to hour).
            battery_capacity (float): Battery size in kWh (0 if no battery).
            losses (float): System losses factor (e.g., 0.05 for 5%).
            neighbourhood (int): Neighbourhood ID.
            latitude (float): Latitude coordinate.
            longitude (float): Longitude coordinate.
            pv_model_path (str, optional): Path to the trained PV generation prediction model. Defaults to None.
        """
        self.id = prosumer_id
        self.pv_capacity = pv_capacity # max PV generation in kW
        self.load_profile = load_profile # list of 24 hour loads [kWh]
        self.battery_capacity = battery_capacity # battery size in kWh
        self.losses = losses
        self.neighbourhood = neighbourhood # neighbourhood
        self.latitude = latitude
        self.longitude = longitude

        # Load the model for PV generation prediction
        self.pv_model = PvModel()
        if pv_model_path:
            self.pv_model.load_model(pv_model_path)

        # State variables
        self.battery_level = 0 # current battery level in kWh
        self.imbalance = 0 # current energy imbalance in kWh for the hour
        self.money_balance = 0 # money balance (earnings - costs)
        self.transactions = {hour: [] for hour in range(24)} # record of transactions per hour
        self.trading_price = 0 # price per kWh for trading in local market for the current hour

        # Performance and exchange metrics
        self.bonus = 1.0  # total bonuses received from regulator
        self.penalty = 1.0  # total penalties paid to regulator
        self.p2p_exchanges = 0 # number of P2P exchanges made
        self.agg_exchanges = 0 # number of exchanges with the Aggregator made

    def get_load(self, hour):
        """
        Retrieves the energy load (consumption) for a given hour.

        Args:
            hour (int): The hour of the day (0-23).

        Returns:
            float: The energy load in kWh for that hour.
        """
        return self.load_profile[hour]

    def get_stats(self, date, hour):
        """
        Gathers a set of statistics and current state of the prosumer.

        Args:
            date (str): The current date ("YYYY-MM-DD").
            hour (int): The current hour of the day (0-23).

        Returns:
            dict: A dictionary containing the prosumer's current stats.
        """
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
        """
        Retrieves the prosumer's parameters needed for the PV generation model.

        Returns:
            dict: A dictionary of prosumer ID, PV capacity, battery capacity, latitude, and longitude.
        """
        params = {
            "id": self.id,
            "pv_capacity": self.pv_capacity,
            "battery_capacity": self.battery_capacity,
            "latitude": self.latitude,
            "longitude": self.longitude
        }
        return params
    
    def self_balance(self, date, hour):
        """
        Calculates the residual energy and attempts to balance it using the battery.

        Args:
            date (str): The current date ("YYYY-MM-DD").
            hour (int): The current hour of the day (0-23).
        """
        pv_generation = self.generate_pv(date, hour)
        load = self.get_load(hour)
        # Calculate residual energy: Load - PV_Generation
        # POSITIVE IF DEFICIT (needs to buy), NEGATIVE IF SURPLUS (needs to sell)
        residual = load - pv_generation

        # Set the imbalance to the residual after generation/consumption
        self.imbalance = residual

        # If prosumer has a deficit of energy (needs to discharge)
        if self.imbalance > 0:
            if self.battery_capacity > 0:
                needed_energy = self.imbalance

                # Discharge: use minimum of what is needed and what is currently in the battery
                energy_from_battery = min(needed_energy, self.battery_level)

                # Update battery level
                self.battery_level = self.battery_level - energy_from_battery

                # Reduce the remaining imbalance
                self.imbalance = self.imbalance - energy_from_battery

        # If prosumer has a surplus of energy (needs to charge)
        elif self.imbalance < 0:
            if self.battery_capacity > 0:
                surplus_energy = abs(self.imbalance)
                # Calculate available storage capacity
                available_storage = self.battery_capacity - self.battery_level

                # Charge: store minimum of the surplus and the available storage space
                energy_to_store = min(surplus_energy, available_storage)

                # Update battery level
                self.battery_level = self.battery_level + energy_to_store

                # Reduce the remaining imbalance: since imbalance is negative (surplus), adding energy_to_store moves it toward zero
                self.imbalance = self.imbalance + energy_to_store

    def calculate_trading_price(self, current_market_price):
        """
        Sets the trading price based on the current market reference price and the prosumer's imbalance.
        Sellers (imbalance < 0) will offer a price slightly below the market.
        Buyers (imbalance > 0) will bid a price around the market price.

        Args:
            current_market_price (float): The reference price for energy trading.

        Returns:
            None: Updates the self.trading_price attribute.
        """
        # Add some randomness to simulate different bidding strategies
        if self.imbalance < 0: # Seller (has surplus energy to sell)
            # Seller Strategy: Offer between 70% and 90% of market price
            self.trading_price = current_market_price * np.random.uniform(0.7, 0.9)
            
        elif self.imbalance > 0: # Buyer (has deficit energy to buy)
            # Buyer Strategy: Willing to pay 90% to 110% of market price
            self.trading_price = current_market_price * np.random.uniform(0.9, 1.1)
        else:
            # No imbalance, no need to trade
            self.trading_price = 0
    
    def generate_pv(self, date, hour):
        """
        Predicts the PV generation for the given date and hour using the loaded model.

        Args:
            date (str): The current date ("YYYY-MM-DD").
            hour (int): The current hour of the day (0-23).

        Returns:
            float: The predicted PV generation in kWh for that hour.
        """
        params = self.get_params()
        generation = self.pv_model.predict_single_point(params, date, hour)
        return generation