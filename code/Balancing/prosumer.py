import numpy as np
class Prosumer:
    def __init__(self, prosumer_id, pv_capacity, load_profile, battery_capacity , neighbourhood):
        """
        prosumer_id: unique identifier (1-100)
        pv_capacity: max PV generation in kW
        load_profile: list of 24 hour loads [kWh]
        battery_capacity: battery size in kWh
        """
        self.id = prosumer_id
        self.pv_capacity = pv_capacity # max PV generation in kW
        self.load_profile = load_profile # list of 24 hour loads [kWh]
        self.battery_capacity = battery_capacity # battery size in kWh
        self.neighbourhood = neighbourhood # neighbourhood ID
        self.battery_level = 0  # current battery level in kWh
        
        self.imbalance = 0  # current energy imbalance in kWh for the hour
        self.money_balance = 0    # money balance (earnings - costs)
        self.transactions = []  # history of trades

    
    def get_load(self, hour):
        return self.load_profile[hour]
    
    def generate_pv(self , hour):
        # simple model: PV generation peaks at midday
        peak_generation = self.pv_capacity
        hours_of_daylight = 12
        time_from_sunrise = hour - 6
        
        # Sin function: 0 at sunrise/sunset, 1 at noon
        sun_intensity = np.sin(np.pi * time_from_sunrise / hours_of_daylight)
        if 6 <= hour <= 19:
            generation = peak_generation * sun_intensity * np.random.uniform(0.8,1.0)  #random for the weather effect
        else:
            generation = 0
        return generation
    

    
    def get_stats(self):
        stats = {
            "id": self.id,
            "pv_capacity": self.pv_capacity,
            "battery_capacity": self.battery_capacity,
            "battery_level": self.battery_level,
            "imbalance": self.imbalance,
            "money_balance": self.money_balance
        }
        return stats
    
    def self_balance(self, hour):
        pv_generation = self.generate_pv(hour)
        load = self.get_load(hour)
        residual = pv_generation - load # positive if surplus, negative if deficit
        #self.imbalance = self.imbalance - residual # - because if residual is positive, imbalance should decrease
        # why adding to imbalance and not only looking at the residual?
        # self.imbalance = residual # isn't it more correct?   # if we say that at each step imbalance is reset to 0 yes, otherwise: ...
        self.imbalance = self.imbalance - residual

        if self.imbalance > 0:  # surplus energy
            if self.battery_capacity > 0:
                available_storage = self.battery_capacity - self.battery_level
                energy_to_store = min(self.imbalance, available_storage)
                self.battery_level =  self.battery_level + energy_to_store
                self.imbalance = self.imbalance - energy_to_store
        elif self.imbalance < 0:  # deficit energy
            if self.battery_level > 0:
                energy_needed = abs(self.imbalance)
                energy_from_battery = min(energy_needed, self.battery_level)
                self.battery_level = self.battery_level - energy_from_battery
                self.imbalance = self.imbalance + energy_from_battery