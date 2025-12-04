from Balancing.balancingProcess import BalancingProcess
import numpy as np
import pandas as pd
from Blockchain.blockchain_v2 import Blockchain , Miner
from Balancing.regulator import Regulator
from Community.community import generate_community
from PvForecast.pvModel import PvModel
import os

NUM_PROSUMERS = 100
NUM_NEIGHBOURHOODS = 10
HOURS = 24
DATE_STRING = '2025-12-04'
DIFFICULTY = 3
NUM_MINERS = 10

NEIGHBOURHOOD_POOL = {
    'Centro': [45.065, 45.080, 7.675, 7.690],
    'San_Salvario': [45.050, 45.065, 7.675, 7.695],
    'Crocetta': [45.060, 45.075, 7.645, 7.660],
    'Aurora': [45.088, 45.100, 7.675, 7.695],
    'Vanchiglia': [45.068, 45.083, 7.690, 7.710],
    'Lingotto': [45.015, 45.035, 7.640, 7.665],
    'Santa_Rita': [45.040, 45.055, 7.630, 7.655],
    'San_Donato': [45.080, 45.095, 7.640, 7.658],
    'Cit_Turin': [45.070, 45.085, 7.660, 7.675],
    'Barriera_di_Milano': [45.100, 45.115, 7.665, 7.685],
}

# Prosumer parameters
PV_NUMBER_RANGE = (0, 20) # Number of PV pannels
PV_CAPACITY = 0.25 # PV pannel capacity in kW
BATTERY_RANGE = [0, 5, 10] # Battery capacity in kW
LOSSES = 14 # Sum of PV system losses in percent

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pv_model_path = os.path.join(BASE_DIR, "PvForecast", "pv_predictor_xgb.joblib")

# pv_model_path = "PvForecast/pv_predictor_xgb.joblib"

def run_simulation():
    #  Date of the simulation 
    date = pd.to_datetime(DATE_STRING)

    # Generate the community
    print("Generating prosumer community")
    prosumers, neighbourhoods = generate_community(NUM_PROSUMERS, NUM_NEIGHBOURHOODS, NEIGHBOURHOOD_POOL, PV_NUMBER_RANGE, PV_CAPACITY, BATTERY_RANGE, LOSSES, pv_model_path)

    balancing = BalancingProcess(neighbourhoods) # take just the neighbourhoods dictionary that already contains the Prosumers
    regulator = Regulator(objective="Maximize_P2P")
    energy_chain = Blockchain(difficulty=DIFFICULTY)
    miners_names = [f"Miner_Node_{i}" for i in range(1, NUM_MINERS + 1)]

    stats_list = []

    print(f"Simulating for {HOURS} hours on {DATE_STRING}")
    for hour in range(HOURS):
        print(f"\n--- Hour {hour} ---")

        #--- BALANCING

        balancing.set_date_and_hour(date, hour)

        current_market_price = 0.2  # assuming a fixed market price for simplicity, but then will be the output of Price Forecasting module
        
        print("- Step 1: self balancing")
        balancing.step1_self_balancing()

        for prosumer in prosumers:
            prosumer.calculate_trading_price(current_market_price=current_market_price)

        print("- Step 2: local market")
        balancing.step2_local_market(energy_chain)

        print("- Step 3: grid integration")
        balancing.step3_grid_interaction(current_market_price=current_market_price, energy_chain=energy_chain)

        #--- REGULATOR

        # regulator.apply_regulations(prosumers, hour)

        #--- BLOCKCHAIN

        competitors_names = []
        competitors_weights = []

        for miner_name in miners_names:
            m = Miner(miner_name)
            power = m.Pow_compete()

            competitors_names.append(miner_name)
            competitors_weights.append(power)
        
        winner_name, winner_power = energy_chain.winner_selection(competitors_names, competitors_weights)
        
        print(f"  Winner Miner: {winner_name} (Hash Power: {winner_power:.4f})")
        
        # Winner miner mines the pending transactions
        energy_chain.mine_pending_transactions(winner_name)

        #--- STATS

        # put stats of each prosumer in a pandas df 
        for neighbourhood , prosumers_in_neighbourhood in neighbourhoods.items():
            for prosumer in prosumers_in_neighbourhood:
                stats = prosumer.get_stats(date, hour)
                stats_list.append({
                    "hour": hour,
                    "id": stats["id"],
                    "pv_capacity": stats["pv_capacity"],
                    "battery_capacity": stats["battery_capacity"],
                    "battery_level": stats["battery_level"],
                    "imbalance": stats["imbalance"],
                    "money_balance": stats["money_balance"],
                    "trading_price": stats["trading_price"],
                    "neighbourhood": stats["neighbourhood"],
                    "transactions": stats["transactions"]
                })

    output_stats = pd.DataFrame(stats_list)
    output_stats.to_csv("prosumer_stats.csv", index=False) # if we run the file from the code folder
    print(f"\nChain Length: {len(energy_chain.chain)} blocks")
    is_valid = energy_chain.is_chain_valid()
    print(f"Blockchain Integrity: {'VALID' if is_valid else 'CORRUPTED'}")
    print("\nSimulation complete. Prosumer stats saved to 'prosumer_stats.csv'.")

if __name__ == "__main__":
    run_simulation()