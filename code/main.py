from Balancing.prosumer import Prosumer
from Balancing.balancingProcess import BalancingProcess
import numpy as np
import pandas as pd
from Balancing.useful_functions import generate_load_profile
from Blockchain.blockchain_v2 import Blockchain , Miner
from Balancing.regulator import Regulator

NUM_PROSUMERS = 100
NUM_NEIGHBOURHOODS = 5
HOURS = 24
DIFFICULTY = 3
NUM_MINERS = 10


def run_simulation():
    #list of all prosumers, it's useless actually because we have the neighbourhoods dictionary
    prosumers = []
    neighbourhoods = {i: [] for i in range(NUM_NEIGHBOURHOODS)}  # dictionnary to hold prosumers by neighbourhood

    for i in range(NUM_PROSUMERS):
        pv_capacity = np.random.uniform(3, 7)  # 3-7 kW PV systems
        load_profile = generate_load_profile()
        battery_capacity = np.random.choice([0, 5, 10])  # some prosumers have batteries
        neighbourhood = np.random.randint(0, NUM_NEIGHBOURHOODS) # assign to a neighbourhood
        
        prosumer = Prosumer(i, pv_capacity, load_profile, battery_capacity , neighbourhood)
        prosumers.append(prosumer) # add to prosumers list
        neighbourhoods[neighbourhood].append(prosumer)  # add prosumer to its neighbourhood
        # neighbourhoods[neighbourhood].append(i)

    balancing = BalancingProcess(neighbourhoods) #take just the neighbourhoods dictionary that already contains the Prosumers
    regulator = Regulator(objective="Maximize_P2P")
    energy_chain = Blockchain(difficulty=DIFFICULTY)
    miners_names = [f"Miner_Node_{i}" for i in range(1, NUM_MINERS + 1)]

    stats_list = []

    for hour in range(HOURS):
        print(f"\n--- Hour {hour} ---")


        #--- BALANCING

        balancing.set_hour(hour)

        current_market_price = 0.2  # assuming a fixed market price for simplicity, but then will be the output of Price Forecasting module
        
        balancing.step1_self_balancing()

        for prosumer in prosumers:
            prosumer.calculate_trading_price(current_market_price = current_market_price)

        balancing.step2_local_market(energy_chain)

        balancing.step3_grid_interaction(current_market_price = current_market_price , energy_chain=energy_chain)

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
        
        winner_name , winner_power = energy_chain.winner_selection(competitors_names , competitors_weights)
        
        print(f"  Winner Miner: {winner_name} (Hash Power: {winner_power:.4f})")
        
        # Winner miner mines the pending transactions
        energy_chain.mine_pending_transactions(winner_name)


        #--- STATS

        # put stats of each prosumer in a pandas df 
        for prosumer in prosumers:
            stats = prosumer.get_stats(hour)
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
    output_stats.to_csv("code/prosumer_stats.csv", index=False)
    print(f"\nChain Length: {len(energy_chain.chain)} blocks")
    is_valid = energy_chain.is_chain_valid()
    print(f"Blockchain Integrity: {'VALID' if is_valid else 'CORRUPTED'}")
    print("\nSimulation complete. Prosumer stats saved to 'prosumer_stats.csv'.")

if __name__ == "__main__":
    run_simulation()