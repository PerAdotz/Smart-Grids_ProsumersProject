from Balancing.balancingProcess import BalancingProcess
import pandas as pd
from Blockchain.blockchain_v2 import Blockchain , Miner
from Balancing.regulator import Regulator
from Community.community import generate_community
from PriceForecast.priceForecaster import PriceForecaster
import os

# --- Configuration Parameters ---
NUM_PROSUMERS = 100
NUM_NEIGHBOURHOODS = 10
HOURS = 24
DATE_STRING = '2025-10-31' # latest point in the price dataset
DIFFICULTY = 3
NUM_MINERS = 10

# Geographical bounding boxes for simulated neighbourhoods
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

# Prosumer physical parameters
PV_NUMBER_RANGE = (0, 20) # Range for number of PV panels
PV_CAPACITY = 0.25 # Single PV panel capacity in kW
BATTERY_RANGE = [0, 5, 10] # Available battery capacities in kWh
LOSSES = 14 # Sum of PV system losses in percent

# Policy definitions
P2P_BONUS_POLICY = {
    '1': 1.02, # 2% bonus for at least 1 P2P exchange
    '5': 1.05, # 5% bonus for at least 5 P2P exchanges
    '10': 1.10 # 10% bonus for at least 10 P2P exchanges
}
GRID_PENALTY_POLICY = {
    '5': 1.05, # 5% penalty for at least 5 grid exchanges
    '10': 1.10 # 10% penalty for at least 10 grid exchanges
}

# Determine the path for the trained PV model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PV_DIR = "PvForecast"
pv_model_path = os.path.join(BASE_DIR, PV_DIR, "pv_predictor_xgb.joblib")

# Define the directory containing the datasets for the price forecasting model
DATA_DIR = 'Data_ElectricityMarketPrices'
file_paths = {
    'df_2021': os.path.join(BASE_DIR, DATA_DIR, 'Anno 2021.xlsx'),
    'df_2022': os.path.join(BASE_DIR, DATA_DIR, 'Anno 2022.xlsx'),
    'df_2023': os.path.join(BASE_DIR, DATA_DIR, 'Anno 2023.xlsx'),
    'df_2024': os.path.join(BASE_DIR, DATA_DIR, 'Anno 2024.xlsx'),
    'df_2025': os.path.join(BASE_DIR, DATA_DIR, 'Anno 2025_10.xlsx'),
}

# Define the path for the trained price forecating model
PRICE_DIR = "PriceForecast"
price_model_path = os.path.join(BASE_DIR, PRICE_DIR, "price_forecaster_multi.joblib")

# Loockback for the price forecasting model (number of hours)
LOOKBACK = 24

def run_simulation():
    """
    Main function to run the hourly energy trading simulation over a single day.
    Initializes all agents, processes energy balancing and trading steps, 
    manages the blockchain, and applies regulatory policies hourly.
    """
    # Initialize simulation date 
    date = pd.to_datetime(DATE_STRING)
    date_int = int(DATE_STRING.replace('-', ''))

    # 1. Generate the community of prosumers and neighbourhoods
    print("Generating prosumer community")
    prosumers, neighbourhoods = generate_community(
        NUM_PROSUMERS, NUM_NEIGHBOURHOODS, NEIGHBOURHOOD_POOL, 
        PV_NUMBER_RANGE, PV_CAPACITY, BATTERY_RANGE, LOSSES, 
        pv_model_path
    )

    # 2. Initialize core simulation modules
    # - Price Forecaster for dynamic market prices
    price_forecaster = PriceForecaster(file_paths, lookback=LOOKBACK)
    price_forecaster.load_model(price_model_path)

    # - BalancingProcess manages the hourly energy exchanges
    balancing = BalancingProcess(prosumers, neighbourhoods)

    # - Regulator applies incentive policies
    regulator = Regulator()

    # - Blockchain tracks transactions
    energy_chain = Blockchain(difficulty=DIFFICULTY)

    # 3. Setup miners for the Proof-of-Work process
    miners_names = [f"Miner_Node_{i}" for i in range(1, NUM_MINERS + 1)]

    stats_list = [] # List to accumulate hourly statistics

    # 4. Simulate
    print(f"Simulating for {HOURS} hours on {DATE_STRING}")
    for hour in range(HOURS):
        print(f"\n--- Hour {hour} ---")

        # --- ENERGY BALANCING AND TRADING ---

        # Set the current time context
        balancing.set_date_and_hour(date, hour)

        # Base market price for grid transactions and P2P bidding reference
        # Use the prediction from the PriceForecaster instance
        current_market_price = price_forecaster.predict_next_hour(date_int, hour)
        current_PUN = current_market_price["PUN"]
        current_NORD = current_market_price["NORD"]
        print(f"- Current Market Price (Predicted):")
        print(f"PUN: {current_PUN:.4f} €/kWh")
        print(f"NORD: {current_NORD:.4f} €/kWh")
        
        # Step 1: Self-Balancing (Generation, Load, Battery)
        print("- Step 1: self balancing")
        balancing.step1_self_balancing()

        # Prosumers set their trading prices based on their imbalance and the reference market price
        for prosumer in prosumers:
            # Use the regional price (NORD)
            prosumer.calculate_trading_price(current_market_price=current_NORD)

        # Step 2: Self-Organized Trading (P2P Exchange)
        print("- Step 2: self-organized trading")
        balancing.step2_self_organized_trading(energy_chain)

        # Step 3: Local Market Clearing (Grid Exchange)
        print("- Step 3: local market")
        # Use the national price (PUN)
        balancing.step3_local_market(current_market_price=current_PUN, energy_chain=energy_chain)

        # --- BLOCKCHAIN MANAGEMENT ---
        
        # Simulate mining competition for the current batch of pending transactions
        competitors_names = []
        competitors_weights = []

        for miner_name in miners_names:
            m = Miner(miner_name)
            power = m.Pow_compete()

            competitors_names.append(miner_name)
            competitors_weights.append(power)
        
        # Select the winner based on weighted random choice (simulating Proof-of-Work)
        winner_name, winner_power = energy_chain.winner_selection(competitors_names, competitors_weights)
        print(f"  Winner Miner: {winner_name} (Hash Power: {winner_power:.4f})")
        
        # The winning miner validates and adds the transactions to a new block
        energy_chain.mine_pending_transactions(winner_name)

        # --- REGULATOR APPLICATION ---

        # The Regulator audits the past hour's activity and sets bonuses/penalties for the *next* hour
        regulator.apply_regulations(prosumers, hour , P2P_BONUS_POLICY, GRID_PENALTY_POLICY)

        # --- DATA COLLECTION ---

        # Collect and store statistics for the current hour
        for _, prosumers_in_neighbourhood in neighbourhoods.items():
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
                    "bonus": stats["bonus"],
                    "penalty": stats["penalty"],
                    "p2p_exchanges": stats["p2p_exchanges"],
                    "agg_exchanges": stats["agg_exchanges"],
                    "transactions": stats["transactions"]
                })

    # Finalize and save simulation results
    output_stats = pd.DataFrame(stats_list)
    stats_path = os.path.join(BASE_DIR, "prosumer_stats.csv")
    output_stats.to_csv(stats_path, index=False)

    # Print blockchain summary
    print(f"\nChain Length: {len(energy_chain.chain)} blocks")
    is_valid = energy_chain.is_chain_valid()
    print(f"Blockchain Integrity: {'VALID' if is_valid else 'CORRUPTED'}")
    print("\nSimulation complete. Prosumer stats saved to 'prosumer_stats.csv'.")

if __name__ == "__main__":
    run_simulation()