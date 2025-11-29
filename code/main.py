from Balancing.prosumer import Prosumer
from Balancing.balancingProcess import BalancingProcess
import numpy as np
from Balancing.useful_functions import generate_load_profile

NUM_PROSUMERS = 100
NUM_NEIGHBOURHOODS = 5
HOURS = 24

#list of all prosumers
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

for hour in range(HOURS):
    print(f"\n=== Hour {hour} ===")
    
    balancing.step1_self_balancing(hour)

    balancing.step2_local_market(current_market_price = 0.2)