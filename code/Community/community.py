from .prosumer import Prosumer
from .city import City
from .useful_functions import generate_load_profile
import numpy as np

def generate_community(num_prosumers, num_neighbourhoods, neighbourhood_pool, pv_number_range, pv_capacity, battery_range, losses, pv_model_path=None):
    prosumers = []
    city = City(num_prosumers, num_neighbourhoods, neighbourhood_pool)
    
    for i in range(num_prosumers):
        # Generate the parameters of the prosumer at random 
        # - Generate a random number of PV pannels
        pv_number = np.random.randint(pv_number_range[0], pv_number_range[1])
        # - Multiply it by the PV pannel capacity to get the total capacity
        total_pv_capacity = pv_number * pv_capacity

        # - Generate the load profile
        load_profile = generate_load_profile()

        # - Generate a random battery capcatity
        battery_capacity = np.random.choice(battery_range)
        
        # Assign the prosumer to a neighbourhood and get its coordinates
        neighbourhood, latitude, longitude = city.assign_prosumer_to_neighbourhood()
        
        # Create the prosumer
        prosumer = Prosumer(i, total_pv_capacity, load_profile, battery_capacity, losses, neighbourhood, latitude, longitude, pv_model_path)
        
        # Add the prosumer to its neighbourhood and to the prosumers list
        city.add_prosumer_to_neighbourhood(prosumer, neighbourhood)
        prosumers.append(prosumer)
    
    city.plot_neighbourhoods()

    return prosumers, city.neighbourhoods