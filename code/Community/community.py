from .prosumer import Prosumer
from .city import City
from .loadProfile import generate_load_profile
import numpy as np

def generate_community(num_prosumers, num_neighbourhoods, neighbourhood_pool, pv_number_range, pv_capacity, battery_range, losses, pv_model_path=None):
    """
    Initializes a community of Prosumers and organizes them into Neighbourhoods within a City.

    Args:
        num_prosumers (int): The total number of Prosumers to create in the simulation.
        num_neighbourhoods (int): The total number of Neighbourhoods to create.
        neighbourhood_pool (dict): A dictionary where keys are neighbourhood names and values are 
                                tuples (min_lat, max_lat, min_lon, max_lon) defining the geographical bounds.
        pv_number_range (tuple): A tuple (min, max) defining the range for the number of PV panels per Prosumer.
        pv_capacity (float): The capacity of a single PV panel in kW.
        battery_range (list): A list of discrete battery capacities (kWh) from which to randomly choose.
        losses (float): System losses factor.
        pv_model_path (str, optional): Path to the trained PV generation prediction model. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - prosumers (list): A list of all created Prosumer objects.
            - neighbourhoods (dict): A dictionary of Neighbourhood objects.
    """
    prosumers = []
    # Initialize the City, which manages neighbourhoods and coordinates
    city = City(num_prosumers, num_neighbourhoods, neighbourhood_pool)
    
    # Loop to create and configure each individual prosumer
    for i in range(num_prosumers):
        # --- Randomly generate prosumer parameters ---
        
        # 1. Determine PV capacity
        # Generate a random integer number of PV panels within the specified range
        pv_number = np.random.randint(pv_number_range[0], pv_number_range[1])

        # Calculate the total PV capacity based on the number of panels
        total_pv_capacity = pv_number * pv_capacity

        # 2. Generate the load profile (24-hour consumption)
        load_profile = generate_load_profile()

        # 3. Determine battery capacity
        # Select a random battery size from the list of available ranges
        battery_capacity = np.random.choice(battery_range)
        
        # --- Assign to Neighbourhood and get geographical data ---
        # Determine the Neighbourhood ID and the geographic coordinates (lat/lon)
        neighbourhood, latitude, longitude = city.assign_prosumer_to_neighbourhood()
        
        # Create the Prosumer instance
        prosumer = Prosumer(
            prosumer_id=i, 
            pv_capacity=total_pv_capacity, 
            load_profile=load_profile, 
            battery_capacity=battery_capacity, 
            losses=losses, 
            neighbourhood=neighbourhood, 
            latitude=latitude, 
            longitude=longitude, 
            pv_model_path=pv_model_path
        )
        
        # Add the prosumer to its neighbourhood and to the prosumers list
        city.add_prosumer_to_neighbourhood(prosumer, neighbourhood)
        prosumers.append(prosumer)
    
    # Plot the geographical distribution of prosumers and neighbourhoods
    city.plot_neighbourhoods()

    # Return the list of Prosumers and the dictionary Neighbourhoods
    return prosumers, city.neighbourhoods