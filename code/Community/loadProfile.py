import numpy as np

def generate_load_profile():
    """
    Generates a realistic 24-hour energy load profile (consumption) in kWh, 
    simulating typical residential usage patterns.

    The profile is based on a randomly selected base load and is modulated
    to create peaks during morning and evening hours.

    Returns:
        list: A list of 24 floats, where index i represents the energy load (kWh) 
            for hour i (0 to 23).
    """
    # Randomly select a base load factor (kW) to introduce variability between prosumers
    base_load = np.random.uniform(0.5, 1.5)
    
    load = []
    for hour in range(24):
        # Determine the consumption multiplier based on the hour of the day
        if 0 <= hour < 6: # Night (00:00 - 05:59) - Lowest consumption (sleeping)
            load.append(base_load * 0.3)
        elif 6 <= hour < 9: # Morning Peak (06:00 - 08:59) - High consumption (breakfast, showering)
            load.append(base_load * 1.2)
        elif 9 <= hour < 17: # Day (09:00 - 16:59) - Low to moderate consumption (people away at work/school)
            load.append(base_load * 0.7)
        elif 17 <= hour < 22: # Evening Peak (17:00 - 21:59) - Highest consumption (cooking, entertainment, heating/cooling)
            load.append(base_load * 1.5)
        else: # Late Evening (22:00 - 23:59) - Moderate consumption
            load.append(base_load * 0.8)
    
    # Add small random noise (±10%) to introduce hourly fluctuation
    load = [l * np.random.uniform(0.9, 1.1) for l in load]

    # The output is a list where load[hour] gives the consumption for that hour
    return load 