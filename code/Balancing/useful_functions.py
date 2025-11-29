import numpy as np
import matplotlib.pyplot as plt
def generate_load_profile():
    base_load = np.random.uniform(0.5, 1.5)  # Base consumption
    
    load = []
    for hour in range(24):
        if 0 <= hour < 6:  # Night - sleeping
            load.append(base_load * 0.3)
        elif 6 <= hour < 9:  # Morning peak
            load.append(base_load * 1.5)
        elif 9 <= hour < 17:  # Day - away
            load.append(base_load * 0.7)
        elif 17 <= hour < 22:  # Evening peak
            load.append(base_load * 2.0)
        else:  # Late evening
            load.append(base_load * 0.8)
    
    # Add randomness
    load = [l * np.random.uniform(0.9, 1.1) for l in load]
    return load  #output is a list of 24 hours load , to access is we can use load[hour] no need for a dictionary 

# plt.figure()
# hours = range(24)
# sample_load = generate_load_profile()
# plt.plot(hours, sample_load, label='Sample Load Profile')
# plt.xlabel('Hour of Day')
# plt.ylabel('Load (kWh)')
# plt.title('Sample Daily Load Profile and PV Generation')
# plt.legend()
# plt.grid()
# plt.show()