import random
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import pandas as pd
import contextily as cx

class City:
    """
    Manages the geographic distribution of Prosumers across different Neighbourhoods.
    Handles neighbourhood selection, prosumer assignment, and geographical plotting.
    """

    def __init__(self, num_prosumers, num_neighbourhoods, neighbourhood_pool):
        """
        Initializes the City by defining and selecting the neighbourhoods based on the pool.

        Args:
            num_prosumers (int): The total number of prosumers in the simulation.
            num_neighbourhoods (int): The requested number of neighbourhoods to include in the simulation.
            neighbourhood_pool (dict): A dictionary where keys are neighbourhood names and values are 
                                    tuples (min_lat, max_lat, min_lon, max_lon) defining the geographical bounds.
        """
        self.num_prosumers = num_prosumers
        self.neighbourhood_pool = neighbourhood_pool

        pool_size = len(self.neighbourhood_pool)
        
        # Get all neighbourhood names available in the pool
        neighbourhood_names = list(self.neighbourhood_pool.keys())
        
        if num_neighbourhoods <= 0:
            raise ValueError("Number of neighbourhoods must be positive.")
        
        elif num_neighbourhoods > pool_size:
            # Handle case where requested number exceeds available pool size
            print(f"Warning: Requested {num_neighbourhoods} neighbourhoods, but only {pool_size} are defined in the pool. Returning all {pool_size}.")
            self.bounds = self.neighbourhood_pool
            
            # Initialize the dictionary to hold Prosumer objects, keyed by neighbourhood name
            self.neighbourhoods = {name: [] for name in neighbourhood_names}
            self.num_neighbourhoods = pool_size
        
        else:
            # Randomly sample the required number of neighbourhoods
            selected_names = random.sample(neighbourhood_names, num_neighbourhoods)
            
            # Create the bounds dictionary containing only the selected neighbourhoods
            self.bounds = {name: self.neighbourhood_pool[name] for name in selected_names}

            # Initialize the prosumer list for each selected neighbourhood
            self.neighbourhoods = {name: [] for name in selected_names}
            self.num_neighbourhoods = num_neighbourhoods
    
    def assign_prosumer_to_neighbourhood(self):
        """
        Randomly selects a neighbourhood and generates random coordinates (latitude, longitude)
        within that neighbourhood's bounding box.

        Returns:
            tuple: A tuple containing the selected neighbourhood name (str), latitude (float), and longitude (float).
        """
        # Choose a neighbourhood at random from the selected ones
        neighbourhood = np.random.choice(list(self.neighbourhoods.keys()))
        min_lat, max_lat, min_lon, max_lon = self.bounds[neighbourhood]
        
        # Generate a random latitude and longitude within the neighbourhood's bounds
        latitude = np.random.uniform(min_lat, max_lat)
        longitude = np.random.uniform(min_lon, max_lon)
        return neighbourhood, latitude, longitude
        
    def add_prosumer_to_neighbourhood(self, prosumer, neighbourhood):
        """
        Adds a Prosumer object to the list of prosumers belonging to the specified neighbourhood.

        Args:
            prosumer (Prosumer): The Prosumer object to be added.
            neighbourhood (str): The name of the neighbourhood.

        Returns:
            None
        """
        self.neighbourhoods[neighbourhood].append(prosumer)

    def get_neighbourhood_prosumers(self, neighbourhood):
        """
        Retrieves the list of Prosumer objects belonging to a specific neighbourhood.

        Args:
            neighbourhood (str): The name of the neighbourhood.

        Returns:
            list: A list of Prosumer objects.
        """
        return self.neighbourhoods[neighbourhood]
    
    def plot_neighbourhoods(self):
        """
        Generates and displays a geographical plot showing the neighbourhood boundaries (Polygons)
        and the location of each prosumer (Points) on a map background.

        Returns:
            None: Displays a plot using matplotlib.
        """
        # --- Prepare Neighbourhood GeoDataFrame (Polygons) ---
        polygons = {}
        for name, bounds in self.bounds.items():
            min_lat, max_lat, min_lon, max_lon = bounds

            # Define the corners of the bounding box as a shapely Polygon
            poly = Polygon([
                (min_lon, min_lat),
                (max_lon, min_lat),
                (max_lon, max_lat),
                (min_lon, max_lat)
            ])
            polygons[name] = poly
        
        # Create a GeoDataFrame for the neighbourhood boundaries
        neighbourhood_gdf = gpd.GeoDataFrame(
            data=pd.DataFrame(list(polygons.keys()), columns=['Name']),
            geometry=list(polygons.values()),
            crs="EPSG:4326" # Original WGS 84 coordinate system
        )

        # Reproject to Web Mercator (EPSG:3857) for compatibility with contextily basemaps
        neighbourhood_gdf = neighbourhood_gdf.to_crs(epsg=3857)

        # --- Prepare Prosumer GeoDataFrame (Points) ---
        prosumer_data = []
        for neighbourhood_name, prosumer_list in self.neighbourhoods.items():
            for prosumer in prosumer_list:
                # Gather location and ID data for all prosumers
                prosumer_data.append({
                    'id': prosumer.id,
                    'latitude': prosumer.latitude,
                    'longitude': prosumer.longitude,
                    'neighbourhood': neighbourhood_name
                })
        
        # Convert the prosumer data list into a regular DataFrame
        prosumer_df = pd.DataFrame(prosumer_data)
        
        # Convert to GeoDataFrame using shapely Points from lat/lon columns
        prosumer_gdf = gpd.GeoDataFrame(
            prosumer_df,
            geometry=gpd.points_from_xy(prosumer_df.longitude, prosumer_df.latitude),
            crs="EPSG:4326"
        )

        # Reproject to Web Mercator to match neighbourhood GeoDataFrame
        prosumer_gdf = prosumer_gdf.to_crs(epsg=3857)
        
        # --- Plot the map ---
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        # Plot neighbourhoods (Polygons)
        neighbourhood_gdf.plot(
            ax=ax, 
            column='Name', # Use the neighbourhood name for color differentiation
            cmap='tab10',
            edgecolor='black',
            linewidth=1,
            legend=True,
            alpha=0.4 # Make them slightly transparent
        )

        # Plot prosumers (Points) on top
        prosumer_gdf.plot(
            ax=ax,
            column='neighbourhood', # Color points based on their neighbourhood
            cmap='tab10',
            marker='o',
            markersize=5,
            alpha=1,
            legend=True,
            legend_kwds={
                'title': "Neighbourhoods",
                'bbox_to_anchor': (1.05, 1), # (x, y) coordinates relative to the plot
                'loc': 'upper left',         # The corner of the legend being placed
                'borderaxespad': 0.          # Padding between the axes and the legend
            }
        )

        # Add the basemap for context
        cx.add_basemap(ax, crs=neighbourhood_gdf.crs.to_string(), source=cx.providers.CartoDB.Positron)
        
        # Final formatting
        ax.set_title("Prosumer Distribution within Neighbourhoods")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        plt.show()