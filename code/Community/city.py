import random
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt
import pandas as pd
import contextily as cx

class City:
    def __init__(self, num_prosumers, num_neighbourhoods, neighbourhood_pool):
        """
        num_prosumers: total number of prosumers
        num_neighbourhoods: number of neighbourhoods
        neighbourhood_pool: bounds of the neighbourhoods
        """
        self.num_prosumers = num_prosumers
        self.neighbourhood_pool = neighbourhood_pool

        pool_size = len(self.neighbourhood_pool)
        # Get all neighbourhood names
        neighbourhood_names = list(self.neighbourhood_pool.keys())
        
        if num_neighbourhoods <= 0:
            raise ValueError("Number of neighbourhoods must be positive.")
        
        elif num_neighbourhoods > pool_size:
            print(f"Warning: Requested {num_neighbourhoods} neighbourhoods, but only {pool_size} are defined in the pool. Returning all {pool_size}.")
            self.bounds = self.neighbourhood_pool
            self.neighbourhoods = {name: [] for name in neighbourhood_names}
            self.num_neighbourhoods = pool_size
        
        else:
            # Randomly sample the required number of neighbourhoods
            selected_names = random.sample(neighbourhood_names, num_neighbourhoods)
            
            # Create the dictionary
            self.bounds = {name: self.neighbourhood_pool[name] for name in selected_names}
            self.neighbourhoods = {name: [] for name in selected_names}
            self.num_neighbourhoods = num_neighbourhoods
    
    def assign_prosumer_to_neighbourhood(self):
        # Choose a neighbourhood at random
        neighbourhood = np.random.choice(list(self.neighbourhoods.keys()))
        min_lat, max_lat, min_lon, max_lon = self.bounds[neighbourhood]
        
        # Generate a random latitude and longitude within the neighbourhood
        latitude = np.random.uniform(min_lat, max_lat)
        longitude = np.random.uniform(min_lon, max_lon)
        return neighbourhood, latitude, longitude
        
    def add_prosumer_to_neighbourhood(self, prosumer, neighbourhood):
        self.neighbourhoods[neighbourhood].append(prosumer)

    def get_neighbourhood_prosumers(self, neighbourhood):
        return self.neighbourhoods[neighbourhood]
    
    def plot_neighbourhoods(self):
        # Prepare Neighbourhood GeoDataFrame (Polygons)
        # Create a list of shapely Polygons for the neighbourhoods
        polygons = {}
        for name, bounds in self.bounds.items():
            min_lat, max_lat, min_lon, max_lon = bounds
            # Define the corners of the bounding box
            poly = Polygon([
                (min_lon, min_lat),
                (max_lon, min_lat),
                (max_lon, max_lat),
                (min_lon, max_lat)
            ])
            polygons[name] = poly
        
        # Create a GeoDataFrame for the neighbourhoods
        neighbourhood_gdf = gpd.GeoDataFrame(
            data=pd.DataFrame(list(polygons.keys()), columns=['Name']),
            geometry=list(polygons.values()),
            crs="EPSG:4326" # WGS 84 coordinate system
        )
        neighbourhood_gdf = neighbourhood_gdf.to_crs(epsg=3857)

        # Prepare Prosumer GeoDataFrame (Points)
        prosumer_data = []
        for neighbourhood_name, prosumer_list in self.neighbourhoods.items():
            for prosumer in prosumer_list:
                prosumer_data.append({
                    'id': prosumer.id,
                    'latitude': prosumer.latitude,
                    'longitude': prosumer.longitude,
                    'neighbourhood': neighbourhood_name
                })
        
        # Convert the prosumer data list into a regular DataFrame
        prosumer_df = pd.DataFrame(prosumer_data)
        
        # Convert to GeoDataFrame using shapely Points
        prosumer_gdf = gpd.GeoDataFrame(
            prosumer_df,
            geometry=gpd.points_from_xy(prosumer_df.longitude, prosumer_df.latitude),
            crs="EPSG:4326"
        )
        prosumer_gdf = prosumer_gdf.to_crs(epsg=3857)
        
        # Plot the map
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        # Plot neighbourhoods (Polygons)
        neighbourhood_gdf.plot(
            ax=ax, 
            column='Name', # Use the name for color differentiation
            cmap='tab10',
            edgecolor='black',
            linewidth=1,
            legend=True,
            alpha=0.4           # Make them slightly transparent
        )

        # Plot prosumers (Points) on top
        prosumer_gdf.plot(
            ax=ax,
            column='neighbourhood', # Color points based on their neighbourhood
            cmap='tab10',
            marker='o',
            markersize=5,
            legend=False,
            alpha=1
        )

        # Add the basemap to the axes
        cx.add_basemap(ax, crs=neighbourhood_gdf.crs.to_string(), source=cx.providers.CartoDB.Positron)
        
        # Final formatting
        ax.set_title("Prosumer Distribution within Neighbourhoods")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        plt.show()