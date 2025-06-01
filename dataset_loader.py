"""
Dataset Loader Module for the MFS©\VRPD Project

This module generates synthetic problem instances that mimic a sparse rural network,
including customer/depot coordinates, demand generation (with drone eligibility),
and the computation of distance and time matrices for both truck (road) and drone (flight) layers.
Instances are generated based on configuration parameters specified in config.py.
"""

import numpy as np
import random
from typing import Dict, Any, List
from dataclasses import dataclass

# Import the global configuration from config.py.
from config import CONFIG


# Constant to simulate longer road distances in sparse rural networks.
ROAD_DISTANCE_FACTOR: float = 1.3  # e.g., truck routes are 30% longer than Euclidean distances.


@dataclass
class InstanceData:
    """
    Data structure to hold all generated instance information.
    Attributes:
        num_customers (int): Number of customer nodes.
        num_nodes (int): Total number of nodes including depot (node0) and duplicate depot (node n+1).
        instance_type (str): Type of instance ("small", "medium", "large").
        area_size (int): The dimension of the square region in which customers are located.
        coordinates (np.ndarray): Array of shape (num_nodes, 2) with x, y coordinates for each node.
        demands (Dict[int, Dict[str, Any]]): Dictionary mapping node index (1..n) to demand info.
            Each demand info is a dictionary with keys "delivery", "pickup", and "drone_eligible".
            Depot nodes (index 0 and index num_nodes-1) do not have demand.
        distance_matrix_truck (np.ndarray): Matrix of truck travel distances (n+2 x n+2).
        distance_matrix_drone (np.ndarray): Matrix of drone travel distances (n+2 x n+2).
        time_matrix_truck (np.ndarray): Matrix of truck travel times (hours) (n+2 x n+2).
        time_matrix_drone (np.ndarray): Matrix of drone travel times (hours) (n+2 x n+2).
    """
    num_customers: int
    num_nodes: int
    instance_type: str
    area_size: int
    coordinates: np.ndarray  # shape (num_nodes, 2)
    demands: Dict[int, Dict[str, Any]]
    distance_matrix_truck: np.ndarray
    distance_matrix_drone: np.ndarray
    time_matrix_truck: np.ndarray
    time_matrix_drone: np.ndarray


class DatasetLoader:
    """
    DatasetLoader generates a synthetic instance for the MFS©\VRPD.
    
    The instance includes:
     - Customer and depot coordinates (with depot at index 0 and duplicate depot at index n+1),
     - Demand generation for each customer with drone eligibility determined probabilistically,
     - Euclidean distance matrix (for drone flight) and adjusted truck distance matrix,
     - Travel time matrices for trucks and drones.
    """

    def __init__(self, config: Any, instance_type: str = "small", num_customers: int = None, seed: int = 42) -> None:
        """
        Initialize the DatasetLoader with a given configuration and instance parameters.
        
        Args:
            config: Global configuration object (from config.py).
            instance_type (str): Type of instance to generate ("small", "medium", or "large").
                Default is "small".
            num_customers (int, optional): Explicit number of customers. If not provided,
                a default is chosen based on instance_type.
            seed (int): Random seed for reproducibility. Default is 42.
        """
        self.config = config
        self.instance_type = instance_type.lower()
        # Set the random seed for reproducibility.
        np.random.seed(seed)
        random.seed(seed)
        
        # Decide number of customers based on provided value or defaults from configuration.
        if num_customers is not None:
            self.num_customers: int = num_customers
        else:
            if self.instance_type == "small":
                self.num_customers = self.config.instance_generation.instance_sizes_small[0]
            elif self.instance_type == "medium":
                self.num_customers = self.config.instance_generation.instance_sizes_medium[0]
            elif self.instance_type == "large":
                self.num_customers = self.config.instance_generation.instance_sizes_large[0]
            else:
                # Default to small if instance type is unrecognized.
                self.num_customers = self.config.instance_generation.instance_sizes_small[0]

    def load_data(self) -> InstanceData:
        """
        Generates and returns the synthetic instance data for the MFS©\VRPD.
        
        Returns:
            InstanceData: Instance data structure containing nodes, coordinates, demands, and distance/time matrices.
        """
        # Total number of nodes = depot (node 0) + customers (nodes 1..n) + duplicate depot (node n+1)
        n_customers: int = self.num_customers
        num_nodes: int = n_customers + 2

        # Select an area size randomly from the provided customer area sizes.
        area_sizes: List[int] = self.config.instance_generation.customer_area_sizes
        area_size: int = int(random.choice(area_sizes))
        
        # Generate coordinates:
        # Set depot coordinates at a fixed position (e.g., (0,0))
        depot_coord: np.ndarray = np.array([0.0, 0.0])
        # Generate customer coordinates uniformly at random in [0, area_size] for both x and y.
        customer_coords: np.ndarray = np.random.uniform(low=0.0, high=area_size, size=(n_customers, 2))
        # Duplicate depot (node n+1) will have same coordinates as depot.
        duplicate_depot_coord: np.ndarray = depot_coord.copy()
        
        # Build full coordinate array: first node is depot, then customers, then duplicate depot.
        coordinates: np.ndarray = np.vstack([depot_coord, customer_coords, duplicate_depot_coord])
        # coordinates shape: (num_nodes, 2)

        # Generate demands and drone eligibility for customer nodes (indices 1 to n_customers).
        demands: Dict[int, Dict[str, Any]] = {}
        drone_eligibility_threshold: float = self.config.fleet.drone.eligibility_percent / 100.0
        
        # For each customer (node index 1 to n_customers)
        for i in range(1, n_customers + 1):
            is_drone_eligible: bool = (np.random.rand() <= drone_eligibility_threshold)
            if is_drone_eligible:
                # For drone-eligible customers, generate demand in the interval [0, 20]
                delivery_demand: int = int(np.random.uniform(0, 21))
                pickup_demand: int = int(np.random.uniform(0, 21))
            else:
                # For truck-only customers, generate demand in the interval [20, 100]
                delivery_demand = int(np.random.uniform(20, 101))
                pickup_demand = int(np.random.uniform(20, 101))
            
            demands[i] = {
                "delivery": delivery_demand,
                "pickup": pickup_demand,
                "drone_eligible": is_drone_eligible
            }
        # Note: Depot nodes (index 0 and index num_nodes-1) are not assigned any demand.

        # Compute Euclidean distance matrix A_D for all nodes.
        # Using vectorized calculation: distances[i,j] = sqrt((x_i - x_j)^2 + (y_i - y_j)^2)
        diff: np.ndarray = coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]
        distance_matrix_drone: np.ndarray = np.sqrt(np.sum(diff ** 2, axis=2))
        
        # Compute truck distance matrix A_T as scaled Euclidean distance.
        distance_matrix_truck: np.ndarray = ROAD_DISTANCE_FACTOR * distance_matrix_drone
        
        # Compute travel times:
        # Truck travel time: distance / truck speed (speed in km/h gives time in hours)
        truck_speed: float = self.config.fleet.truck.speed
        time_matrix_truck: np.ndarray = distance_matrix_truck / truck_speed
        
        # Drone travel time: distance / drone speed (drone speed in km/h)
        drone_speed: float = self.config.fleet.drone.speed
        time_matrix_drone: np.ndarray = distance_matrix_drone / drone_speed
        
        # Create and return the InstanceData object
        instance_data: InstanceData = InstanceData(
            num_customers=n_customers,
            num_nodes=num_nodes,
            instance_type=self.instance_type,
            area_size=area_size,
            coordinates=coordinates,
            demands=demands,
            distance_matrix_truck=distance_matrix_truck,
            distance_matrix_drone=distance_matrix_drone,
            time_matrix_truck=time_matrix_truck,
            time_matrix_drone=time_matrix_drone
        )
        
        # Debug Logging: print summary information (can be replaced with proper logging if needed)
        print(f"Generated instance with {n_customers} customers in a {area_size}x{area_size} area.")
        print(f"Total nodes (including depot and duplicate depot): {num_nodes}")
        print(f"Truck speed: {truck_speed} km/h, Drone speed: {drone_speed} km/h")
        print(f"Depot coordinate: {depot_coord}, Duplicate depot coordinate: {duplicate_depot_coord}")
        
        return instance_data


# For testing purposes, if this module is run directly, generate an instance and print details.
if __name__ == "__main__":
    # Create a DatasetLoader using the global CONFIG from config.py
    loader = DatasetLoader(config=CONFIG, instance_type="small")
    instance = loader.load_data()
    
    # Print out key details of the generated instance.
    print("\nInstance Data Summary:")
    print(f"Instance Type: {instance.instance_type}")
    print(f"Area Size: {instance.area_size} x {instance.area_size}")
    print(f"Number of Customers: {instance.num_customers}")
    print(f"Number of Nodes: {instance.num_nodes}")
    print("Coordinates (first 5 nodes):")
    print(instance.coordinates[:5])
    print("Sample Demands (first 5 customer nodes):")
    for i in range(1, min(6, instance.num_customers + 1)):
        print(f"  Node {i}: {instance.demands[i]}")
    print("Distance Matrix (Truck) shape:", instance.distance_matrix_truck.shape)
    print("Time Matrix (Drone) shape:", instance.time_matrix_drone.shape)
