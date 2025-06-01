"""Configuration module for the MFS©\VRPD project.

This module defines the configuration parameters for training, fleet specifications,
ALNS settings, MILP settings, instance generation, cost parameters, and working hours.
These constants serve as the single source of truth for all modules and are used 
throughout the project to guarantee consistency.
"""

import math
from dataclasses import dataclass, field
from typing import List

def compute_initial_temperature(relative_delta: float, acceptance_probability: float) -> float:
    """Compute initial temperature for simulated annealing.
    
    Given that a solution that is relative_delta (e.g. 30% worse -> 0.30)
    than the current solution should be accepted with acceptance_probability,
    this function computes T satisfying:
    
        exp(-relative_delta / T) == acceptance_probability

    Args:
        relative_delta (float): The relative difference (e.g., 0.30 for 30% worse).
        acceptance_probability (float): Desired acceptance probability (0 < p < 1).

    Returns:
        float: The computed initial temperature.

    Raises:
        ValueError: If acceptance_probability is not in the interval (0, 1).
    """
    if acceptance_probability <= 0.0 or acceptance_probability >= 1.0:
        raise ValueError("Acceptance probability must be in the interval (0, 1).")
    return -relative_delta / math.log(acceptance_probability)


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for training parameters and simulated annealing settings."""
    niters_small: int = 2000  # Number of iterations for small instances.
    niters_large: int = 4000  # Number of iterations for large instances.
    # Compute initial_temperature such that a solution 30% worse (delta=0.30)
    # is accepted with probability 0.5.
    initial_temperature: float = field(default_factory=lambda: compute_initial_temperature(0.30, 0.5))
    cooling_rate: float = 0.97  # Cooling factor per iteration.
    max_non_improve: int = 300  # Maximum consecutive iterations without improvement.


@dataclass(frozen=True)
class FleetTruckConfig:
    """Configuration for truck fleet parameters."""
    capacity: int = 500         # Capacity in kg.
    speed: float = 35.0         # Speed in km/h.
    cost_per_km: float = 1.0      # Cost per kilometer.


@dataclass(frozen=True)
class FleetDroneConfig:
    """Configuration for drone fleet parameters."""
    capacity: int = 30          # Capacity in kg.
    speed: float = 70.0         # Speed in km/h.
    endurance: int = 60         # Endurance in minutes.
    eligibility_percent: int = 90  # Percentage of customers eligible for drone service.


@dataclass(frozen=True)
class FleetConfig:
    """Aggregate fleet configuration including truck and drone settings."""
    truck: FleetTruckConfig = FleetTruckConfig()
    drone: FleetDroneConfig = FleetDroneConfig()


@dataclass(frozen=True)
class ALNSRemovalNodesConfig:
    """ALNS removal nodes configuration parameters."""
    r_L: float = 0.15           # Lower bound ratio for removal nodes.
    r_U_small: float = 0.5      # Upper bound ratio for small instances.
    r_U_large: float = 0.3      # Upper bound ratio for large instances.


@dataclass(frozen=True)
class ALNSAdaptiveSelectionConfig:
    """ALNS adaptive operator selection configuration parameters."""
    sigma1: int = 33            # Reward for top-performing operator.
    sigma2: int = 13            # Secondary reward.
    sigma3: int = 9             # Tertiary reward.
    sigma4: int = 1             # Minimal reward.
    reaction_index: float = 0.6 # Reaction index for weight updates.


@dataclass(frozen=True)
class ALNSParameters:
    """Aggregate configuration for ALNS parameters."""
    removal_nodes: ALNSRemovalNodesConfig = ALNSRemovalNodesConfig()
    adaptive_selection: ALNSAdaptiveSelectionConfig = ALNSAdaptiveSelectionConfig()


@dataclass(frozen=True)
class MILPParameters:
    """MILP solver configuration parameters."""
    time_limit: int = 7200   # Time limit for MILP solver in seconds.
    big_M: float = 1e6       # Big-M constant used in MILP formulations.


@dataclass(frozen=True)
class InstanceGenerationConfig:
    """Configuration for synthetic instance generation."""
    customer_area_sizes: List[int] = field(default_factory=lambda: [30, 40, 50])
    num_instances: int = 15
    instance_sizes_small: List[int] = field(default_factory=lambda: [10, 25])
    instance_sizes_medium: List[int] = field(default_factory=lambda: [50, 75])
    instance_sizes_large: List[int] = field(default_factory=lambda: [100])


@dataclass(frozen=True)
class CostParameters:
    """Cost configuration parameters for trucks and drones."""
    truck_cost: float = 1.0   # Cost per km for trucks.
    drone_cost: float = 0.2   # Cost per km for drones.


@dataclass(frozen=True)
class GlobalConfig:
    """Global configuration aggregating all settings."""
    training: TrainingConfig = TrainingConfig()
    fleet: FleetConfig = FleetConfig()
    alns_parameters: ALNSParameters = ALNSParameters()
    milp_parameters: MILPParameters = MILPParameters()
    instance_generation: InstanceGenerationConfig = InstanceGenerationConfig()
    cost_parameters: CostParameters = CostParameters()
    working_hours: int = 8    # Maximum working hours for vehicles.


# Instantiate the global configuration object.
CONFIG = GlobalConfig()

# When other modules need access to configuration constants,
# they can import CONFIG from this module.
if __name__ == "__main__":
    # For testing purposes, print the configuration to verify values.
    print("Training Configuration:")
    print(f"  Small niters: {CONFIG.training.niters_small}")
    print(f"  Large niters: {CONFIG.training.niters_large}")
    print(f"  Initial Temperature: {CONFIG.training.initial_temperature:.4f}")
    print(f"  Cooling Rate: {CONFIG.training.cooling_rate}")
    print(f"  Max Non-improve Iterations: {CONFIG.training.max_non_improve}")

    print("\nFleet Configuration:")
    print(f"  Truck Capacity: {CONFIG.fleet.truck.capacity} kg")
    print(f"  Truck Speed: {CONFIG.fleet.truck.speed} km/h")
    print(f"  Truck Cost per km: {CONFIG.fleet.truck.cost_per_km}")
    print(f"  Drone Capacity: {CONFIG.fleet.drone.capacity} kg")
    print(f"  Drone Speed: {CONFIG.fleet.drone.speed} km/h")
    print(f"  Drone Endurance: {CONFIG.fleet.drone.endurance} minutes")
    print(f"  Drone Eligibility: {CONFIG.fleet.drone.eligibility_percent}%")

    print("\nALNS Parameters:")
    print(f"  Removal Nodes r_L: {CONFIG.alns_parameters.removal_nodes.r_L}")
    print(f"  Removal Nodes r_U (small): {CONFIG.alns_parameters.removal_nodes.r_U_small}")
    print(f"  Removal Nodes r_U (large): {CONFIG.alns_parameters.removal_nodes.r_U_large}")
    print(f"  Adaptive sigma1: {CONFIG.alns_parameters.adaptive_selection.sigma1}")
    print(f"  Adaptive sigma2: {CONFIG.alns_parameters.adaptive_selection.sigma2}")
    print(f"  Adaptive sigma3: {CONFIG.alns_parameters.adaptive_selection.sigma3}")
    print(f"  Adaptive sigma4: {CONFIG.alns_parameters.adaptive_selection.sigma4}")
    print(f"  Reaction Index: {CONFIG.alns_parameters.adaptive_selection.reaction_index}")

    print("\nMILP Parameters:")
    print(f"  Time Limit: {CONFIG.milp_parameters.time_limit} seconds")
    print(f"  Big-M: {CONFIG.milp_parameters.big_M}")

    print("\nInstance Generation Parameters:")
    print(f"  Customer Area Sizes: {CONFIG.instance_generation.customer_area_sizes}")
    print(f"  Number of Instances: {CONFIG.instance_generation.num_instances}")
    print(f"  Small Instance Sizes: {CONFIG.instance_generation.instance_sizes_small}")
    print(f"  Medium Instance Sizes: {CONFIG.instance_generation.instance_sizes_medium}")
    print(f"  Large Instance Sizes: {CONFIG.instance_generation.instance_sizes_large}")

    print("\nCost Parameters:")
    print(f"  Truck Cost: {CONFIG.cost_parameters.truck_cost}")
    print(f"  Drone Cost: {CONFIG.cost_parameters.drone_cost}")

    print(f"\nWorking Hours: {CONFIG.working_hours} hours")
