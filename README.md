# MFS/VRPD Flexible-Docking Solver

This repository contains a research prototype for solving the **Multi-Visit Flexible Docking Vehicle Routing Problem with Drones (MFS/VRPD)**.  The code base is intentionally small and focuses on clarity so new contributors can easily understand the workflow and extend the models.

## Project Overview
The optimization process generates a synthetic routing instance, selects either an exact MILP solver or a heuristic ALNS solver, trains the model, and evaluates the resulting solution.  The `main.py` file orchestrates these steps:

1. Load configuration from `config.py`.
2. Generate an instance with `DatasetLoader`.
3. Choose `MILPModel` for small instances or `ALNSModel` for larger ones.
4. Optimize the instance using `Trainer`.
5. Compute metrics and visualization via `Evaluation`.

Source: lines 1–18 of `main.py` describe this sequence【F:main.py†L1-L18】.

## Module Structure

| Module | Purpose |
| ------ | ------- |
| `config.py` | Defines all configuration dataclasses such as `TrainingConfig`, `FleetConfig`, and `ALNSParameters`【F:config.py†L37-L66】【F:config.py†L101-L112】. Constants are imported by other modules for consistency. |
| `dataset_loader.py` | Generates synthetic instances of customer locations, demands, and distance/time matrices. Returns an `InstanceData` structure containing these arrays【F:dataset_loader.py†L23-L61】【F:dataset_loader.py†L96-L174】. |
| `model.py` | Implements `MILPModel` (exact MILP using Gurobi) and `ALNSModel` (adaptive large-neighborhood search heuristic)【F:model.py†L1-L11】【F:model.py†L27-L31】【F:model.py†L371-L377】. |
| `trainer.py` | Coordinates training by calling either `MILPModel.solve()` or `ALNSModel.optimize()` and logs progress【F:trainer.py†L1-L9】【F:trainer.py†L51-L60】. |
| `evaluation.py` | Computes objective cost, cost breakdowns, service composition, and visualization utilities for the final solution【F:evaluation.py†L1-L19】【F:evaluation.py†L92-L146】【F:evaluation.py†L147-L193】. |
| `main.py` | Entry point that wires together the other modules and parses command line options【F:main.py†L1-L40】. |

## Key Classes and Functions

### Configuration
- **`TrainingConfig`**, **`FleetConfig`**, **`ALNSParameters`**, etc. – dataclasses holding parameters such as iteration counts, fleet capacities, and solver limits【F:config.py†L37-L79】【F:config.py†L84-L115】【F:config.py†L117-L127】.
- **`compute_initial_temperature`** – utility to compute the simulated annealing temperature from a relative delta and desired acceptance probability【F:config.py†L13-L34】.
- A global `CONFIG` object aggregates all settings for import by other modules【F:config.py†L117-L131】【F:config.py†L136-L138】.

### Dataset Generation
- **`InstanceData`** dataclass summarizes the generated problem instance, including coordinates, demand dictionary, and travel matrices【F:dataset_loader.py†L24-L49】.
- **`DatasetLoader`** generates instances based on the configuration; key steps include coordinate generation, demand creation, and distance/time computations【F:dataset_loader.py†L53-L61】【F:dataset_loader.py†L96-L181】.

### Optimization Models
- **`MILPModel`** – builds and solves a MILP formulation with decision variables for truck and drone routing, timing, endurance, and load constraints. Important methods:
  - `build_model()` constructs variables, objective, and constraints【F:model.py†L98-L163】【F:model.py†L160-L315】.
  - `solve()` optimizes the model and extracts truck and drone routes along with cost【F:model.py†L304-L366】.
- **`ALNSModel`** – adaptive large-neighborhood search heuristic. Main methods:
  - `initialize_solution()` creates a feasible starting solution using a two-phase heuristic【F:model.py†L420-L521】.
  - `optimize()` runs the iterative destroy/repair loop with simulated annealing acceptance【F:model.py†L535-L604】.
  - `apply_destroy_operator()` and `apply_repair_operator()` modify solutions probabilistically based on adaptive weights【F:model.py†L608-L733】【F:model.py†L693-L733】.
  - `compute_cost()` computes total cost combining truck and drone routes【F:model.py†L762-L787】.
  - `check_solution_feasible()` verifies every customer is visited exactly once【F:model.py†L788-L818】.

### Training and Evaluation
- **`Trainer`** executes the chosen model, logs runtime, and attaches metadata to the final solution【F:trainer.py†L20-L49】【F:trainer.py†L51-L80】.
- **`Evaluation`** calculates cost metrics, improvement percentages, and optionally visualizes results in a bar chart. Metrics are assembled in `evaluate()`【F:evaluation.py†L92-L146】【F:evaluation.py†L160-L193】.

## Module Interaction and Data Flow

```
main.py ──▶ DatasetLoader.load_data() ──▶ InstanceData
       │
       ├─▶ select model (MILPModel or ALNSModel)
       │        │
       │        ├─▶ Trainer.train()
       │        │      ├─▶ MILPModel.build_model()/solve()  OR
       │        │      └─▶ ALNSModel.optimize()
       │        └─ returns solution dict
       │
       └─▶ Evaluation(solution, instance)
              ├─▶ evaluate() → metrics
              └─▶ generate_summary_table()/visualize()
```
The `InstanceData` object produced by `DatasetLoader` contains distance and time matrices used by both models. Each model returns a solution dictionary including routes and objective cost, which `Trainer` augments with runtime. `Evaluation` reads both the solution and original instance to compute metrics.

## Installation and Usage

1. **Prerequisites**
   - Python 3.8+
   - `numpy`, `pandas`, `matplotlib`
   - `gurobipy` is required for solving MILP instances.
2. **Installation**
   - Clone the repository and install dependencies:
     ```bash
     pip install numpy pandas matplotlib gurobipy
     ```
3. **Running the solver**
   - Execute the project via `main.py`:
     ```bash
     python main.py --instance_type small --num_customers 10
     ```
   - `--instance_type` can be `small`, `medium`, or `large`. If `--num_customers` is omitted, sizes from `config.py` are used.

## Example
The evaluation module includes a demonstration that creates a dummy solution and prints the computed metrics. Running `python evaluation.py` outputs a summary table similar to:
```
Evaluation Metrics Summary:
   Objective Cost  Truck Cost  Drone Cost  Runtime (s)  # Truck Customers  # Drone Customers  # Drone Subroutes  Cost per Truck Customer  Cost per Drone Customer  Improvement (%)  Gap (%)
0           150.0        ...         ...         25.0                  3                 3                  2                     ...                     ...            ...       ...
```
Source: example provided in the script’s `__main__` block【F:evaluation.py†L276-L315】.

## Design Notes
- **Small vs. large instances** – `MILPModel` is designed for small instances; the heuristic `ALNSModel` scales to larger problems. The switch is made automatically in `main.py` based on the selected `instance_type`【F:main.py†L30-L37】.
- **Simulated annealing** – the initial temperature is derived from a 30% worse acceptance at probability 0.5【F:config.py†L13-L34】.
- **Road vs. drone distances** – truck travel distances are set 30% longer than Euclidean to model sparse rural networks (`ROAD_DISTANCE_FACTOR`)【F:dataset_loader.py†L20-L21】.

## Conclusion
This repository demonstrates an end-to-end prototype for flexible-docking VRPs using either MILP or ALNS methods. By keeping the modules lightweight and well-documented, new contributors can readily modify configurations, experiment with new operators, or integrate additional datasets.

