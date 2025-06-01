"""main.py

This is the entry point of the MFS©\VRPD project. It orchestrates the entire workflow:
  1. Load configuration parameters from config.py.
  2. Generate a synthetic instance using DatasetLoader.
  3. Select the appropriate model (MILPModel for small instances, ALNSModel for medium/large instances).
  4. Train the model using the Trainer which performs optimization.
  5. Evaluate the final solution using the Evaluation module and report key performance metrics.
  
Usage:
  python main.py [--instance_type {small,medium,large}] [--num_customers NUM]

Default values:
  --instance_type: "small"
  --num_customers: (if not provided, defaults are chosen from config)
  
This file uses strong typing, explicit variable definitions, and sets default values per configuration.
"""

import argparse
import logging
import sys

# Import global configuration
from config import CONFIG
# Import DatasetLoader for instance generation
from dataset_loader import DatasetLoader
# Import MILPModel and ALNSModel for problem formulation
from model import MILPModel, ALNSModel
# Import Trainer to manage the optimization process
from trainer import Trainer
# Import Evaluation to compute and visualize performance metrics
from evaluation import Evaluation


def main() -> None:
    """Main function to execute the MFS©\VRPD optimization process."""
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(
        description="MFS©\VRPD Optimization: Solve the multi-visit flexible-docking VRP with drones."
    )
    parser.add_argument(
        "--instance_type",
        type=str,
        default="small",
        choices=["small", "medium", "large"],
        help="Type of instance to generate (small/medium/large). Default: small."
    )
    parser.add_argument(
        "--num_customers",
        type=int,
        default=None,
        help="Optional number of customers to generate. If omitted, defaults from config are used."
    )
    args = parser.parse_args()

    # Configure logging to output time, level and message.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stdout
    )
    logging.info("Starting MFS©\VRPD optimization process...")

    # Step 1: Generate/load instance data using DatasetLoader.
    logging.info("Loading instance data...")
    dataset_loader = DatasetLoader(config=CONFIG, instance_type=args.instance_type, num_customers=args.num_customers)
    instance = dataset_loader.load_data()
    logging.info(f"Instance loaded with {instance.num_customers} customers in a region of size {instance.area_size}.")

    # Step 2: Select the appropriate model based on instance size.
    # According to our design: MILPModel is used for 'small' instances; ALNSModel for others.
    instance_type_lower = instance.instance_type.lower()
    if instance_type_lower == "small":
        logging.info("Instance type is small. Using MILPModel (exact MILP formulation) for optimization.")
        model = MILPModel(instance=instance, config=CONFIG)
    else:
        logging.info("Instance type is medium/large. Using ALNSModel (heuristic ALNS method) for optimization.")
        model = ALNSModel(instance=instance, config=CONFIG)

    # Step 3: Create the Trainer with the chosen model, instance data, and configuration.
    logging.info("Initializing Trainer for the optimization process...")
    trainer = Trainer(model=model, instance=instance, config=CONFIG)

    # Step 4: Run the optimization process through Trainer.
    logging.info("Running optimization (training) process...")
    final_solution = trainer.train()
    logging.info("Optimization finished.")

    # Step 5: Evaluate the final solution.
    logging.info("Evaluating the final solution...")
    evaluator = Evaluation(solution=final_solution, instance=instance)
    metrics = evaluator.evaluate()
    summary_table = evaluator.generate_summary_table(metrics)
    
    # Log the evaluation results using summary table.
    logging.info("Evaluation Metrics Summary:")
    print(summary_table.to_string(index=False))
    
    # Optionally, visualize the cost breakdown and other metrics.
    evaluator.visualize(metrics)
    
    logging.info("MFS©\VRPD optimization process complete.")


if __name__ == "__main__":
    main()
