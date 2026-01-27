import csv
from pathlib import Path
import numpy as np
import itertools
from scipy.optimize import minimize, LinearConstraint, Bounds
from prettytable import PrettyTable
import argparse


def parse_currency(value_str):
    """Parse currency string by removing commas and converting to float."""
    return float(value_str.replace(",", ""))


def parse_asset_allocation(filepath):
    """Parse asset allocation CSV and return portfolio data."""
    current_category = ""
    category_totals = {}
    allocations = {}
    allocation_targets = {}

    filepath = Path(filepath)

    with filepath.open("r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar='"')

        for row_id, row in enumerate(reader):
            # Skip header row and get total from second row
            if row_id == 0:
                continue
            elif row_id == 1:
                current_total = parse_currency(row[7])
                continue

            category = row[1]
            # Process category headers (Equity, Bonds)
            if category in ["Equity", "Bonds"] and category != current_category:
                current_category = category
                category_totals[current_category] = float(row[5])
                continue

            # Process allocation rows within current category
            if category == current_category and row[4] == "":
                asset_name = row[2]
                allocations[asset_name] = parse_currency(row[10])

                # Calculate target allocation
                target_percentage = parse_currency(row[5])
                category_weight = category_totals[category] / 100
                allocation_targets[asset_name] = target_percentage * category_weight

    return {
        "current_total": current_total,
        "category_totals": category_totals,
        "allocations": allocations,
        "allocation_targets": allocation_targets,
    }


def calculate_allocation_deviation(values, target_weights):
    total = sum(values)
    deviation = (values / (target_weights * total)) - 1
    return deviation


def iterate(current_values, additional_investment, target_weights, n_assets=2):
    num_investments = len(current_values)
    # If n_assets is None or greater than total investments, use all
    if n_assets is None or n_assets > num_investments:
        n_assets = num_investments

    investment_indices = range(num_investments)

    # We will store the best result found so far here
    min_overall_sse = float("inf")
    best_combination = None
    best_allocations = None

    best_allocations = None

    # Constraints: sum of allocations equals additional_investment
    # cons = {"type": "eq", "fun": lambda x: np.sum(x) - additional_investment}

    constraint_matrix = np.ones((1, n_assets))
    linear_constraint = LinearConstraint(
        constraint_matrix, [additional_investment], [additional_investment]
    )

    # Bounds: each allocation must be non-negative
    # bounds = [(0, additional_investment) for _ in range(n_assets)]
    # Use Bounds object for clearer definition
    bounds = Bounds(np.zeros(n_assets), np.full(n_assets, additional_investment))

    # Initial guess: distribute evenly
    initial_guess = [additional_investment / n_assets] * n_assets

    # Iterate over all combinations of size n_assets
    for combination in itertools.combinations(investment_indices, n_assets):
        result = minimize(
            fun=calculate_sse,
            x0=initial_guess,
            args=(
                combination,
                current_values,
                target_weights,
            ),
            method="trust-constr",
            bounds=bounds,
            constraints=[linear_constraint],
            jac=calculate_jacobian,
            hess=calculate_hessian,
        )

        if result.fun < min_overall_sse:
            min_overall_sse = result.fun
            best_combination = combination
            best_allocations = result.x

    return {
        "best_combination": best_combination,
        "best_allocations": best_allocations,
        "min_sse": min_overall_sse,
    }


def calculate_sse(
    allocations,
    model_indices,
    current_vals,
    target_weights,
):
    # Create a copy of the portfolio to modify
    new_values = current_vals.copy()

    # Allocate the new money based on the split
    model_indices = np.array(model_indices, dtype=int)
    new_values[model_indices] += allocations

    # Calculate the new weights
    new_weights = calculate_allocation_deviation(new_values, target_weights)

    # Calculate and return the Sum of Squared Errors
    sse = np.sum(new_weights**2)
    return sse


def calculate_jacobian(
    allocations,
    model_indices,
    current_vals,
    target_weights,
):
    """Calculate the gradient of the SSE function."""
    total_value = np.sum(current_vals) + np.sum(allocations)

    model_indices = np.array(model_indices, dtype=int)

    # Get values for the active assets
    active_current = current_vals[model_indices]
    active_targets = target_weights[model_indices]
    active_allocs = allocations

    # Current values after allocation
    active_final_values = active_current + active_allocs

    # The term inside the square: (V / (T * Total)) - 1
    # Derivative of SSE = sum(w^2) w.r.t alloc_i
    # = 2 * w_i * d(w_i)/d(alloc_i)
    # w_i = (V_i / (T_i * Total)) - 1
    # d(w_i)/d(a_i) = 1 / (T_i * Total)

    denom = active_targets * total_value
    weights = (active_final_values / denom) - 1

    grad = 2 * weights * (1 / denom)
    return grad


def calculate_hessian(
    allocations,
    model_indices,
    current_vals,
    target_weights,
):
    """Calculate the Hessian matrix of the SSE function."""
    total_value = np.sum(current_vals) + np.sum(allocations)
    model_indices = np.array(model_indices, dtype=int)
    active_targets = target_weights[model_indices]

    # Second derivative
    # d2(SSE)/d(a_i)^2 = 2 * (1 / (T_i * Total))^2
    # Off-diagonal terms are 0 (assuming Total is treated as constant or approximations are fine)

    denom = active_targets * total_value
    diag_values = 2 * (1 / denom) ** 2

    return np.diag(diag_values)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Calculate optimal asset allocation.")
    parser.add_argument(
        "--n-assets",
        "-n",
        type=int,
        default=2,
        help="Number of assets to distribute investment across (default: 2)",
    )
    parser.add_argument(
        "filepath",
        help="Path to the asset allocation CSV file",
    )
    parser.add_argument(
        "additional_investment",
        type=float,
        help="Amount of additional cash to invest",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    filepath = args.filepath
    n_assets = args.n_assets
    additional_investment = args.additional_investment

    portfolio_data = {}
    investment_names = []

    try:
        portfolio_data = parse_asset_allocation(filepath)
        current_values = np.array(list(portfolio_data["allocations"].values()))
        target_weights = np.array(list(portfolio_data["allocation_targets"].values()))
        target_weights /= 100

    except FileNotFoundError:
        print(f"Error: Could not find file at {filepath}")
        exit(1)

    except Exception as e:
        print(f"Error processing file: {e}")
        exit(1)

    result = iterate(current_values, additional_investment, target_weights, n_assets)
    investment_names = list(portfolio_data["allocations"])

    best_combination = result["best_combination"]
    best_allocations = result["best_allocations"]

    print("--- Optimal Allocation Found ---")
    print(f"Invest in: {[investment_names[i] for i in best_combination]}")
    for idx, amount in zip(best_combination, best_allocations):
        name = investment_names[idx]
        print(
            f"Optimal allocation: £{amount:,.2f} to '{name}' ({amount / additional_investment * 100:.2f}%)"
        )
    print("\n")
    print(f"SSE: {result['min_sse']:.2e}\n")

    final_values = current_values.copy()
    for idx, amount in zip(best_combination, best_allocations):
        final_values[idx] += amount

    final_weights = calculate_allocation_deviation(final_values, target_weights)
    old_weight = calculate_allocation_deviation(current_values, target_weights)

    print("--- Portfolio Weights ---")
    output = PrettyTable()
    output.field_names = ["Investment", "Target", "Old Deviance", "New Deviance"]
    output.align = "r"
    output.align["Investment"] = "l"
    for i in range(len(investment_names)):
        output.add_row(
            [
                investment_names[i],
                f"{target_weights[i]:.1%}",
                f"{old_weight[i]:.1%}",
                f"{final_weights[i]:.1%}",
            ]
        )
    print(output)


if __name__ == "__main__":
    main()
