import csv
from pathlib import Path
import numpy as np
import itertools
from scipy.optimize import minimize_scalar
from prettytable import PrettyTable


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


def iterate(current_values, additional_investment, target_weights):
    num_investments = len(current_values)
    investment_indices = range(num_investments)

    # We will store the best result found so far here
    min_overall_sse = float("inf")
    best_pair = None
    best_split = None

    for pair in itertools.combinations(investment_indices, 2):
        result = minimize_scalar(
            fun=calculate_sse,
            bounds=(100, additional_investment - 100),
            args=(
                pair,
                current_values,
                additional_investment,
                target_weights,
            ),
        )

        if result.fun < min_overall_sse:
            min_overall_sse = result.fun
            best_pair = pair
            best_split = result.x

    return {
        "best_pair": best_pair,
        "best_split": best_split,
        "min_sse": min_overall_sse,
    }


def calculate_sse(
    split_amount,
    pair_indices,
    current_vals,
    additional_investment,
    target_weights,
):
    # Create a copy of the portfolio to modify
    new_values = current_vals.copy()

    # Allocate the new money based on the split
    idx1, idx2 = pair_indices
    new_values[idx1] += split_amount
    new_values[idx2] += additional_investment - split_amount

    # Calculate the new weights
    new_weights = calculate_allocation_deviation(new_values, target_weights)

    # Calculate and return the Sum of Squared Errors
    sse = np.sum(new_weights**2)
    return sse


if __name__ == "__main__":
    filepath = "/Users/jonny/Downloads/Asset_Allocation.csv"

    portfolio_data = {}
    investment_names = []
    additional_investment = 1500

    try:
        portfolio_data = parse_asset_allocation(filepath)
        current_values = np.array(list(portfolio_data["allocations"].values()))
        target_weights = np.array(list(portfolio_data["allocation_targets"].values()))
        target_weights /= 100

    except FileNotFoundError:
        print(f"Error: Could not find file at {filepath}")

    except Exception as e:
        print(f"Error processing file: {e}")

    result = iterate(current_values, additional_investment, target_weights)
    investment_names = list(portfolio_data["allocations"])

    best_pair = result["best_pair"]
    best_split = result["best_split"]

    allocation1 = best_split
    allocation2 = additional_investment - best_split
    name1 = investment_names[best_pair[0]]
    name2 = investment_names[best_pair[1]]

    print("--- Optimal Allocation Found ---")
    print(f"Invest in this pair: '{name1}' and '{name2}'")
    print(
        f"Optimal allocation: £{allocation1:,.2f} to '{name1}' and £{allocation2:,.2f} to '{name2}'\n"
    )

    final_values = current_values.copy()
    final_values[best_pair[0]] += allocation1
    final_values[best_pair[1]] += allocation2

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
