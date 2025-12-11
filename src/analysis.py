# src/analysis.py

import os
import csv
from typing import List, Dict, Any

import numpy as np
import matplotlib.pyplot as plt

from .config import BarrierOptionParams
from .monte_carlo import price_barrier_monte_carlo
from .binomial_tree import price_barrier_binomial_tree


def ensure_directories():
    os.makedirs("data", exist_ok=True)
    os.makedirs("plots", exist_ok=True)


def run_mc_convergence(
    params: BarrierOptionParams,
    n_paths: int,
    steps_list: List[int],
    seed: int | None = 42,
) -> List[Dict[str, Any]]:
    """
    Run Monte Carlo pricing for different time steps to study convergence.
    """
    results = []
    for n_steps in steps_list:
        price, std_err, runtime = price_barrier_monte_carlo(
            params, n_paths=n_paths, n_steps=n_steps, antithetic=True, seed=seed
        )
        results.append({
            "method": "MC",
            "n_steps": n_steps,
            "n_paths": n_paths,
            "price": price,
            "std_error": std_err,
            "runtime": runtime,
        })
        print(f"[MC] steps={n_steps}, price={price:.4f}, stderr={std_err:.4f}, time={runtime:.4f}s")
    return results


def run_tree_convergence(
    params: BarrierOptionParams,
    steps_list: List[int],
) -> List[Dict[str, Any]]:
    """
    Run binomial tree pricing for different numbers of steps.
    """
    results = []
    for n_steps in steps_list:
        price, runtime = price_barrier_binomial_tree(params, n_steps=n_steps)
        results.append({
            "method": "Tree",
            "n_steps": n_steps,
            "n_paths": None,
            "price": price,
            "std_error": None,
            "runtime": runtime,
        })
        print(f"[Tree] steps={n_steps}, price={price:.4f}, time={runtime:.4f}s")
    return results


def save_results_to_csv(records: List[Dict[str, Any]], filename: str = "data/results.csv"):
    fieldnames = ["method", "n_steps", "n_paths", "price", "std_error", "runtime"]
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)
    print(f"Saved results to {filename}")


def plot_convergence(records: List[Dict[str, Any]]):
    """
    Generate basic convergence and runtime plots from the records list.
    """
    ensure_directories()

    # Convert to arrays for convenience
    mc = [r for r in records if r["method"] == "MC"]
    tree = [r for r in records if r["method"] == "Tree"]

    # --- Price vs Steps ---
    plt.figure()
    if mc:
        mc_steps = [r["n_steps"] for r in mc]
        mc_price = [r["price"] for r in mc]
        plt.plot(mc_steps, mc_price, marker="o", label="MC price")
    if tree:
        tree_steps = [r["n_steps"] for r in tree]
        tree_price = [r["price"] for r in tree]
        plt.plot(tree_steps, tree_price, marker="s", label="Tree price")

    plt.xlabel("Number of Time Steps / Tree Steps")
    plt.ylabel("Option Price")
    plt.title("Price Convergence: Monte Carlo vs Binomial Tree")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/price_convergence.png", bbox_inches="tight")

    # --- Runtime vs Steps ---
    plt.figure()
    if mc:
        mc_steps = [r["n_steps"] for r in mc]
        mc_time = [r["runtime"] for r in mc]
        plt.plot(mc_steps, mc_time, marker="o", label="MC runtime")
    if tree:
        tree_steps = [r["n_steps"] for r in tree]
        tree_time = [r["runtime"] for r in tree]
        plt.plot(tree_steps, tree_time, marker="s", label="Tree runtime")

    plt.xlabel("Number of Steps")
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime vs Steps: Monte Carlo vs Binomial Tree")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/runtime_comparison.png", bbox_inches="tight")

    # --- MC Std Error vs Steps (if available) ---
    if mc:
        plt.figure()
        mc_steps = [r["n_steps"] for r in mc]
        mc_err = [r["std_error"] for r in mc]
        plt.plot(mc_steps, mc_err, marker="o")
        plt.xlabel("Number of Steps")
        plt.ylabel("MC Standard Error")
        plt.title("Monte Carlo Standard Error vs Time Steps")
        plt.grid(True)
        plt.savefig("plots/mc_std_error.png", bbox_inches="tight")

    print("Saved plots in 'plots/' directory")
