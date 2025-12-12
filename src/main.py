# src/main.py

from .config import DEFAULT_PARAMS
from .analysis import (
    ensure_directories,
    run_mc_convergence,
    run_tree_convergence,
    save_results_to_csv,
    plot_convergence,
)


def main():
    ensure_directories()

    params = DEFAULT_PARAMS

    # Choose experiment grids
    mc_steps_list = [50, 100, 200, 400, 800]       # time steps for MC
    tree_steps_list = [50, 100, 200, 400, 800]     # steps in tree
    n_paths = 50_000                               # MC paths

    # Run experiments
    mc_results = run_mc_convergence(params, n_paths=n_paths, steps_list=mc_steps_list, seed=123)
    tree_results = run_tree_convergence(params, steps_list=tree_steps_list)

    # Combine and save
    all_records = mc_results + tree_results
    save_results_to_csv(all_records, filename="data/results.csv")

    # Make basic plots
    plot_convergence(all_records)


if __name__ == "__main__":
    main()
