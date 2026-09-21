"""
Shared CLI handling for workflow scripts that operate over basin/ensemble-filter
combinations (--filter and --basin arguments).
"""
import argparse
import sys


def parse_filter_basin_args(description: str) -> argparse.Namespace:
    """Parse the standard --filter/--basin arguments shared by workflow scripts."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--filter', help='Filter name to process (e.g., basic, cooler, hotter)')
    parser.add_argument('--basin', help='Basin name to process (e.g., Colorado, Trinity, Brazos)')
    return parser.parse_args()


def select_filter_sets_and_basins(basins: dict, ensemble_config: list, filter_name: str = None, basin_name: str = None):
    """Narrow the full basin/ensemble-filter configuration down to a single --filter
    and/or --basin selection, if given. Exits with an error message if a named
    filter or basin isn't found in the configuration.
    """
    if filter_name is None:
        filter_name = "All Models"
    if filter_name:
        filter_sets = [fs for fs in ensemble_config if fs["name"] == filter_name]
        if not filter_sets:
            print(f"Error: Filter '{filter_name}' not found in configuration")
            sys.exit(1)
    else:
        filter_sets = ensemble_config

    if basin_name:
        if basin_name not in basins:
            print(f"Error: Basin '{basin_name}' not found in configuration")
            sys.exit(1)
        selected_basins = {basin_name: basins[basin_name]}
    else:
        selected_basins = basins

    return filter_sets, selected_basins
