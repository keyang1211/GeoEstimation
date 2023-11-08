import argparse
import csv

from geopy.distance import geodesic

def divide_global_grid(num_levels):
    # Full extent of the Earth in degrees
    max_lat, min_lat = 90.0, -90.0
    max_lon, min_lon = 180.0, -180.0

    # Calculate the size of a single grid block at the first level
    lat_step = (max_lat - min_lat) / (2 ** num_levels)
    lon_step = (max_lon - min_lon) / (2 ** num_levels)

    grids = []

    for i in range(2 ** num_levels):
        for j in range(2 ** num_levels):
            # Calculate the latitude and longitude boundaries for each grid block
            grid_min_lat = min_lat + i * lat_step
            grid_max_lat = min_lat + (i + 1) * lat_step
            grid_min_lon = min_lon + j * lon_step
            grid_max_lon = min_lon + (j + 1) * lon_step

            # Calculate the center of the grid block
            center_lat = (grid_min_lat + grid_max_lat) / 2
            center_lon = (grid_min_lon + grid_max_lon) / 2

            grid = {
                "grid_id": f"Grid_{i}_{j}",
                "latitude_range": (grid_min_lat, grid_max_lat),
                "longitude_range": (grid_min_lon, grid_max_lon),
                "center_latitude": center_lat,
                "center_longitude": center_lon,
            }
            grids.append(grid)

    return grids

def save_grids_to_csv(grids, output_dir):
    output_file = f"{output_dir}/grid_info.csv"

    with open(output_file, mode='w', newline='') as file:
        fieldnames = ['class_label', 'grid_id', 'center_latitude', 'center_longitude', 'latitude_range', 'longitude_range']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for i, grid in enumerate(grids):
            writer.writerow({
                'class_label': i,
                'grid_id': grid['grid_id'],
                'center_latitude': grid['center_latitude'],
                'center_longitude': grid['center_longitude'],
                'latitude_range': f"{grid['latitude_range'][0]} to {grid['latitude_range'][1]}",
                'longitude_range': f"{grid['longitude_range'][0]} to {grid['longitude_range'][1]}"
            })

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate global grid and save info to CSV.")
    parser.add_argument("--output_dir", type=str, help="Output directory for CSV file.")
    parser.add_argument("--num_levels", type=int, help="Number of grid levels")
    args = parser.parse_args()

    if args.output_dir is None or args.num_levels is None:
        print("Both output_dir and num_levels are required arguments.")
    else:
        grids = divide_global_grid(args.num_levels)
        save_grids_to_csv(grids, args.output_dir)
        print(f"Grid information saved to {args.output_dir}/grid_info.csv")
