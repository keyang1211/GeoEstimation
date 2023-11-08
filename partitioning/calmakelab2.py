import pandas as pd
import numpy as np
from multiprocessing import Pool
import logging
from functools import partial  # 导入 functools.partial
import time
df_grid5 = pd.read_csv('grid_info_5.csv')
df_grid6 = pd.read_csv('grid_info_6.csv')
df_grid7 = pd.read_csv('grid_info_7.csv')

df_p365 = pd.read_csv('~/Geoloc/GeoEstimation/resources/yfcc25600_places365.csv')
df_p365 = df_p365[['IMG_ID','LAT','LON']]

newdata = df_p365.copy()
data = newdata

grid_info5 = [df_grid5,5]
grid_info6 = [df_grid6,6]
grid_info7 = [df_grid7,7]

num_processes = 10





def find_grid_block(latitude, longitude, num_levels):
    # Full extent of the Earth in degrees
    max_lat, min_lat = 90.0, -90.0
    max_lon, min_lon = 180.0, -180.0

    # Calculate the size of a single grid block at the given level
    lat_step = (max_lat - min_lat) / (2 ** num_levels)
    lon_step = (max_lon - min_lon) / (2 ** num_levels)

    # Calculate the latitude and longitude offset from the minimum values
    lat_offset = latitude - min_lat
    lon_offset = longitude - min_lon

    # Calculate the grid block indices
    lat_index = int(lat_offset / lat_step)
    lon_index = int(lon_offset / lon_step)
    # 返回的是gridID
    return f"Grid_{lat_index}_{lon_index}"

# 返回grid['class_label']
def assign_to_grid(row, grids):
    findgridID = find_grid_block(row['LAT'],row['LON'],grids[1])
    # classlab = grids[0][grids[0]['grid_id'] == findgridID]['class_label'][0]
    filtered_grid = grids[0][grids[0]['grid_id'] == findgridID]
    if not filtered_grid.empty:
        classlab = filtered_grid['class_label'].values[0]
    else:
        print(f"No matching grid block found at level {grids[1]} for the following row:")
        print(row)
        classlab = None  # 或者您可以选择适当的默认值

    return classlab
    # for index, grid in grids.iterrows():
    #     lat_range = eval(grid['latitude_range'])
    #     lon_range = eval(grid['longitude_range'])
    #     if lat_range[0] <= row['LAT'] <= lat_range[1] and lon_range[0] <= row['LON'] <= lon_range[1]:
    #         return grid['class_label']
    # return None

def process_data(data_chunk, grids):  # 修改参数名为 grids
    data_chunk['class'] = data_chunk.apply(partial(assign_to_grid, grids=grids), axis=1)
    return data_chunk

def process_split_data(data, grid_info, num_processes):
    split_data = np.array_split(data, num_processes)
    print("Script is running...")
    with Pool(num_processes) as pool:
        processed_data = pool.map(partial(process_data, grids=grid_info), split_data)  # 修改这里
    return pd.concat(processed_data)

if __name__ == '__main__':
   # 记录程序开始时间
    start_time = time.time()
    processed_data5 = process_split_data(data.copy(), grid_info5, num_processes)
    processed_data6 = process_split_data(data.copy(), grid_info6, num_processes)
    processed_data7 = process_split_data(data.copy(), grid_info7, num_processes)

    result_data = data.copy()
    result_data['class5'] = processed_data5['class']
    result_data['class6'] = processed_data6['class']
    result_data['class7'] = processed_data7['class']

    logging.basicConfig(filename='my_log.log', level=logging.INFO)
    logging.info(result_data.head(50))

    result_data.to_csv("yfccwithlabel.csv", index=False)
    # 记录程序结束时间
    end_time = time.time()

    # 计算运行时间
    running_time = end_time - start_time

    print(f"程序运行时间为: {running_time} 秒")








# # Example usage
# latitude = 40.0  # Example latitude
# longitude = -100.0  # Example longitude
# num_levels = 5  # Example number of grid levels
# grid_block = find_grid_block(latitude, longitude, num_levels)
# print(f"Coordinates ({latitude}, {longitude}) are in grid block: {grid_block}")
