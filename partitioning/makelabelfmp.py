import pandas as pd
import numpy as np
from multiprocessing import Pool
import logging
from functools import partial  # 导入 functools.partial
import time




df_grid5 = pd.read_csv('grid_info_5.csv')
df_grid6 = pd.read_csv('grid_info_6.csv')
df_grid7 = pd.read_csv('grid_info_7.csv')

df_p365 = pd.read_csv('/Users/yeyuan/Documents/GitHub/GeoEstimation/resources/mp16_places365.csv')
df_p365 = df_p365[['IMG_ID','LAT','LON']]

newdata = df_p365.copy()
data = newdata.head(100)

grid_info5 = df_grid5
grid_info6 = df_grid6
grid_info7 = df_grid7

num_processes = 4

def assign_to_grid(row, grids):
    for index, grid in grids.iterrows():
        lat_range = eval(grid['latitude_range'])
        lon_range = eval(grid['longitude_range'])
        if lat_range[0] <= row['LAT'] <= lat_range[1] and lon_range[0] <= row['LON'] <= lon_range[1]:
            return grid['class_label']
    return None

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

    result_data.to_csv("mp16withlabel.csv", index=False)
    # 记录程序结束时间
    end_time = time.time()

    # 计算运行时间
    running_time = end_time - start_time

    print(f"程序运行时间为: {running_time} 秒")
