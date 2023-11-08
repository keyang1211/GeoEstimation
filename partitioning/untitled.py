import pandas as pd
import simplekml

# 从CSV文件读取网格划分数据
def read_grids_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    df['latitude_range'] = df['latitude_range'].apply(lambda x: tuple(map(float, x.strip('()').split(','))))
    df['longitude_range'] = df['longitude_range'].apply(lambda x: tuple(map(float, x.strip('()').split(','))))
    return df

# 创建KML文件
def create_kml(grids, kml_file):
    kml = simplekml.Kml()

    for index, row in grids.iterrows():
        grid_id = row['grid_id']
        latitude_range = row['latitude_range']
        longitude_range = row['longitude_range']

        pol = kml.newpolygon(name=grid_id)
        # 设置边界线样式，颜色可根据需要进行调整
        pol.style.polystyle.outline = 1
        pol.style.linestyle.color = simplekml.Color.black  # 设置为黑色
        pol.style.polystyle.fill = 0  # 设置填充为透明

        pol.outerboundaryis.coords = [
            (longitude_range[0], latitude_range[0], 0),
            (longitude_range[1], latitude_range[0], 0),
            (longitude_range[1], latitude_range[1], 0),
            (longitude_range[0], latitude_range[1], 0),
            (longitude_range[0], latitude_range[0], 0)
        ]

    kml.save(kml_file)

if __name__ == "__main__":
    # 从CSV文件读取网格划分数据
    grids = read_grids_from_csv("grid_info_5.csv")

    # 创建KML文件
    create_kml(grids, "grid_info_5.kml")
