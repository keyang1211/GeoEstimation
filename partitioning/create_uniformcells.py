import csv
import argparse
from s2sphere import CellId, RegionCoverer, LatLng, LatLngRect, Cell
import os  # 导入 os 模块

def generate_s2_cells(level, output_dir):
    # 创建一个 RegionCoverer 实例
    coverer = RegionCoverer()

    # 设置要划分的级别
    coverer.min_level = level
    coverer.max_level = level

    # 定义一个经纬度范围，这里表示全球
    rect = LatLngRect.full()

    # 使用 RegionCoverer 获取 S2 单元列表
    covering = coverer.get_covering(rect)

    # 准备将数据写入CSV文件
    output_file = os.path.join(output_dir, f"S2_cells_level_{level}.csv")  # 使用 os.path.join 添加输出目录

    with open(output_file, mode='w', newline='') as file:
        csv_writer = csv.writer(file)

        # 写入CSV文件的标题行，按指定的顺序
        csv_writer.writerow(["Class_Label", "Hex_ID", "Latitude_Mean", "Longitude_Mean"])

        # 初始化 class_label 为 0
        class_label = 0

        # 遍历S2单元列表并写入CSV文件
        for cell_id in covering:
            # ... 以下为写入CSV的代码

    print(f"S2 cells at level {level} generated and saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate S2 cells and save to CSV")
    parser.add_argument("--level", type=int, required=True, help="S2 level for cell generation")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for CSV file")
    args = parser.parse_args()
    
    generate_s2_cells(args.level, args.output_dir)
