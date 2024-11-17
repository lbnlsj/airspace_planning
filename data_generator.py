import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os


def generate_test_data(num_flights=100, num_sectors=5, start_date="2024-02-05"):
    """
    生成测试用的航班数据和空域容量数据，保存在data/目录下

    参数:
    num_flights: 要生成的航班数量
    num_sectors: 空域扇区数量
    start_date: 起始日期
    """
    # 确保data目录存在
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"创建目录: {data_dir}/")

    # 设置随机种子保证可重复性
    np.random.seed(42)
    random.seed(42)

    # 生成扇区标识
    sectors = [f'SECTOR_{i:02d}' for i in range(num_sectors)]

    # 生成空域容量数据
    sector_capacity_data = {
        '扇区标识': sectors,
        '对外声明的15分钟静态通行能力': np.random.randint(3, 8, num_sectors),
        '对外声明的1小时静态通行能力': np.random.randint(10, 20, num_sectors)
    }
    sector_capacity_df = pd.DataFrame(sector_capacity_data)

    # 转换起始日期为timestamp
    base_date = datetime.strptime(start_date, "%Y-%m-%d")
    base_timestamp = int(base_date.timestamp())

    # 生成航班数据
    drive_in_data = []
    drive_out_data = []

    for flight in range(num_flights):
        flight_data_in = []
        flight_data_out = []

        # 为每个扇区生成时间
        last_exit_time = None

        for sector in sectors:
            # 随机决定是否经过这个扇区
            if random.random() < 0.7:  # 70%的概率经过扇区
                if last_exit_time is None:
                    # 第一个扇区的进入时间
                    entry_time = base_timestamp + random.randint(0, 24 * 60 * 60)
                else:
                    # 后续扇区的进入时间在前一个扇区退出后0-10分钟
                    entry_time = last_exit_time + random.randint(0, 600)

                # 在扇区内停留10-30分钟
                duration = random.randint(600, 1800)
                exit_time = entry_time + duration

                # 格式化时间
                entry_time_str = datetime.fromtimestamp(entry_time).strftime("%Y-%m-%d %H:%M:%S")
                exit_time_str = datetime.fromtimestamp(exit_time).strftime("%Y-%m-%d %H:%M:%S")

                flight_data_in.append(entry_time_str)
                flight_data_out.append(exit_time_str)

                last_exit_time = exit_time
            else:
                flight_data_in.append(np.nan)
                flight_data_out.append(np.nan)

        drive_in_data.append(flight_data_in)
        drive_out_data.append(flight_data_out)

    # 创建DataFrame
    drive_in_df = pd.DataFrame(drive_in_data, columns=sectors)
    drive_out_df = pd.DataFrame(drive_out_data, columns=sectors)

    # 保存数据到data目录下的CSV文件
    save_paths = {
        '空域容量数据': os.path.join(data_dir, '空域容量数据.csv'),
        '驶入时间': os.path.join(data_dir, '驶入时间.csv'),
        '驶出时间': os.path.join(data_dir, '驶出时间.csv')
    }

    sector_capacity_df.to_csv(save_paths['空域容量数据'], index=False)
    drive_in_df.to_csv(save_paths['驶入时间'], index=True)
    drive_out_df.to_csv(save_paths['驶出时间'], index=True)

    print(f"数据已保存到以下文件：")
    for name, path in save_paths.items():
        print(f"- {name}: {path}")

    return sector_capacity_df, drive_in_df, drive_out_df


if __name__ == "__main__":
    # 生成测试数据
    print("开始生成测试数据...")
    sector_capacity, drive_in, drive_out = generate_test_data(
        num_flights=100,
        num_sectors=5,
        start_date="2024-02-05"
    )

    print("\n数据生成完成！")
    print(f"\n空域容量数据预览:\n{sector_capacity.head()}")
    print(f"\n驶入时间数据预览:\n{drive_in.head()}")
    print(f"\n驶出时间数据预览:\n{drive_out.head()}")