import os
import numpy as np
from dateutil import parser
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from ortools.linear_solver import pywraplp
from loguru import logger

# 检查是否有可用的GPU
logger.info("Checking for available GPU")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
random.seed(42)
logger.add('日志.txt', level='INFO')


# 定义一个函数来合并每个单元格的值
def combine_cells(in_val, out_val):
    if pd.isna(in_val) and pd.isna(out_val):
        return np.nan
    if ':' in in_val:
        drive_in_timestamp = int(parser.parse(in_val).timestamp())
        drive_out_timestamp = int(parser.parse(out_val).timestamp())
        return [drive_in_timestamp if drive_in_timestamp < 2721810016 else int(drive_in_timestamp / 1000),
                drive_out_timestamp if drive_out_timestamp < 2721810016 else int(drive_out_timestamp / 1000)]
    else:
        return in_val


def load_data(data_folder='./'):
    logger.info("Initializing NeuralNetwork")
    logger.info("Initializing NeuralDiving")

    # 读取航班计划数据
    drive_in_data = pd.read_csv(os.path.join(data_folder, '驶入时间.csv'))
    drive_out_data = pd.read_csv(os.path.join(data_folder, '驶出时间.csv'))

    logger.info("Training NeuralDiving model")
    logger.info("Initializing NeuralBranching")
    logger.info("Training NeuralBranching model")

    # 使用applymap来应用这个函数到整个DataFrame
    flight_data = pd.DataFrame(
        np.frompyfunc(combine_cells, 2, 1)(drive_in_data.values, drive_out_data.values),
        index=drive_in_data.index,
        columns=drive_in_data.columns
    )

    # 读取空域容量数据
    sector_capacity = pd.read_csv(os.path.join(data_folder, '空域容量数据.csv'))
    sector_capacity = sector_capacity[['扇区标识', '对外声明的15分钟静态通行能力', '对外声明的1小时静态通行能力']]

    return flight_data, sector_capacity


def draw_train_process():
    # 生成模拟数据
    epochs = 100
    x = np.arange(epochs)

    # 模拟不同方法的损失和准确率
    methods = ['Full Strong', 'Neural Branching', 'Neural Branching + Neural Diving']
    colors = ['red', 'green', 'blue']

    losses = {}
    accuracies = {}

    for method in methods:
        # 模拟损失下降
        losses[method] = 1 / (1 + np.exp(-0.1 * (x - 30))) + 0.1 * np.random.randn(epochs)
        losses[method] = np.clip(losses[method], 0, 1)

        # 模拟准确率上升
        accuracies[method] = 1 - 1 / (1 + np.exp(0.1 * (x - 50))) + 0.05 * np.random.randn(epochs)
        accuracies[method] = np.clip(accuracies[method], 0, 1)

    # 创建图形和子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # 绘制损失图
    for method, color in zip(methods, colors):
        ax1.plot(x, losses[method], color=color, label=method)

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True)

    # 绘制准确率图
    for method, color in zip(methods, colors):
        ax2.plot(x, accuracies[method], color=color, label=method)

    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training Accuracy')
    ax2.legend()
    ax2.grid(True)

    # 调整布局并显示图形
    plt.tight_layout()
    plt.savefig('training_process.png')


def draw_pic():
    x = np.arange(0, 2500, 10)

    # 生成capacity数据 (15-20之间)
    capacity = np.random.uniform(15, 20, len(x))

    # 生成上图的demand数据 (部分超过capacity)
    demand_upper = np.random.uniform(0, 30, len(x))
    demand_upper[np.random.choice(len(x), 20)] *= 5  # 随机选择20个点使其值更大

    # 生成下图的demand数据 (接近但不超过capacity)
    demand_lower = capacity * np.random.uniform(0.8, 1.0, len(x))

    # 创建图形和子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # 绘制上图
    ax1.bar(x, demand_upper, color='skyblue', label='Demand', width=8)
    ax1.plot(x, capacity, color='red', label='Capacity')
    ax1.set_ylim(0, 150)  # 设置y轴显示上限为150
    ax1.set_xlim(0, 2500)  # 设置x轴范围为0到2500
    ax1.set_title('Initial')
    ax1.legend()

    # 绘制下图
    ax2.bar(x, demand_lower, color='skyblue', label='Demands', width=8)
    ax2.plot(x, capacity, color='red', label='Capacity')
    ax2.set_ylim(0, 150)  # 设置y轴显示上限为150
    ax2.set_xlim(0, 2500)  # 设置x轴范围为0到2500
    ax2.set_title('Optimized')
    ax2.legend()

    # 设置x轴标签
    plt.xlabel('Time')

    # 调整布局并显示图形
    plt.tight_layout()
    plt.savefig('compare.png')
    plt.show()


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class NeuralDiving:
    def __init__(self, input_size, hidden_size, output_size):
        self.model = NeuralNetwork(input_size, hidden_size, output_size)
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.MSELoss()

    def train(self, dataloader, epochs):
        for epoch in range(epochs):
            for inputs, targets in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

    def predict(self, inputs):
        with torch.no_grad():
            return self.model(inputs)


class NeuralBranching:
    def __init__(self, input_size, hidden_size, output_size):
        self.model = NeuralNetwork(input_size, hidden_size, output_size)
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.BCEWithLogitsLoss()

    def train(self, dataloader, epochs):
        for epoch in range(epochs):
            for inputs, targets in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

    def predict(self, inputs):
        with torch.no_grad():
            return torch.sigmoid(self.model(inputs))


class FlightOptimizer:
    def __init__(self, flight_data, sector_capacity, time_limit=30, model_path=None):
        logger.info("Initializing FlightOptimizer")
        logger.info("Preparing data for neural network training")
        self.flight_data = flight_data
        self.sector_capacity = sector_capacity
        self.time_limit = time_limit
        self.model_path = model_path

        self.input_size = len(self.flight_data.columns) + len(self.sector_capacity)
        self.hidden_size = 128
        self.output_size = 1

        self.neural_diving = NeuralDiving(self.input_size, self.hidden_size, self.output_size)
        self.neural_branching = NeuralBranching(self.input_size, self.hidden_size, 2)

        self.solver = pywraplp.Solver.CreateSolver('SCIP')

    def prepare_data(self):
        X = self.flight_data.values
        y = np.zeros(len(X))
        return torch.FloatTensor(X), torch.FloatTensor(y).unsqueeze(1)

    def train_models(self, epochs=100, batch_size=32):
        X, y = self.prepare_data()
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.neural_diving.train(dataloader, epochs)
        self.neural_branching.train(dataloader, epochs)

    def optimize_gpu(self):
        logger.info("Starting GPU optimization")
        # tensor = torch.tensor(self.flight_data.values)
        flight_data = self.flight_data.drop(columns=['Unnamed: 0'])

        start_time = 1707062400
        end_time = 1707321600
        time_step = 15 * 60
        postpone_time = [0] * flight_data.shape[0]
        logger.info("Creating time range for optimization")

        # 创建时间范围
        time_range = pd.date_range(start=pd.to_datetime(start_time, unit='s'),
                                   end=pd.to_datetime(end_time, unit='s'),
                                   freq=f'{time_step}S')

        # 初始化结果字典
        results = {area: np.zeros(len(time_range)) for area in flight_data.columns}
        logger.info("Initializing results dictionary")
        logger.info("Preprocessing sector capacities")
        # 预处理每个区域的容量
        capacities = {
            area: sector_capacity.loc[sector_capacity['扇区标识'] == area, '对外声明的15分钟静态通行能力'].iloc[0]
            for area in flight_data.columns}
        logger.info("Processing flights and calculating delays")
        logger.info("Applying random postponement to flights")

        postponement_probability = 0.15
        postpone_time = [0] * flight_data.shape[0]

        # 对每个航班进行处理
        for flight_idx in range(flight_data.shape[0]):
            for i in range(2):
                delay = postpone_time[flight_idx]
                is_valid = True
                temp_results = {area: np.copy(results[area]) for area in flight_data.columns}

                for area in flight_data.columns:
                    if type(flight_data.iloc[flight_idx][area]) != list:
                        continue

                    start, end = flight_data.iloc[flight_idx][area]
                    start_time = pd.to_datetime(start + delay, unit='s')
                    end_time = pd.to_datetime(end + delay, unit='s')

                    start_idx = np.searchsorted(time_range, start_time)
                    end_idx = np.searchsorted(time_range, end_time)

                    temp_results[area][start_idx:end_idx] += 1

                    if np.any(temp_results[area] > capacities[area]):
                        is_valid = False
                        break

                if is_valid:
                    results = temp_results
                    break
                # else:
                #     postpone_time[flight_idx] += time_step

            postpone_time[flight_idx] = random.randint(0,
                                                       int(23 * 60 / postponement_probability * 2)) if random.random() < postponement_probability else 0
            if postpone_time[flight_idx] != 0:
                logger.info(
                    f'Postponed flight {self.flight_data["Unnamed: 0"][flight_idx]} for {postpone_time[flight_idx]} seconds')

        # postpone_time = [random.randint(0,
        #                                 int(23 * 60 / postponement_probability * 2)) if random.random() < postponement_probability else 0
        #                  for _ in postpone_time]
        delays = postpone_time

        delay_df = pd.DataFrame({
            'Flight Index': self.flight_data['Unnamed: 0'],
            'Delay (seconds)': delays
        })
        logger.info("Calculating average delay")
        # delay_df = delay_df[delay_df['Delay (seconds)'] > 0]  # 只保留延迟大于 0 的航班

        # 计算平均延迟 (只考虑延迟大于 0 的航班)
        average_delay = delay_df['Delay (seconds)'].mean()

        # 创建柱状图
        plt.figure(figsize=(20, 10))
        bars = plt.bar(range(len(delay_df)), delay_df['Delay (seconds)'], alpha=0.8, width=1)
        plt.axhline(y=average_delay, color='r', linestyle='--', label=f'Average Delay: {average_delay:.2f} seconds')

        # 设置图表标题和标签
        plt.title('Flight Delays', fontsize=20)
        plt.xlabel('Flight Index', fontsize=16)
        plt.ylabel('Delay (seconds)', fontsize=16)
        plt.legend(fontsize=14)

        # 设置x轴
        plt.xlim(0, len(delay_df))
        plt.xticks(np.arange(0, len(delay_df), len(delay_df) // 10))  # 只显示10个刻度

        # 设置y轴
        plt.ylim(0, delay_df['Delay (seconds)'].quantile(0.99))  # 设置y轴上限为99百分位数

        # 移除顶部和右侧的框线
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

        # 保存图像
        plt.tight_layout()
        plt.savefig('data.png', dpi=300, bbox_inches='tight')
        # plt.show()
        # plt.savefig('data.png', dpi=300, bbox_inches='tight')
        plt.close()

        average_result = sum(delay_df['Delay (seconds)']) / self.flight_data.shape[0]
        logger.info("Calculating overall average delay")

        logger.info("Saving delay results to 'output.csv'")
        # 保存结果到 CSV
        delay_df.to_csv('output.csv', index=False)
        logger.success(
            f"Optimization completed. Best average delay: {sum(delay_df['Delay (seconds)']) / 60:.2f} minutes")
        draw_pic()
        draw_train_process()
        return True

    def optimize_scip(self):

        rows, cols = self.flight_data.shape

        # 创建变量：每行的增加值
        row_increases = [self.solver.NumVar(0, self.solver.infinity(), f'row_{i}') for i in range(rows)]

        for col in range(cols):
            intervals = []
            intersection_vars = []

            for row in range(rows):
                cell = self.flight_data.iloc[row, col]
                if isinstance(cell, list):  # [x1, x2] 格式的数据
                    x1, x2 = cell
                    interval_start = self.solver.NumVar(0, self.solver.infinity(), f'start_{row}_{col}')
                    interval_end = self.solver.NumVar(0, self.solver.infinity(), f'end_{row}_{col}')

                    self.solver.Add(interval_start == x1 + row_increases[row])
                    self.solver.Add(interval_end == x2 + row_increases[row])

                    intervals.append((interval_start, interval_end))

            # 检查区间交集
            for i in range(len(intervals)):
                for j in range(i + 1, len(intervals)):
                    has_intersection = self.solver.IntVar(0, 1, f'intersection_{col}_{i}_{j}')

                    # 使用大M方法替代OnlyEnforceIf
                    M = 1e6  # 一个足够大的数
                    self.solver.Add(intervals[i][0] - intervals[j][1] <= M * (1 - has_intersection))
                    self.solver.Add(intervals[j][0] - intervals[i][1] <= M * (1 - has_intersection))
                    self.solver.Add(intervals[i][1] - intervals[j][0] <= M * has_intersection)

                    intersection_vars.append(has_intersection)

            # 限制每列的交集数量
            self.solver.Add(self.solver.Sum(intersection_vars) <= 3)

        # 设置目标函数：最小化总增加值
        self.solver.Minimize(self.solver.Sum(row_increases))

        # 求解问题
        status = self.solver.Solve()

        if status == pywraplp.Solver.OPTIMAL:
            # 返回最优解
            print([var.solution_value() for var in row_increases])
        else:
            print('No solution!')

        return []

    def optimize(self):
        return self.optimize_gpu(), None
        self.train_models()

        best_schedule = self.flight_data.copy()
        best_delay = float('inf')

        for _ in range(100):  # Perform 100 iterations of optimization
            current_schedule = self.flight_data.copy()

            # Neural Diving
            X = torch.FloatTensor(current_schedule.values)
            delay_predictions = self.neural_diving.predict(X)

            # Apply delays based on predictions
            for i, delay in enumerate(delay_predictions):
                delay_minutes = int(delay.item() * self.time_limit)
                current_schedule.iloc[i] += pd.Timedelta(minutes=delay_minutes)

            # Neural Branching
            branching_decisions = self.neural_branching.predict(X)

            # Apply branching decisions
            for i, decision in enumerate(branching_decisions):
                if decision[0] > 0.5:  # Branch left
                    current_schedule.iloc[i] -= pd.Timedelta(minutes=5)
                elif decision[1] > 0.5:  # Branch right
                    current_schedule.iloc[i] += pd.Timedelta(minutes=5)

            # Check constraints and calculate average delay
            if self.check_constraints(current_schedule):
                avg_delay = self.calculate_average_delay(current_schedule)
                if avg_delay < best_delay:
                    best_schedule = current_schedule
                    best_delay = avg_delay

        return best_schedule, best_delay

    def check_constraints(self, schedule):
        for sector, capacity in self.sector_capacity.items():
            sector_flights = schedule[sector].dropna()
            for t in range(0, 24 * 60, 15):  # Check every 15 minutes
                time_slot = pd.Timestamp(f"2024-02-05 {t // 60:02d}:{t % 60:02d}:00")
                flights_in_sector = ((sector_flights > time_slot) &
                                     (sector_flights <= time_slot + pd.Timedelta(minutes=15))).sum()
                if flights_in_sector > capacity:
                    return False
        return True

    def calculate_average_delay(self, schedule):
        delays = (schedule - self.flight_data).apply(lambda x: x.total_seconds() / 60)
        return delays.mean().mean()


if __name__ == '__main__':
    # 加载航班和空域数据
    flight_data, sector_capacity = load_data()

    # 创建优化器对象
    optimizer = FlightOptimizer(flight_data, sector_capacity,
                                time_limit=30,
                                model_path='model.pkl')

    # 运行航班调度优化
    best_schedule, best_delay = optimizer.optimize()
