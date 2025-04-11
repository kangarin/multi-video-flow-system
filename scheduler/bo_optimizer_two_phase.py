import optuna
import numpy as np
from typing import List, Tuple
import os
import shutil

import optuna
import numpy as np
from typing import List, Tuple
import os
import shutil

optuna.logging.set_verbosity(optuna.logging.WARNING) 

class TwoPhaseFlowScheduler:
    def __init__(
        self,
        flow_rates: List[float],      
        flow_weights: List[float],     
        trans_delays: List[List[float]], 
        nodes_config: List[List[Tuple[float, float]]], 
        current_queue_lengths: List[float],
        T: float,                      
        delay_weight: float = 0.7,     
        quality_weight: float = 0.3    
    ):
        self.flow_rates = flow_rates
        self.flow_weights = flow_weights
        self.trans_delays = trans_delays
        self.nodes_config = nodes_config
        self.current_queue_lengths = current_queue_lengths
        self.T = T
        self.delay_weight = delay_weight
        self.quality_weight = quality_weight
        
        self.m = len(flow_rates)       
        self.n = len(nodes_config)     
        
        # 验证队列长度列表的维度
        if len(current_queue_lengths) != self.n:
            raise ValueError("队列长度列表维度必须等于节点数量")
        
        # 生成有效的流量分配方案
        self.valid_distributions = self._generate_valid_distributions()
        print(f"共有 {len(self.valid_distributions)} 种有效的流量分配方案")
        
        # 数据库文件名定义
        self.phase1_db = "phase1_optimizer.db"
        self.phase2_db = "phase2_optimizer.db"
        self.prev_phase1_db = "prev_phase1_optimizer.db"
        self.prev_phase2_db = "prev_phase2_optimizer.db"

    def _generate_valid_distributions(self) -> List[List[float]]:
        """生成所有有效的流量分配方案（和为1，步长0.1）"""
        def generate_partitions(n_parts: int, remaining: float, current: List[float]) -> List[List[float]]:
            if n_parts == 1:
                return [current + [round(remaining, 1)]]
            results = []
            for i in range(int(remaining * 10) + 1):
                value = i / 10.0
                if value <= remaining:
                    results.extend(generate_partitions(n_parts - 1, remaining - value, current + [value]))
            return results
        
        return generate_partitions(self.n, 1.0, [])

    def calculate_node_delay(
        self,
        node_idx: int,
        flow_distributions: List[List[float]],
        node_speeds: List[float]
    ) -> float:
        """计算节点的平均任务时延，考虑当前队列长度"""
        node_speed = node_speeds[node_idx]
        process_delay = 1.0 / node_speed
        
        # 计算新增任务的到达率和传输时延
        total_arrival_rate = 0
        weighted_trans_delay = 0
        
        for flow_idx in range(self.m):
            ratio = flow_distributions[flow_idx][node_idx]
            if ratio > 0:
                flow_arrival_rate = self.flow_rates[flow_idx] * ratio
                total_arrival_rate += flow_arrival_rate
                weighted_trans_delay += flow_arrival_rate * self.trans_delays[flow_idx][node_idx]
        
        if total_arrival_rate == 0:
            # 即使没有新任务，也需要考虑现有队列的处理时延
            if self.current_queue_lengths[node_idx] > 0:
                return self.current_queue_lengths[node_idx] / node_speed
            return 0
            
        avg_trans_delay = weighted_trans_delay / total_arrival_rate
        
        # 考虑现有队列和新增任务的综合影响
        current_queue_delay = self.current_queue_lengths[node_idx] / node_speed
        
        if total_arrival_rate < node_speed:
            # 系统稳定，但需要等待当前队列处理完
            return avg_trans_delay + process_delay + current_queue_delay
        
        # 系统不稳定，需要考虑新增任务导致的额外排队
        total_tasks = total_arrival_rate * self.T
        additional_queue_length = (total_arrival_rate - node_speed) * self.T / 2
        additional_queue_delay = additional_queue_length / node_speed
        
        return avg_trans_delay + process_delay + current_queue_delay + additional_queue_delay

    def _calculate_total_cost(
        self, 
        flow_distributions: List[List[float]], 
        node_speeds: List[float],
        node_qualities: List[float]
    ) -> float:
        """计算给定配置的总成本"""
        total_cost = 0
        for flow_id in range(self.m):
            flow_cost = 0
            for node_id in range(self.n):
                ratio = flow_distributions[flow_id][node_id]
                if ratio > 0:
                    delay = self.calculate_node_delay(
                        node_id, flow_distributions, node_speeds
                    )
                    delay_cost = delay * self.delay_weight
                    quality_benefit = (1 - node_qualities[node_id]) * self.quality_weight
                    flow_cost += ratio * (delay_cost + quality_benefit)
            
            total_cost += self.flow_weights[flow_id] * flow_cost
        
        return total_cost

    def optimize_phase1(self, prev_node_params=None, n_trials=100) -> Tuple[List[List[float]], List[int], float]:
        """第一阶段优化：固定节点配置，优化流量分配"""
        def objective_phase1(trial: optuna.Trial) -> float:
            # 使用固定的节点配置
            node_speeds = []
            node_qualities = []
            for node_id in range(self.n):
                if prev_node_params is None:
                    config_idx = 0  # 使用默认的第一个配置
                else:
                    config_idx = prev_node_params[node_id]
                speed, quality = self.nodes_config[node_id][config_idx]
                node_speeds.append(speed)
                node_qualities.append(quality)
            
            # 只优化流量分配
            flow_distributions = []
            for flow_id in range(self.m):
                dist_idx = trial.suggest_categorical(
                    f'flow_{flow_id}',
                    range(len(self.valid_distributions))
                )
                flow_distributions.append(self.valid_distributions[dist_idx])
            
            return self._calculate_total_cost(flow_distributions, node_speeds, node_qualities)

        # 创建study并优化
        study = optuna.create_study(
            study_name="flow_optimization_phase1",
            direction='minimize',
            sampler=optuna.samplers.TPESampler(),
            storage=f'sqlite:///{self.phase1_db}'
        )
        
        study.optimize(objective_phase1, n_trials=n_trials)
        
        # 获取最优参数
        best_params = study.best_params
        best_distributions = []
        for flow_id in range(self.m):
            dist_idx = best_params[f'flow_{flow_id}']
            best_distributions.append(self.valid_distributions[dist_idx])
        
        # 获取使用的节点配置
        used_node_configs = []
        if prev_node_params is None:
            used_node_configs = [0] * self.n
        else:
            used_node_configs = prev_node_params
        
        # 清理临时文件
        study._storage.remove_session()
        if os.path.exists(self.phase1_db):
            os.remove(self.phase1_db)
        
        return best_distributions, used_node_configs, study.best_value

    def optimize_phase2(self, best_distributions: List[List[float]], n_trials=100) -> Tuple[List[int], float]:
        """第二阶段优化：固定流量分配，优化节点配置"""
        def objective_phase2(trial: optuna.Trial) -> float:
            # 只优化节点配置
            node_speeds = []
            node_qualities = []
            for node_id in range(self.n):
                config_idx = trial.suggest_categorical(
                    f'node_{node_id}',
                    range(len(self.nodes_config[node_id]))
                )
                speed, quality = self.nodes_config[node_id][config_idx]
                node_speeds.append(speed)
                node_qualities.append(quality)
            
            return self._calculate_total_cost(best_distributions, node_speeds, node_qualities)

        # 创建study并优化
        study = optuna.create_study(
            study_name="flow_optimization_phase2",
            direction='minimize',
            sampler=optuna.samplers.TPESampler(),
            storage=f'sqlite:///{self.phase2_db}'
        )
        
        study.optimize(objective_phase2, n_trials=n_trials)
        
        # 获取最优参数
        best_params = study.best_params
        node_configs = []
        for node_id in range(self.n):
            config_idx = best_params[f'node_{node_id}']
            node_configs.append(config_idx)
        
        # 先不要清理临时文件，在optimize函数中处理
        study._storage.remove_session()
        return node_configs, study.best_value

    def optimize(self, n_trials_per_phase=100) -> Tuple[dict, float]:
        """执行两阶段优化"""
        # 获取上一次的节点配置（如果存在）
        prev_node_params = None
        if os.path.exists(self.prev_phase2_db):
            try:
                prev_study = optuna.load_study(
                    study_name="flow_optimization_phase2",
                    storage=f'sqlite:///{self.prev_phase2_db}'
                )
                if len(prev_study.trials) > 0:
                    prev_node_params = []
                    best_params = prev_study.best_trial.params
                    for node_id in range(self.n):
                        config_idx = best_params[f'node_{node_id}']
                        prev_node_params.append(config_idx)
                    print(f"使用上一次的节点配置作为起点：{prev_node_params}")
                prev_study._storage.remove_session()
            except Exception as e:
                print(f"加载上一次配置失败: {e}")
                prev_node_params = None
        
        # 第一阶段：优化流量分配
        print("\n执行第一阶段优化：")
        best_distributions, used_node_configs, phase1_value = self.optimize_phase1(
            prev_node_params, n_trials_per_phase)
        print(f"第一阶段完成，最优值: {phase1_value:.4f}")
        
        # 第二阶段：优化节点配置
        print("\n执行第二阶段优化：")
        best_node_configs, phase2_value = self.optimize_phase2(
            best_distributions, n_trials_per_phase)
        print(f"第二阶段完成，最优值: {phase2_value:.4f}")
        
        # 保存当前的节点配置
        if os.path.exists(self.prev_phase2_db):
            os.remove(self.prev_phase2_db)
        if os.path.exists(self.phase2_db):
            shutil.copy2(self.phase2_db, self.prev_phase2_db)
            # 复制完成后再删除临时文件
            os.remove(self.phase2_db)
        
        # 构建返回结果
        result = {
            'node_speeds': [],
            'node_qualities': [],
            'node_indexes': used_node_configs,
            'flow_distribution': best_distributions
        }
        
        # 获取最终的节点配置
        for node_id, config_idx in enumerate(best_node_configs):
            speed, quality = self.nodes_config[node_id][config_idx]
            result['node_speeds'].append(speed)
            result['node_qualities'].append(quality)

        result['node_indexes'] = best_node_configs

        return result, phase2_value

class TwoPhaseFlowSchedulerAgent:
    '''流量调度器代理，包装TwoPhaseFlowScheduler类'''
    def __init__(self, T, delay_weight=0.7, quality_weight=0.3):
        self.T = T
        self.delay_weight = delay_weight
        self.quality_weight = quality_weight

    def get_action(self, flow_weights, nodes_config, flow_rates, trans_delays, current_queue_lengths):
        scheduler = TwoPhaseFlowScheduler(
            flow_rates=flow_rates,
            trans_delays=trans_delays,
            current_queue_lengths=current_queue_lengths,
            flow_weights=flow_weights,
            nodes_config=nodes_config,
            T=self.T,
            delay_weight=self.delay_weight,
            quality_weight=self.quality_weight
        )
        best_params, best_value = scheduler.optimize(n_trials_per_phase=100)
        speed_list = best_params['node_speeds']
        quality_list = best_params['node_qualities']
        config_index_list = best_params['node_indexes']
        stream_distribution_list = best_params['flow_distribution']
        print(f"最优速度配置: {speed_list}")
        print(f"最优质量配置: {quality_list}")
        print(f"最优节点配置索引: {config_index_list}")
        print(f"最优流量分配: {stream_distribution_list}")
        return config_index_list, stream_distribution_list


# 示例使用
if __name__ == "__main__":
    # 示例配置
    flow_rates = [4, 2]  # 2个流
    flow_weights = [1, 1]
    trans_delays = [
        [0.1, 0.2],  # 流1到两个节点的传输时延
        [0.2, 0.1]   # 流2到两个节点的传输时延
    ]
    nodes_config = [
        [(3.0, 1.0), (4.0, 0.8), (5.0, 0.5), (6.0, 0.3)],  # 节点1的速度-质量配置
        [(4.0, 1.0), (5.0, 0.8), (6.0, 0.5)]   # 节点2的速度-质量配置
    ]
    T = 10  # 调度周期

    current_queue_lengths = [2, 0]  # 两个节点的当前队列长度
    
    # 创建代理并获取调度结果
    agent = TwoPhaseFlowSchedulerAgent(T)
    config_index_list, stream_distribution_list = agent.get_action(flow_weights, nodes_config, flow_rates, trans_delays, current_queue_lengths)
    print(config_index_list)
    print(stream_distribution_list)
