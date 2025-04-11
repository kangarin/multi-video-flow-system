import redis
import ast
import numpy as np
import redis
import json

class SchedulingDataGetter:
    def __init__(self, redis_host="localhost", redis_port=6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=0, decode_responses=True)

    def get_scheduling_data(self):
        # 获取stream和processor的数量
        stream_keys = self.redis_client.keys('stream:*')
        processor_keys = self.redis_client.keys('processor:*')
        
        num_streams = len(stream_keys)
        num_processors = len(processor_keys)
        
        print(f"发现 {num_streams} 个流和 {num_processors} 个处理器")
        
        # 初始化参数
        flow_weights = [1.0] * num_streams  # 默认权重为1.0
        nodes_config = [[] for _ in range(num_processors)]
        flow_rates = [0.0] * num_streams
        trans_delays = np.zeros((num_streams, num_processors))
        current_queue_lengths = [0] * num_processors
        
        # 获取flow_weights (从status:stream:* 中的flow_weights字段)
        for i in range(1, num_streams + 1):
            key = f"status:stream:{i}"
            if self.redis_client.exists(key):
                stream_data = self.redis_client.hgetall(key)
                if 'flow_weights' in stream_data:
                    flow_weights[i-1] = float(stream_data['flow_weights'])
        
        # 获取nodes_config (从status:processor:* 中的speed_quality_pairs字段)
        for i in range(1, num_processors + 1):
            key = f"status:processor:{i}"
            if self.redis_client.exists(key):
                processor_data = self.redis_client.hgetall(key)
                if 'speed_quality_pairs' in processor_data:
                    # 将字符串转换为Python列表
                    pairs_str = processor_data['speed_quality_pairs']
                    try:
                        # 使用ast.literal_eval安全地评估字符串
                        pairs = ast.literal_eval(pairs_str)
                        nodes_config[i-1] = pairs
                    except (SyntaxError, ValueError) as e:
                        print(f"解析speed_quality_pairs时出错: {e}")
        
        # 获取flow_rates (从status:stream:* 中的generation_rate字段)
        for i in range(1, num_streams + 1):
            key = f"status:stream:{i}"
            if self.redis_client.exists(key):
                stream_data = self.redis_client.hgetall(key)
                if 'generation_rate' in stream_data:
                    flow_rates[i-1] = float(stream_data['generation_rate'])
        
        # 获取trans_delays (从status:latency:* 中)
        latency_keys = self.redis_client.keys('status:latency:*')
        for key in latency_keys:
            # 解析键名以获取流ID和处理器ID
            parts = key.split(':')
            if len(parts) >= 3:
                stream_id = int(parts[2])
                processor_id = int(parts[3]) if len(parts) >= 4 else 1
                
                latency_data = self.redis_client.hgetall(key)
                if 'latency' in latency_data:
                    # 数组索引从0开始，而ID通常从1开始
                    trans_delays[stream_id-1][processor_id-1] = float(latency_data['latency'])
        
        # 获取current_queue_lengths (从status:processor:* 中的queue_length字段)
        for i in range(1, num_processors + 1):
            key = f"status:processor:{i}"
            if self.redis_client.exists(key):
                processor_data = self.redis_client.hgetall(key)
                if 'queue_length' in processor_data:
                    current_queue_lengths[i-1] = int(processor_data['queue_length'])
        
        # 打印收集到的数据
        print("\n=== 收集到的调度数据 ===")
        print(f"流权重: {flow_weights}")
        print(f"节点配置: {nodes_config}")
        print(f"流速率: {flow_rates}")
        print(f"传输延迟矩阵:\n{trans_delays}")
        print(f"当前队列长度: {current_queue_lengths}")
        
        return flow_weights, nodes_config, flow_rates, trans_delays, current_queue_lengths

import redis
import json

class DecisionWriter:
    """用于将调度决策写入Redis的类"""
    
    def __init__(self, redis_host="localhost", redis_port=6379):
        """初始化Redis连接"""
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=0)
        
    def write_decisions(self, config_index_list, stream_distribution_list):
        """
        将调度决策写入Redis
        
        参数:
            config_index_list: 配置索引列表，表示每个处理器应该使用的配置索引
            stream_distribution_list: 流分配列表，表示每个流的任务分配比例
        
        返回:
            bool: 操作是否成功
        """
        try:
            # 获取系统中处理器和流的数量
            processor_keys = self.redis_client.keys('processor:*')
            stream_keys = self.redis_client.keys('stream:*')
            
            num_processors = len(processor_keys)
            num_streams = len(stream_keys)
            
            print(f"系统中共有 {num_processors} 个处理器和 {num_streams} 个流")
            
            # 1. 存储处理器配置决策
            for processor_id in range(1, num_processors + 1):
                # 处理器ID从1开始
                if processor_id <= len(config_index_list):
                    config_idx = config_index_list[processor_id - 1]
                    
                    # 写入配置索引到Redis
                    key = f"decision:processor:{processor_id}"
                    self.redis_client.hset(key, "speed_config_idx", config_idx)
                    
                    print(f"已存储处理器 {processor_id} 的配置: config_idx={config_idx}")
            
            # 2. 存储流分配决策
            for stream_id in range(1, num_streams + 1):
                # 流ID从1开始
                if stream_id <= len(stream_distribution_list):
                    distribution = stream_distribution_list[stream_id - 1]
                    
                    # 写入分配策略到Redis
                    key = f"decision:stream:{stream_id}"
                    self.redis_client.hset(key, "distribution", json.dumps(distribution))
                    
                    print(f"已存储流 {stream_id} 的分配策略: {distribution}")
            
            return True
            
        except Exception as e:
            print(f"写入决策到Redis时出错: {e}")
            return False
        
if __name__ == "__main__":
    try:
        flow_weights, nodes_config, flow_rates, trans_delays, current_queue_lengths = SchedulingDataGetter().get_scheduling_data()
        from bo_optimizer_two_phase import TwoPhaseFlowSchedulerAgent
        # 创建调度器代理实例
        agent = TwoPhaseFlowSchedulerAgent(T=10.0)
        config_index_list, stream_distribution_list = agent.get_action(
            flow_weights, nodes_config, flow_rates, trans_delays, current_queue_lengths
        )
        print(f"配置索引列表: {config_index_list}")
        print(f"流分配列表: {stream_distribution_list}")
        # 将决策写入Redis
        writer = DecisionWriter()
        success = writer.write_decisions(config_index_list, stream_distribution_list)
    except Exception as e:
        print(f"获取调度数据时出错: {e}")