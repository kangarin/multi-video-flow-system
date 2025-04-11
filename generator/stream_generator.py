import cv2
import time
import os
import redis
import argparse
import socketio
import base64
import random
import math
import numpy as np
import json
import threading
from task import Task

# 编码
def encode_frame_base64(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

# 解码
def decode_frame_base64(base64_string):
    img_data = base64.b64decode(base64_string)
    buffer = np.frombuffer(img_data, dtype=np.uint8)
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)

class StreamGenerator:
    def __init__(self, 
                 stream_id, 
                 flow_weights=1.0,
                 min_fps=5, 
                 max_fps=20, 
                 min_duration=100, 
                 max_duration=300, 
                 redis_host="localhost", 
                 redis_port=6379,
                 video_dir=None):
        # 基本配置
        self.stream_id = stream_id
        self.video_dir = video_dir
        self.flow_weights = flow_weights
        
        # FPS控制配置
        self.min_fps = min_fps
        self.max_fps = max_fps
        self.min_duration = min_duration
        self.max_duration = max_duration
        
        # 当前周期设置
        self.current_min_fps = min_fps
        self.current_max_fps = max_fps
        self.cycle_start_time = time.time()
        self.cycle_duration = random.uniform(min_duration, max_duration)
        
        # 初始化Redis连接
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=0)
        
        # 初始化处理器ID映射和服务器列表
        self.processor_id_map = {}
        self.server_list = self.get_servers_from_redis()
        print(f"服务器列表: {self.server_list}")
        
        # 初始化SocketIO客户端池
        self.sio_clients = {}
        self.sio_locks = {}
        self._initialize_socketio_clients()
        
        # 心跳线程
        self.heartbeat_thread = None
        self.running = True
        
    def register_with_redis(self):
        """将流注册到Redis中"""
        stream_key = f"stream:{self.stream_id}"
        self.redis_client.hset(stream_key, "last_heartbeat", time.time())
        print(f"流 {self.stream_id} 已在Redis中注册")
        
        # 启动心跳线程
        self.heartbeat_thread = threading.Thread(
            target=self._send_heartbeats, 
            daemon=True
        )
        self.heartbeat_thread.start()
        return True
    
    def _send_heartbeats(self):
        """向Redis发送定期心跳"""
        stream_key = f"stream:{self.stream_id}"
        while self.running:
            current_fps = self.calculate_current_fps()
            self.redis_client.hset(stream_key, "last_heartbeat", time.time())
            time.sleep(15)  # 每15秒发送一次心跳
    
    def get_servers_from_redis(self):
        """从Redis获取处理器服务器列表"""
        processor_keys = self.redis_client.keys("processor:*")
        servers = []
        
        for key in processor_keys:
            # 从key中提取处理器ID
            processor_id = int(key.decode('utf-8').split(':')[1])
            
            # 获取处理器信息
            processor_info = self.redis_client.hgetall(key)
            host = processor_info[b'host'].decode('utf-8')
            port = processor_info[b'port'].decode('utf-8')
            
            # 构建服务器URL
            server_url = f"http://{host}:{port}"
            servers.append(server_url)
            
            # 保存处理器ID和服务器URL的映射
            self.processor_id_map[server_url] = processor_id
            print(f"添加处理器映射: {server_url} -> 处理器ID {processor_id}")
                
        return servers
            
    def _initialize_socketio_clients(self):
        """初始化所有服务器的SocketIO客户端"""
        for server_url in self.server_list:
            sio = socketio.Client()
            
            @sio.event
            def connect():
                print(f"已连接到服务器: {server_url}")
                self.sio_clients[server_url]["connected"] = True
            
            @sio.event
            def task_queued(data):
                print(f"任务确认 - 流ID: {data['stream_id']}, 任务ID: {data['task_id']}, 队列位置: {data['queue_position']}")
            
            # 存储客户端及其状态
            self.sio_clients[server_url] = {
                "client": sio,
                "connected": False
            }
            
            # 为该客户端创建一个锁
            self.sio_locks[server_url] = threading.Lock()
            
            # 连接到服务器
            sio.connect(server_url)
    
    def get_distribution_from_redis(self):
        """从Redis获取流量分配决策"""
        key = f"decision:stream:{self.stream_id}"
        distribution_json = self.redis_client.hget(key, "distribution")
        
        if distribution_json:
            distribution = json.loads(distribution_json.decode('utf-8') if isinstance(distribution_json, bytes) else distribution_json)
            print(f"流 {self.stream_id} 获取到分配决策: {distribution}")
            return distribution
        return None
    
    def get_server_by_distribution(self):
        """基于分配决策选择服务器"""
        # 从Redis获取分配决策
        distribution = self.get_distribution_from_redis()
        
        if not distribution:
            # 随机选择一个服务器
            return random.choice(self.server_list)
        
        # 构建处理器ID -> 服务器URL的映射
        processor_to_server = {}
        for server_url in self.server_list:
            processor_id = self.processor_id_map.get(server_url)
            if processor_id is not None and 1 <= processor_id <= len(distribution):
                processor_to_server[processor_id] = server_url
        
        # 准备处理器ID列表和对应的概率
        processor_ids = list(processor_to_server.keys())
        probabilities = [distribution[pid-1] for pid in processor_ids]  # 处理器ID从1开始
        
        # 规范化概率
        total_prob = sum(probabilities)
        if total_prob <= 0:
            return random.choice(self.server_list)
        
        normalized_probs = [p/total_prob for p in probabilities]
        
        # 使用numpy.random.choice基于概率选择处理器ID
        selected_processor_id = np.random.choice(processor_ids, p=normalized_probs)
        selected_server = processor_to_server[selected_processor_id]
        
        print(f"选择处理器ID {selected_processor_id}，服务器 {selected_server}")
        return selected_server
    
    def calculate_current_fps(self):
        """计算当前时间应该使用的帧率"""
        # 计算在周期内的位置
        current_time = time.time()
        elapsed_time = current_time - self.cycle_start_time
        
        # 检查是否需要开始新周期
        if elapsed_time >= self.cycle_duration:
            # 开始新周期
            self.cycle_start_time = current_time
            self.cycle_duration = random.uniform(self.min_duration, self.max_duration)
            elapsed_time = 0
            print(f"新周期开始，持续时间: {self.cycle_duration:.2f}秒")
        
        # 使用正弦波计算当前fps
        phase = (elapsed_time / self.cycle_duration) * (2 * math.pi)
        sine_value = math.sin(phase)
        fps_range = self.max_fps - self.min_fps
        current_fps = (sine_value + 1) / 2 * fps_range + self.min_fps
        
        return current_fps
    
    def update_redis_rate(self, rate):
        """更新Redis中的当前生成率和优先级"""
        self.redis_client.hset(f"status:stream:{self.stream_id}", "generation_rate", rate)
        self.redis_client.hset(f"status:stream:{self.stream_id}", "flow_weights", self.flow_weights)
    
    def get_video_files(self):
        """获取视频目录中的所有视频文件"""
        if not self.video_dir:
            raise ValueError("未指定视频目录")
            
        video_extensions = ('.mp4', '.avi', '.mkv', '.mov', '.wmv')
        video_files = []
        
        for file in os.listdir(self.video_dir):
            if file.lower().endswith(video_extensions):
                video_files.append(os.path.join(self.video_dir, file))
                
        if not video_files:
            raise Exception(f"在目录中未找到视频文件: {self.video_dir}")
            
        return sorted(video_files)
    
    def send_task(self, task):
        """发送任务到基于分配决策选择的服务器"""
        # 基于分配决策选择服务器
        server_url = self.get_server_by_distribution()
        
        # 获取所选服务器的客户端和锁
        sio = self.sio_clients[server_url]["client"]
        lock = self.sio_locks[server_url]
        
        # 使用锁确保线程安全
        with lock:
            # 发送任务进行处理
            sio.emit('task', task.to_dict())
            processor_id = self.processor_id_map.get(server_url, "未知")
            print(f"任务 {task.task_id} 已发送到服务器 {server_url} (处理器ID: {processor_id})")
    
    def start_generating(self):
        """开始生成视频帧并发送任务"""
        video_files = self.get_video_files()
        frame_count = 0
        
        while True:  # 循环处理所有视频
            for video_file in video_files:
                print(f"处理视频: {video_file}")
                cap = cv2.VideoCapture(video_file)
                
                video_fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                start_time = time.time()
                
                try:
                    while True:
                        # 计算当前帧率
                        current_time = time.time()
                        target_fps = self.calculate_current_fps()
                        
                        # 更新Redis中的生成率
                        self.update_redis_rate(target_fps)
                        
                        # 计算要读取的帧位置
                        elapsed_time = current_time - start_time
                        target_frame = int(elapsed_time * video_fps)
                        
                        # 根据目标帧率计算跳帧
                        skip_frames = max(1, int(video_fps / target_fps))
                        next_frame = (target_frame // skip_frames) * skip_frames
                        
                        # 检查是否到达视频末尾
                        if next_frame >= total_frames:
                            print(f"视频处理完成: {video_file}")
                            break
                            
                        # 设置位置并读取帧
                        cap.set(cv2.CAP_PROP_POS_FRAMES, next_frame)
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # 将帧统一大小
                        frame = cv2.resize(frame, (640, 480))

                        # 生成任务
                        task = Task(stream_id=self.stream_id,
                                    task_id=frame_count,
                                    frame=encode_frame_base64(frame),
                                    generated_time=current_time,
                                    start_process_time=None,
                                    end_process_time=None)
                        
                        # 发送任务
                        self.send_task(task)

                        # 计数生成的帧
                        frame_count += 1

                        # 控制帧率
                        processing_time = time.time() - current_time
                        sleep_time = max(0, (1.0 / target_fps) - processing_time)
                        time.sleep(sleep_time)
                        
                finally:
                    cap.release()
                    
            print("所有视频处理完成，重新开始...")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='启动视频流生成器')
    
    # 基本参数
    parser.add_argument('--id', type=int, required=True, help='流ID')
    parser.add_argument('--flow_weights', type=float, default=1.0, help='流优先级')
    parser.add_argument('--video-dir', type=str, required=True, help='视频文件目录')
    
    # 帧率控制参数
    parser.add_argument('--min-fps', type=float, default=1, help='最小帧率')
    parser.add_argument('--max-fps', type=float, default=5, help='最大帧率')
    parser.add_argument('--min-duration', type=float, default=100, help='最小周期持续时间（秒）')
    parser.add_argument('--max-duration', type=float, default=300, help='最大周期持续时间（秒）')
    
    # 连接参数
    parser.add_argument('--redis-host', type=str, default='localhost', help='Redis主机地址')
    parser.add_argument('--redis-port', type=int, default=6379, help='Redis端口')
    
    args = parser.parse_args()
    
    # 创建并启动生成器
    generator = StreamGenerator(
        stream_id=args.id,
        flow_weights=args.flow_weights,
        min_fps=args.min_fps,
        max_fps=args.max_fps,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        video_dir=args.video_dir
    )

    generator.register_with_redis()
    
    try:
        print(f"开始生成视频流，流ID：{args.id}")
        generator.start_generating()
    except KeyboardInterrupt:
        print("用户中断生成")
    except Exception as e:
        print(f"生成过程出错: {e}")
    finally:
        generator.running = False
        # 断开所有连接
        for server_url, client_info in generator.sio_clients.items():
            if client_info.get("connected"):
                client_info["client"].disconnect()
        print("生成器已停止")

if __name__ == "__main__":
    main()

# python generator/stream_generator.py --id 1 --video-dir /path/to/videos