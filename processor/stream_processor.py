import torch
import cv2
import numpy as np
from collections import deque
import threading
import time
from pathlib import Path
import sys
import redis
import warnings
import base64
from flask import Flask
from flask_socketio import SocketIO, emit
import argparse
import json
from task import Task

warnings.filterwarnings("ignore", category=FutureWarning)

# 将YOLOv5添加到Python路径
sys.path.append(str(Path(__file__).parent / 'yolov5'))

from models.common import AutoShape
from models.experimental import attempt_load

# 编码和解码函数
def encode_frame_base64(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

def decode_frame_base64(base64_string):
    img_data = base64.b64decode(base64_string)
    buffer = np.frombuffer(img_data, dtype=np.uint8)
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)

import socket

def get_local_ip():
    """获取本机IP地址"""
    try:
        # 创建一个UDP套接字
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # 连接到一个外部地址，这样socket会获取本机的对外IP地址
        # 实际上不会发送数据
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception as e:
        print(f"获取本机IP地址失败: {e}")
        return "127.0.0.1"  # 失败时返回本地回环地址

# 初始化Flask和SocketIO
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

class YOLODetector:
    def __init__(self, 
                processor_id, 
                redis_host="localhost", 
                redis_port=6379,
                models_config=None,
                max_queue_length=50):

        self.processor_id = processor_id
        self.redis_client = redis.Redis(host=redis_host, port=redis_port)
        
        # 初始化模型配置
        self.models_config = models_config or {
            'weights_dir': str(Path(__file__).parent /'models'),
            'allowed_sizes': ['n', 's', 'm', 'l', 'x'],
            'default': 's'
        }
        
        self.models_dir = Path(self.models_config['weights_dir'])
        
        # 每个模型的mAP值
        self.model_maps = {
            'n': 25.7,  # YOLOv5n
            's': 37.4,  # YOLOv5s
            'm': 45.2,  # YOLOv5m
            'l': 49.0,  # YOLOv5l
            'x': 50.7   # YOLOv5x
        }
        
        # 初始化速度-质量对列表
        # 格式: [[speed_n, map_n], [speed_s, map_s], ...]
        # 速度是延迟的倒数 (1/延迟)
        self.speed_quality_pairs = []
        for model_size in self.models_config['allowed_sizes']:
            # 默认初始值，将在初始测试中更新
            map_value = self.model_maps.get(model_size) / 100.0
            self.speed_quality_pairs.append([0.0, map_value])

        # 初始化模型和处理
        self.models = {}  # 存储所有加载的模型
        self.current_model_name = None
        self.current_map = None
        self.model_lock = threading.Lock()
        
        # 初始化帧队列，设置最大长度
        self.frame_queue = deque(maxlen=max_queue_length)
        self.queue_lock = threading.Lock()
        
        # 初始化处理线程
        self.processing_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.running = True
        
        # 加载所有模型并设置默认模型
        self._load_all_models()
        default_model = self.models_config['default']
        self.switch_model(default_model)

        # 执行初始测试以获取基准处理时间
        self._run_initial_speed_test()

        # 启动处理线程
        self.processing_thread.start()

        # 心跳线程
        self.heartbeat_thread = None
        
        print(f"YOLODetector初始化完成，处理器ID: {processor_id}")

    def register_with_redis(self, host, port):
        """将处理器注册到Redis中用于服务发现"""
        try:
            if host is None or host == "0.0.0.0":
                host = get_local_ip()
            # 为此处理器创建唯一键
            processor_key = f"processor:{self.processor_id}"
            
            # 存储处理器信息
            self.redis_client.hset(processor_key, mapping={
                "host": host,
                "port": port,
                "last_heartbeat": time.time()
            })
            
            # 设置过期时间(TTL)，处理器崩溃时自动清理
            # self.redis_client.expire(processor_key, 60)  # 60秒TTL
            
            print(f"处理器 {self.processor_id} 已在Redis中注册")
            
            # 启动后台线程用于发送心跳
            self.heartbeat_thread = threading.Thread(
                target=self._send_heartbeats, 
                daemon=True
            )
            self.heartbeat_thread.start()
            
            return True
        except Exception as e:
            print(f"注册处理器失败: {e}")
            return False

    def _send_heartbeats(self):
        """向Redis发送定期心跳以维持注册状态"""
        processor_key = f"processor:{self.processor_id}"
        while self.running:
            try:
                # 更新心跳时间戳和队列长度
                self.redis_client.hset(processor_key, "last_heartbeat", time.time())
                
                # 重置过期时间
                # self.redis_client.expire(processor_key, 60)  # 60秒TTL
            except Exception as e:
                print(f"心跳错误: {e}")
            
            # 下一次心跳前休眠
            time.sleep(15)  # 每15秒发送一次心跳

    def _load_all_models(self):
        """在初始化时将所有模型加载到内存中"""
        print("正在加载所有YOLOv5模型...")
        for model_size in self.models_config['allowed_sizes']:
            try:
                weight_file = self.models_dir / f'yolov5{model_size}.pt'
                if not weight_file.exists():
                    print(f"警告：未找到模型权重：{weight_file}")
                    continue
                    
                print(f"正在加载YOLOv5{model_size}...")
                model = attempt_load(weight_file)
                model = AutoShape(model)
                if torch.cuda.is_available():
                    model = model.cuda()
                
                self.models[model_size] = model
                print(f"成功加载YOLOv5{model_size}")
                
            except Exception as e:
                print(f"加载YOLOv5{model_size}失败：{e}")
                
        print("所有模型加载完成")

    def _run_initial_speed_test(self):
        """对所有模型运行初始测试以获取基准处理时间"""
        print("正在执行初始速度测试...")
        
        # 创建一个测试帧
        test_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        
        for i, model_size in enumerate(self.models_config['allowed_sizes']):
                
            # 切换到当前测试模型
            self.switch_model(model_size)
            model = self.get_active_model()
            
            # 执行多次测试并计算平均时间
            test_times = []
            test_count = 10
            
            for _ in range(test_count):
                start_time = time.time()
                _ = model(test_frame)
                process_time = time.time() - start_time
                test_times.append(process_time)
            
            # 计算平均处理时间
            avg_time = sum(test_times) / len(test_times)
            speed = 1.0 / avg_time if avg_time > 0 else 0.0
            
            # 更新速度-质量对
            self.speed_quality_pairs[i][0] = speed
            
            print(f"模型 YOLOv5{model_size} 平均处理时间: {avg_time:.3f}秒, 速度: {speed:.2f}, mAP: {self.model_maps[model_size]}")
        
        # 切换回默认模型
        self.switch_model(self.models_config['default'])
        print("初始速度测试完成")
        self._update_speed_quality_in_redis()
        print("速度质量对已更新到Redis")

    def update_speed_quality_pair(self, model_index, processing_time):
        """更新指定模型的速度值（使用滑动平均）"""
        if model_index < 0 or model_index >= len(self.speed_quality_pairs):
            return
        
        # 计算速度（时延的倒数）
        speed = 1.0 / processing_time if processing_time > 0 else 0.0
        
        # 使用滑动平均更新速度
        alpha = 0.1  # 滑动平均系数，较小的值意味着新的测量结果影响较小
        current_speed = self.speed_quality_pairs[model_index][0]
        
        if current_speed == 0.0:  # 如果是第一次更新
            new_speed = speed
        else:
            new_speed = (1 - alpha) * current_speed + alpha * speed
            
        self.speed_quality_pairs[model_index][0] = new_speed
        
        # 更新Redis中的速度质量对信息
        self._update_speed_quality_in_redis()

    def _update_speed_quality_in_redis(self):
        """将当前的速度质量对信息更新到Redis中"""
        try:
            # 将整个列表序列化为JSON字符串
            json_data = json.dumps(self.speed_quality_pairs)
            self.redis_client.hset(f"status:processor:{self.processor_id}", "speed_quality_pairs", json_data)
        except redis.ConnectionError as e:
            print(f"Redis连接错误：{e}")

    def check_and_apply_redis_decisions(self):
        """检查Redis中的速度配置决策并应用它们"""
        key = f"decision:processor:{self.processor_id}"
        speed_config_idx_bytes = self.redis_client.hget(key, "speed_config_idx")
        
        if speed_config_idx_bytes is not None:
            speed_config_idx = int(speed_config_idx_bytes.decode('utf-8') if isinstance(speed_config_idx_bytes, bytes) else speed_config_idx_bytes)
            print(f"处理器 {self.processor_id} 获取到速度配置决策: {speed_config_idx}")
            
            # 确保索引在有效范围内
            if 0 <= speed_config_idx < len(self.models_config['allowed_sizes']):
                model_size = self.models_config['allowed_sizes'][speed_config_idx]
                
                # 只有当模型需要变化时才切换
                if model_size != self.current_model_name:
                    print(f"根据Redis决策从模型 '{self.current_model_name}' 切换到 '{model_size}'")
                    self.switch_model(model_size)
                    return True
                else:
                    print(f"当前模型已经是 '{model_size}'，无需切换")
            else:
                print(f"配置索引 {speed_config_idx} 无效，有效范围是0-{len(self.models_config['allowed_sizes'])-1}")
        
        return False

    def switch_model(self, new_model_name):
        """切换到指定的模型"""
        if new_model_name not in self.models_config['allowed_sizes']:
            raise ValueError(f"无效的模型大小：{new_model_name}")
            
        if new_model_name not in self.models:
            raise ValueError(f"模型YOLOv5{new_model_name}未加载")
            
        with self.model_lock:
            self.current_model_name = new_model_name
            self.current_map = self.model_maps.get(new_model_name, 0)
            print(f"已切换到模型：YOLOv5{new_model_name}")
        
        return True

    def get_active_model(self):
        """获取当前激活的模型"""
        with self.model_lock:
            return self.models.get(self.current_model_name)
            
    def add_task_to_queue(self, task):
        """将任务添加到处理队列
        
        参数：
            task: Task对象
        """
        # 解码帧
        task.received_time = time.time()
        print(f"任务传输时间: {(task.received_time - task.generated_time):.3f}秒")
        self.update_latency_matrix(task.stream_id, self.processor_id, task.received_time - task.generated_time)
        frame = decode_frame_base64(task.frame)
        task.frame = None
        
        with self.queue_lock:
            # 记录任务开始处理的时间
            task.start_process_time = time.time()
            
            # 如果队列达到最大长度，最老的帧将自动被丢弃
            self.frame_queue.append({
                'task': task,
                'frame': frame,
                'timestamp': time.time()
            })
            
            queue_length = len(self.frame_queue)
            self.update_redis_queue(queue_length)
            
            # 通过SocketIO发送队列更新事件
            socketio.emit('queue_update', {
                'processor_id': self.processor_id, 
                'queue_length': queue_length
            })
            
            return queue_length
    
    def update_latency_matrix(self, stream_id, processor_id, latency):
        """更新延迟矩阵"""
        try:
            # 使用Redis存储延迟数据
            # 先获取当前延迟，若有，则通过指数平均更新，否则直接设置
            current_latency = self.redis_client.hget(f"status:latency:{stream_id}:{processor_id}", "latency")
            if current_latency:
                current_latency = float(current_latency)
                new_latency = 0.9 * current_latency + 0.1 * latency
            else:
                new_latency = latency
            self.redis_client.hset(f"status:latency:{stream_id}:{processor_id}", "latency", new_latency)
        except redis.ConnectionError as e:
            print(f"Redis连接错误：{e}")
    def update_redis_queue(self, queue_length):
        """用当前队列长度更新Redis服务器"""
        try:
            self.redis_client.hset(f"status:processor:{self.processor_id}", "queue_length", queue_length)
        except redis.ConnectionError as e:
            print(f"Redis连接错误：{e}")

    def _process_queue(self):
        """处理队列帧的后台线程"""
        decision_check_interval = 1.0
        last_decision_check = time.time()
        
        while self.running:
            # 定期检查Redis中的决策
            current_time = time.time()
            if current_time - last_decision_check > decision_check_interval:
                self.check_and_apply_redis_decisions()
                last_decision_check = current_time
            
            task_data = None
            
            # 使用锁从队列获取帧
            with self.queue_lock:
                if len(self.frame_queue) > 0:
                    task_data = self.frame_queue.popleft()
                    # 更新队列长度信息
                    self.update_redis_queue(len(self.frame_queue))
            
            # 如果获取到帧则处理
            if task_data is not None:
                self._process_single_task(task_data)
            else:
                # 没有帧要处理，短暂休眠
                time.sleep(0.01)
    
    def _process_single_task(self, task_data):
        """处理单个任务"""
        try:
            # 提取任务和帧
            task = task_data['task']
            frame = task_data['frame']
            
            # 获取当前模型并运行推理
            model = self.get_active_model()
            if model is None:
                return
                
            # 执行检测
            results = model(frame)
            
            # 记录处理结束时间
            task.end_process_time = time.time()

            # 更新模型的速度-质量对
            current_model_index = self.models_config['allowed_sizes'].index(self.current_model_name)
            processing_time = task.end_process_time - task.start_process_time
            print(f"任务 {task.task_id} 处理完成，用时: {processing_time:.3f}秒")
            self.update_speed_quality_pair(current_model_index, processing_time)
            
            # === 发送到汇总器 ===
            # 更新任务数据
            processed_frame = encode_frame_base64(results.render()[0])  # 渲染后的图像
            
            # 如果extra_info为None，初始化为空字典
            if task.extra_info is None:
                task.extra_info = {}
            
            # 将检测结果添加到extra_info
            task.extra_info.update({
                'processor_id': self.processor_id,
                'model_used': f"yolov5{self.current_model_name}",
                'detection_results': results.pandas().xyxy[0].to_dict('records')
            })
            
            # 创建结果任务对象
            result_task = Task(
                stream_id=task.stream_id,
                task_id=task.task_id,
                generated_time=task.generated_time,
                frame=processed_frame,  # 使用处理后的帧
                received_time=task.received_time,
                start_process_time=task.start_process_time,
                end_process_time=task.end_process_time,
                extra_info=task.extra_info
            )
            
            # 将结果推送到Redis队列
            self.redis_client.lpush(f"queue:results:{task.stream_id}", json.dumps(result_task.to_dict()))
            print(f"结果已推送到队列 queue:results:{task.stream_id}")
            # === 发送结束 ===
                
        except Exception as e:
            print(f"处理任务时出错：{e}")
    
    def get_queue_length(self):
        """获取当前队列长度"""
        with self.queue_lock:
            return len(self.frame_queue)
    
    def stop(self):
        """停止处理线程"""
        self.running = False
        if self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)

        # 等待心跳线程结束
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            self.heartbeat_thread.join(timeout=2.0)

        # 清理Redis连接
        self.redis_client.close()
        print(f"YOLODetector {self.processor_id} 已停止")

# 全局变量来存储检测器实例
detector = None

# Flask SocketIO事件处理
@socketio.on('connect')
def handle_connect():
    print('客户端已连接')
    emit('status', {'message': '已连接到YOLO检测服务器'})

@socketio.on('disconnect')
def handle_disconnect():
    print('客户端已断开连接')

@socketio.on('task')
def handle_task(data):
    """处理传入的完整任务"""
    try:
        # 创建任务对象
        task = Task.from_dict(data)
        
        # 添加任务到处理队列
        queue_position = detector.add_task_to_queue(task)
        
        # 确认接收
        emit('task_queued', {
            'stream_id': task.stream_id,
            'task_id': task.task_id, 
            'queue_position': queue_position
        })
        
    except Exception as e:
        print(f"处理任务时出错：{e}")
        emit('error', {'message': f'处理任务时出错：{str(e)}'})

@socketio.on('switch_model')
def handle_switch_model(data):
    """处理切换模型的请求"""
    try:
        model_name = data.get('model_name')
        if not model_name:
            emit('error', {'message': '未提供模型名称'})
            return
            
        success = detector.switch_model(model_name)
        if success:
            emit('model_switched', {
                'model': f'yolov5{model_name}',
                'map': detector.current_map
            })
        else:
            emit('error', {'message': f'切换到模型失败：{model_name}'})
            
    except Exception as e:
        print(f"切换模型时出错：{e}")
        emit('error', {'message': f'切换模型时出错：{str(e)}'})

@socketio.on('get_status')
def handle_get_status():
    """返回当前检测器状态"""
    status = {
        'processor_id': detector.processor_id,
        'current_model': f'yolov5{detector.current_model_name}',
        'map': detector.current_map,
        'queue_length': detector.get_queue_length(),
        'available_models': detector.models_config['allowed_sizes']
    }
    emit('status_update', status)


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='启动YOLO检测服务器')
    parser.add_argument('--id', type=int, default=1, help='处理器ID')
    parser.add_argument('--redis-host', type=str, default='localhost', help='Redis主机地址')
    parser.add_argument('--redis-port', type=int, default=6379, help='Redis端口')
    parser.add_argument('--model', type=str, default='s', help='初始模型大小 (n, s, m, l, x)')
    parser.add_argument('--max-queue', type=int, default=50, help='最大队列长度')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='服务器主机地址')
    parser.add_argument('--port', type=int, default=5000, help='服务器端口')
    
    args = parser.parse_args()
    
    global detector
    
    # 创建检测器实例
    detector = YOLODetector(
        processor_id=args.id,
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        max_queue_length=args.max_queue,
        models_config={
            'weights_dir': str(Path(__file__).parent / 'models'),
            'allowed_sizes': ['n', 's', 'm', 'l', 'x'],
            'default': args.model
        }
    )

    # 将处理器注册到Redis用于服务发现
    detector.register_with_redis(host=args.host, port=args.port)

    try:
        # 启动SocketIO服务器
        print(f"正在启动Flask SocketIO服务器于 {args.host}:{args.port}...")
        socketio.run(app, host=args.host, port=args.port, debug=False, allow_unsafe_werkzeug=True)
        
    except KeyboardInterrupt:
        print("服务器被用户停止")
    finally:
        # 清理
        if detector:
            detector.stop()

if __name__ == "__main__":
    main()

    # python processor/stream_processor.py --id 1 --port 7777 --model s