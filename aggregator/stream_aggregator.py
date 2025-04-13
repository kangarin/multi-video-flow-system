import redis
import json
import threading
import time
import heapq
from concurrent.futures import ThreadPoolExecutor

# 假设 Task 类从其他地方导入
from task import Task

class ResultAggregator:
    def __init__(self, redis_host="localhost", redis_port=6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port)
        self.running = True
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # 滑动窗口配置
        self.window_size = 20  # 窗口大小(帧数)
        self.max_latency = 5.0  # 最大等待时间(秒)
        
        # 保存每个流的窗口状态
        self.stream_windows = {}
        self.window_locks = {}
    
    def discover_streams(self):
        """发现系统中的活跃流"""
        stream_keys = self.redis_client.keys('stream:*')
        streams = [key.decode('utf-8').split(':')[1] for key in stream_keys]
        print(f"发现 {len(streams)} 个流")
        return streams
    
    def initialize_window(self, stream_id):
        """初始化流的滑动窗口"""
        if stream_id not in self.stream_windows:
            self.stream_windows[stream_id] = {
                'expected_task_id': 0,  # 期望的下一个任务ID
                'buffer': {},  # 缓存未到期望ID的任务
                'last_delivered_time': time.time()  # 上次交付时间
            }
            self.window_locks[stream_id] = threading.Lock()
    
    def process_stream(self, stream_id):
        """处理单个流的结果队列，使用滑动窗口"""
        queue_key = f"queue:results:{stream_id}"
        print(f"开始处理流 {stream_id} 的结果队列")
        
        # 初始化窗口
        self.initialize_window(stream_id)
        
        while self.running:
            try:
                # 阻塞式获取结果，最多等待1秒
                result = self.redis_client.brpop(queue_key, timeout=1)
                if result:
                    _, data = result
                    result_dict = json.loads(data)
                    task = Task.from_dict(result_dict)
                    
                    # 处理任务(按序或缓存)
                    self.handle_task(stream_id, task)
                    
                # 检查窗口是否需要滑动
                self.try_advance_window(stream_id)
            
            except Exception as e:
                print(f"处理流 {stream_id} 时出错: {e}")
                time.sleep(1)
    
    def handle_task(self, stream_id, task):
        """处理单个任务，按序交付或缓存"""
        with self.window_locks[stream_id]:
            window = self.stream_windows[stream_id]
            
            # 如果收到了期望的任务ID，直接处理并推进窗口
            if task.task_id == window['expected_task_id']:
                self.deliver_task(stream_id, task)
                window['expected_task_id'] += 1
                window['last_delivered_time'] = time.time()
                
                # 处理缓存中后续的任务
                self.process_buffered_tasks(stream_id)
            
            # 如果是未来的任务ID，缓存它
            elif task.task_id > window['expected_task_id']:
                print(f"流 {stream_id}: 缓存任务 {task.task_id}，当前期望任务ID为 {window['expected_task_id']}")
                window['buffer'][task.task_id] = task
            
            # 如果是过去的任务ID，丢弃它
            else:
                print(f"流 {stream_id}: 丢弃过时任务 {task.task_id}，当前期望任务ID为 {window['expected_task_id']}")
    
    def process_buffered_tasks(self, stream_id):
        """处理缓冲区中已经可以交付的任务"""
        window = self.stream_windows[stream_id]
        buffer = window['buffer']
        
        # 当缓冲区中有下一个期望的任务ID时，继续处理
        while window['expected_task_id'] in buffer:
            task = buffer.pop(window['expected_task_id'])
            self.deliver_task(stream_id, task)
            window['expected_task_id'] += 1
            window['last_delivered_time'] = time.time()
    
    def try_advance_window(self, stream_id):
        """检查是否需要滑动窗口(丢弃过期任务)"""
        with self.window_locks[stream_id]:
            window = self.stream_windows[stream_id]
            current_time = time.time()
            
            # 如果超过最大延迟时间没有收到期望的任务，滑动窗口
            if (current_time - window['last_delivered_time'] > self.max_latency and 
                window['buffer']):
                
                # 找出缓冲区中最小的任务ID
                next_available_id = min(window['buffer'].keys())
                
                if next_available_id > window['expected_task_id']:
                    skipped_count = next_available_id - window['expected_task_id']
                    print(f"流 {stream_id}: 滑动窗口，跳过 {skipped_count} 个缺失任务，从 {window['expected_task_id']} 到 {next_available_id}")
                    window['expected_task_id'] = next_available_id
                    # 处理缓存中已经可以交付的任务
                    self.process_buffered_tasks(stream_id)
    
    def deliver_task(self, stream_id, task):
        """交付任务 - 这里只是打印信息，可以扩展为其他处理"""
        # 提取处理信息
        processor_id = task.extra_info.get('processor_id', 'unknown')
        model_used = task.extra_info.get('model_used', 'unknown')
        processing_time = task.extra_info.get('processing_time', 0)
        detections = task.extra_info.get('detection_results', [])
        
        # 输出任务处理摘要
        print(f"\n--------- 流 {stream_id} - 任务 {task.task_id} 已交付 ---------")
        print(f"处理器: {processor_id}, 模型: {model_used}")
        print(f"处理时间: {processing_time:.3f}秒")
        print(f"检测到 {len(detections)} 个对象")
        
        # 计算延迟
        if task.end_process_time and task.generated_time:
            total_latency = task.end_process_time - task.generated_time
            print(f"总延迟: {total_latency:.3f}秒")
    
    def start(self):
        """启动汇总器"""
        print("启动结果汇总器...")
        
        # 发现流
        streams = self.discover_streams()
        
        if not streams:
            print("没有发现活跃的流，等待中...")
            
            # 定期检查直到发现活跃的流
            while not streams and self.running:
                time.sleep(5)
                streams = self.discover_streams()
        
        # 为每个流启动处理线程
        stream_threads = {}
        for stream_id in streams:
            future = self.executor.submit(self.process_stream, stream_id)
            stream_threads[stream_id] = future
        
        print(f"已为 {len(streams)} 个流启动处理线程")
        
        try:
            # 主循环，可以定期检查新的流
            while self.running:
                time.sleep(10)  # 每10秒检查一次
                
                new_streams = self.discover_streams()
                
                # 找出新的流
                for stream_id in new_streams:
                    if stream_id not in stream_threads or stream_threads[stream_id].done():
                        print(f"为新流 {stream_id} 启动处理线程")
                        future = self.executor.submit(self.process_stream, stream_id)
                        stream_threads[stream_id] = future
                
        except KeyboardInterrupt:
            print("汇总器收到停止信号")
            self.stop()
    
    def stop(self):
        """停止汇总器"""
        print("停止汇总器...")
        self.running = False
        self.executor.shutdown(wait=False)
        print("汇总器已停止")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='结果汇总器')
    parser.add_argument('--redis-host', type=str, default='localhost', help='Redis主机地址')
    parser.add_argument('--redis-port', type=int, default=6379, help='Redis端口')
    parser.add_argument('--window-size', type=int, default=60, help='滑动窗口大小(帧数)')
    parser.add_argument('--max-latency', type=float, default=5.0, help='最大等待时间(秒)')
    
    args = parser.parse_args()
    
    aggregator = ResultAggregator(
        redis_host=args.redis_host,
        redis_port=args.redis_port
    )
    
    # 设置滑动窗口参数
    aggregator.window_size = args.window_size
    aggregator.max_latency = args.max_latency
    
    try:
        aggregator.start()
    except KeyboardInterrupt:
        print("程序被用户中断")
    finally:
        aggregator.stop()