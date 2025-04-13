from dataclasses import dataclass
from typing import Dict

@dataclass
class Task:
    """任务
        stream_id: 产生该任务的流ID
        task_id: 任务ID
        generated_time: 任务生成时间(单位: 秒)
        frame: 任务处理的图像帧(base64编码)
        received_time: 任务接收时间(单位: 秒)
        start_process_time: 任务开始处理时间(单位: 秒)
        end_process_time: 任务结束处理时间(单位: 秒)
        
    """
    stream_id: int
    task_id: int
    generated_time: float
    frame: str
    received_time: float = None
    start_process_time: float = None
    end_process_time: float = None
    extra_info: Dict = None


    @staticmethod
    def from_dict(data: Dict):
        """从字典中加载任务数据"""
        return Task(
            stream_id=data.get('stream_id'),
            task_id=data.get('task_id'),
            generated_time=data.get('generated_time'),
            frame=data.get('frame'),
            received_time=data.get('received_time'),
            start_process_time=data.get('start_process_time'),
            end_process_time=data.get('end_process_time'),
            extra_info=data.get('extra_info', {})
        )
    
    def to_dict(self) -> Dict:
        """将任务数据转换为字典"""
        return {
            'stream_id': self.stream_id,
            'task_id': self.task_id,
            'generated_time': self.generated_time,
            'frame': self.frame,
            'received_time': self.received_time,
            'start_process_time': self.start_process_time,
            'end_process_time': self.end_process_time,
            'extra_info': self.extra_info
        }

