import redis
import argparse

class RedisKeysCleaner:
    def __init__(self, host='localhost', port=6379, db=0):
        """初始化 Redis 连接"""
        self.redis_client = redis.Redis(host=host, port=port, db=db)
        print(f"已连接到 Redis 服务器: {host}:{port}")
    
    def clean_keys_by_pattern(self, pattern):
        """清理所有匹配指定模式的键"""
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                count = len(keys)
                print(f"发现 {count} 个键匹配模式 '{pattern}'")
                for key in keys:
                    value = self.redis_client.hgetall(key)  # 获取哈希表中的所有字段
                    print(f"键: {key.decode('utf-8')}, 值: {value}")  # 打印键值对
                    # 删除匹配的键
                    self.redis_client.delete(key)
                print(f"已删除所有匹配 '{pattern}' 的键")
            else:
                print(f"没有找到匹配模式 '{pattern}' 的键")
            
            return True
        except Exception as e:
            print(f"清理键 '{pattern}' 时出错: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description='清理 Redis 中的特定键')
    parser.add_argument('--host', type=str, default='localhost', help='Redis 主机地址')
    parser.add_argument('--port', type=int, default=6379, help='Redis 端口')
    
    args = parser.parse_args()
    
    cleaner = RedisKeysCleaner(host=args.host, port=args.port)
    
    # 要清理的键模式列表
    patterns = [
        "stream:*",
        "processor:*",
        "status:stream:*",
        "status:processor:*",
        "status:latency:*",
    ]
    
    print("=== 开始清理 Redis 键 ===")
    
    for pattern in patterns:
        cleaner.clean_keys_by_pattern(pattern)
    
    print("=== 清理完成! ===")

if __name__ == "__main__":
    main()