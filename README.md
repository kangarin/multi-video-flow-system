# multi-video-flow-system

1. 启动redis
docker run --name my-redis -p 6379:6379 -d redis

2. 清理数据库（如果上次运行）
python setup/db_initializer.py

3. 在不同节点上启动一组服务器，序号必须从1开始连续
python processor/stream_processor.py --id 1 --port 7777 --model s
python processor/stream_processor.py --id 2 --port 7778 --model s
...

4. 在不同节点启动一组生成器，序号必须从1开始连续
python generator/stream_generator.py --id 1 --video-dir /path/to/videos
python generator/stream_generator.py --id 2 --video-dir /path/to/videos
...

5. 在服务器启动调度器更新决策
python scheduler/db_scheduler_reader_writer.py