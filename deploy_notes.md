# 为不同架构构建镜像
docker build --platform linux/amd64 -t repo:5000/dwy/multi-video-flow-base:amd64 .
docker build --platform linux/arm64 -t repo:5000/dwy/multi-video-flow-base:arm64 .
docker push repo:5000/dwy/multi-video-flow-base:amd64
docker push repo:5000/dwy/multi-video-flow-base:arm64

# 创建并推送多架构清单
docker manifest create --insecure repo:5000/dwy/multi-video-flow-base:latest \
  repo:5000/dwy/multi-video-flow-base:amd64 \
  repo:5000/dwy/multi-video-flow-base:arm64

docker manifest push --insecure repo:5000/dwy/multi-video-flow-base:latest 