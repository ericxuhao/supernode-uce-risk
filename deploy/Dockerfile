# 监测服务镜像（节点 Agent 需要访问宿主机的计算卡与内核日志，建议直接用 systemd 部署在节点上）
#   docker build -t supernode-server -f deploy/Dockerfile .
#   docker run -d -p 8765:8765 -v $PWD/var:/data --name supernode supernode-server
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/supernode
COPY requirements-lock.txt requirements.txt ./
RUN pip install --no-cache-dir -r requirements-lock.txt || pip install --no-cache-dir -r requirements.txt
COPY uce_risk ./uce_risk
COPY supernode ./supernode
COPY models ./models
COPY demo_data ./demo_data
ENV HOST=0.0.0.0 PORT=8765 SUPERNODE_DATA_DIR=/data
VOLUME ["/data"]
EXPOSE 8765
CMD ["python", "supernode/server.py"]
