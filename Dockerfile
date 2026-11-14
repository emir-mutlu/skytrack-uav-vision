# CPU image (classic + YOLO). For Jetson use an l4t-pytorch base image instead.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
ARG EXTRAS=""
RUN pip install --no-cache-dir -e ".${EXTRAS}"

ENTRYPOINT ["skytrack"]
CMD ["demo", "--no-show", "--save", "runs/demo.mp4", "--telemetry", "runs/demo.jsonl"]
