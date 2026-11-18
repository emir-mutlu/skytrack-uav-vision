# Public demo on Hugging Face Spaces (Docker SDK).
# Runs the closed-loop simulation console in read-only mode on port 7860.
# Copy this file and README.md into a new Docker Space; the source is pulled from GitHub.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 fonts-dejavu-core && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN useradd -m -u 1000 user
WORKDIR /home/user

# Branch or tag to build from
ARG SKYTRACK_REF=master
ADD https://github.com/emir-mutlu/skytrack-uav-vision/archive/${SKYTRACK_REF}.tar.gz /tmp/skytrack.tar.gz
RUN mkdir app && tar xzf /tmp/skytrack.tar.gz --strip-components=1 -C app && rm /tmp/skytrack.tar.gz \
    && pip install --no-cache-dir -e "./app[mavlink]"

USER user
WORKDIR /home/user/app
EXPOSE 7860
CMD ["skytrack", "console", "--host", "0.0.0.0", "--port", "7860", "--no-browser", "--read-only"]
