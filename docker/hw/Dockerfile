FROM franky_franky-run:latest

RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*

COPY build/wheels/ /tmp/wheels/
RUN pip3 install --no-index --find-links=/tmp/wheels/ \
    /tmp/wheels/franky_control-2.0.0-cp310-cp310-linux_x86_64.whl

WORKDIR /workspace
CMD ["/bin/bash"]
