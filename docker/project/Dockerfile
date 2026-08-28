FROM franky-hw:latest

# Project's own Python deps (GTSAM, OSQP, MuJoCo, etc.)
COPY mppi-arm-gpmp2-cbf/requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt
RUN pip3 install "numpy<2.0" --force-reinstall

# Bake in the project code (dev iteration can still override via -v mount)
COPY mppi-arm-gpmp2-cbf/ /workspace/mppi-arm-gpmp2-cbf/

WORKDIR /workspace/mppi-arm-gpmp2-cbf
CMD ["/bin/bash"]
