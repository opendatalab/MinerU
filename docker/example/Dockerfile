# Use the official sglang image
FROM lmsysorg/sglang:v0.4.7-cu124

# install mineru latest
RUN python3 -m pip install -U 'mineru[core]' --break-system-packages

# Download models and update the configuration file
RUN /bin/bash -c "mineru-models-download -s huggingface -m all"

COPY example /tmp/example

ENV GRADIO_SERVER_NAME=0.0.0.0

WORKDIR /tmp/example
RUN /bin/bash -c "source /opt/mineru_venv/bin/activate && \
    pip3 install -r requirements.txt"

# Set the entry point to activate the virtual environment and run the command line tool
ENTRYPOINT ["/bin/bash", "-c", "export MINERU_MODEL_SOURCE=local && exec \"$@\"", "--"]