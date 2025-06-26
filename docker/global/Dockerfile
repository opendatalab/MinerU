# Use the official sglang image
FROM lmsysorg/sglang:v0.4.8-cu124

# install mineru latest
RUN python3 -m pip install -U 'mineru[core]' --break-system-packages

# Download models and update the configuration file
RUN /bin/bash -c "mineru-models-download -s huggingface -m all"

# Set the entry point to activate the virtual environment and run the command line tool
ENTRYPOINT ["/bin/bash", "-c", "export MINERU_MODEL_SOURCE=local && exec \"$@\"", "--"]