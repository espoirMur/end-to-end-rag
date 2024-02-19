# %load Dockerfile
# Use the base image
FROM nvcr.io/nvidia/tritonserver:23.06-py3

# Install the required Python packages
RUN pip install transformers==4.27.1


