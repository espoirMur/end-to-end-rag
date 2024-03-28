# %load Dockerfile
# Use the base image
FROM jackiexiao/tritonserver:23.12-onnx-py-cpu



# Install the required Python packages
RUN pip install transformers==4.27.1 sacremoses==0.1.1


