

## Lauch minikube cluster, where the volumes are mapped in the cluster

`minikube start --mount --mount-string ~/Projects/Personal/end-to-end-rag/models_repository/:/models`


To allow the pod in the minikube cluster to pull local images, run the following

`eval $(minikube docker-env)  `


And then build the image.


#### Building smaller triton inference server cpu only:

Clone the triton inference server at `https://github.com/triton-inference-server/server` and run the following

git checkout r22.06
python3 build.py  \
    --enable-logging --enable-stats\
    --enable-tracing \
    --enable-metrics \
    --endpoint=http \
    --endpoint=grpc \
    --backend=ensemble \
    --backend=python \
    --backend=onnxruntime
docker tag tritonserver:latest tritonserver:22.06-onnx-py-cpu



Need to find a bettew way to build the image. It shouldn't kept on failing


After creating the service you need to run 

` minikube service service-name --url  `
