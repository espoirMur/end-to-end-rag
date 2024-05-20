

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

#### Run the service with kuberenetes

- Run the deployment with 

`kubectl apply -f kubernetes/embedding-deployment.yml`

- Then run the service with 

`kubectl apply -f kubernetes/service.yaml`


After creating the service you need to run 

` minikube service service-name --url  `


Once you have a model running you can interact with it using the following url:

http://127.0.0.1:58498.

You can perform inference request to it as normal.


## As of 21st May 2024

The project is running fine and we have run it on kubernetes.
The next step are:

- Finding a way to run it a proper production setting with kubernetes and cloudblaze as model repository.
- Automate everything in a ML op fashion.
- Write a post to document the process.
I need to get back to this and stop with the laziness.
