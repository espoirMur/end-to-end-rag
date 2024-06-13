

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



Need to find a better way to build the image. It shouldn't kept on failing

#### Run the service with kuberenetes


Make sure you have kubernetes and minikube installed to use kubernetes backend.

- ` minikube start --mount --mount-string ~/fuull/pathh/to/models_repository/:/models `

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



#### As of 8 June 2024

This what I call consitency, looll.


Managed to create a kubernetes Cluster on Oracle cloud.

Here are the sep to run it.


`Make sure you have your kubernetes config saved to a path.`

- Authenticate to your account using `oci session authenticate`
- setup the config `export KUBECONFIG=$HOME/.kube/config_oci`
- Make sure your authentication client is using client auth: `export OCI_CLI_AUTH=security_token`

With that setup you can now connect to your cluster:

With 

`kubectl get nodes`


### As of 13 June 2024

Why am I spending so much time dealing with kubernetes? Am I a DevOp/Infra engineer?

Basically in the past few days I spend sometime trying to share model data I have in my Oracle cloud bucket storage to my kubernetes pods. But that was unsuccessful.
I have managed to mount Oracle cloud storage in a container using s3fs-fuse library, but unfortuantely, I wasn't able to share the data to the model container.

I have tried other option but didn't work.

My last attempt tommorow will be to download the data from the bucket and share it with the containers using PVC.

> Note the :shared - it's a mount propagation modifier in the mountPath field that allows this volume to be shared by multiple pods/containers on the same node.

I want to stop this and continue with the rag stuff.


Ps: remember to create an empty folder named 1.
