

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


End of the day managed to make stuff work.

Next step write a blog on how to mount data volume .


https://dev.to/otomato_io/mount-s3-objects-to-kubernetes-pods-12f5

- https://blog.meain.io/2020/mounting-s3-bucket-kube/

Write a blog post on the mount volume
Write a post on deployment with kubernetes
Set ci/cd for the deployment
Prometheus and graphana for loggin.


### Notes On Retrieval

Back to the RAG, I want to work on the retrieval model.

Read on this document to learn more about chunking: https://github.com/pinecone-io/examples/blob/master/learn/generation/better-rag/02a-alt-chunking-methods.ipynb

Implement chuncking with document at scale. Try picetone database.


### As of 12 July 2024

I managed to ingested 86k Documents in postgres database and calculate the embedding of those documents using bert case.

How we need to implement the search feature that will be using hybrid search (embedding) + keywords.

I need to have a look at this chat application that implemented everything https://github.com/Azure-Samples/rag-postgres-openai-python/tree/main/src/fastapi_app
I need to read more on [hybrid search ](https://techcommunity.microsoft.com/t5/ai-azure-ai-services-blog/azure-ai-search-outperforming-vector-search-with-hybrid/ba-p/3929167) and follow the project to read more on the implementation.


# Tuesday 16th of July 2024

We managed to get hte question answering generation working wit h a T5 models, the next step will be to anodate at least 1k questions with the ocontext, then fine tune the model on them and generate an improved model.


Wednsday 17th July

I discovered the croissan LLM model. 

https://huggingface.co/blog/manu/croissant-llm-blog

I need to explore it.


I have managed to run the croissant model locally on my laptop which is really fun, it has good strong french capabilities. I will use it to annodate my question, but in the future I will keep on using it to for more cool stuff on my dataset.


I discover https://www.reddit.com/r/LocalLLaMA/comments/15ak5k4/short_guide_to_hosting_your_own_llamacpp_openai/

Friday 19 July,


Note 19 on the tokeniser, I seem to find out what the issue was. 

I basically need to edit the code \u0000 from the tokenire


### Update on 23 July!

The RAG experiment is considered as completed, the next step is to organize the code into module and start the productionarization.
