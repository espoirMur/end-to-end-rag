IN this project I will try to build a RAG system at leas from stratch on medical dataset. 

I will be using pubmed dataset and I will try to deploy a model in production setting using limited ressources.

We will be using the following elements: 


- https://huggingface.co/michiyasunaga/BioLinkBERT-large:  As language model to use to learn embedding for both question and paragraphs
- https://huggingface.co/datasets/pubmed_qa/viewer/pqa_unlabeled?row=0: the dataset to use as it contains the questions and the paragraphs..
- For the generative model to use to generate answers will be : https://github.com/stanford-crfm/BioMedLM
- Another tutorial to use : https://www.anyscale.com/blog/a-comprehensive-guide-for-building-rag-based-llm-applications-part-1
- https://aws.amazon.com/blogs/database/building-ai-powered-search-in-postgresql-using-amazon-sagemaker-and-pgvector/
- Use this https://www.backblaze.com/sign-up/cloud-storage?referrer=pricing for cloud storage
- https://blog.ml6.eu/triton-ensemble-model-for-deploying-transformers-into-production-c0f727c012e3

- https://github.com/oracle-samples/oci-data-science-ai-samples/blob/main/model-deployment/containers/Triton/gpt2_ensemble/Deploy_GPT2_Ensemble.md

- https://huggingface.co/blog/how-to-generate

- Infrastructure:
 https://www.runpod.io/console/gpu-cloud

 - https://blog.griddynamics.com/retrieval-augmented-generation-llm/

 - https://medium.com/@neum_ai/retrieval-augmented-generation-at-scale-building-a-distributed-system-for-synchronizing-and-eaa29162521

 - https://github.com/infiniflow/ragflow/blob/main/rag/llm/embedding_model.py

 - https://www.pinecone.io/learn/hybrid-search-intro/

 -  https://haystack.deepset.ai/blog/haystack-nvidia-nim-rag-guide

 - https://www.cloudraft.io/blog/deploy-llms-on-kubernetes-using-openllm
 

On deployment with triton server...
ffff
This is not a research paper it is an engineer implementation, of an ML system to see if things will work in Production... 
f


Renaming Task:

- Build the Generative Model
- Deploy generative ai and set it container
- Deploy it with Triton, opitmize 
- Setup KubeFlow 
- Setup design the kubernetes components.
- Document the whole process.
- Use this process to clean the notebook: https://zhauniarovich.com/post/2020/2020-10-clearing-jupyter-output-p3/
- Use the kuberentes.https://developer.nvidia.com/blog/deploying-nvidia-triton-at-scale-with-mig-and-kubernetes/

- https://blog.marvik.ai/2023/10/16/deploying-llama2-with-nvidia-triton-inference-server/

The second project will be news summarization 

Good bored and tired of this RAG stuff, let me try news summarizaiton:

https://towardsdatascience.com/summarize-reddit-comments-using-t5-bart-gpt-2-xlnet-models-a3e78a5ab944


https://towardsdatascience.com/summarize-reddit-comments-using-t5-bart-gpt-2-xlnet-models-a3e78a5ab944


Other project ideas

- Scrape the data from edup epst on exetat and analyse the data about diplome d'etat and find which category of school have the best results on the national state exam.

- Build a project where small buisiness can share their portfolio, such as macon, carelleurs, etc.



