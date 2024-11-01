from pathlib import Path

import yaml
from argo_workflows.api_client import ApiClient
from argo_workflows.apis import WorkflowServiceApi
from argo_workflows.configuration import Configuration
from argo_workflows.model.io_argoproj_workflow_v1alpha1_workflow_create_request import (
	IoArgoprojWorkflowV1alpha1WorkflowCreateRequest,
)

API_HOST = "https://127.0.0.1:2746"
pipelines_directory = Path(__file__).parent.parent.parent.parent.joinpath(
	"pipelines_files"
)
ingestion_pipeline_file = pipelines_directory.joinpath("ingestion_pipeline.yaml")


def main():
	configuration = Configuration(API_HOST)
	configuration.verify_ssl = False
	client = ApiClient(configuration=configuration)
	service = WorkflowServiceApi(client)
	with open(ingestion_pipeline_file, "r") as file:
		manifest: dict = yaml.safe_load(file)
	request = IoArgoprojWorkflowV1alpha1WorkflowCreateRequest(
		workflow=manifest, _check_type=False
	)
	print(request)
	service.create_workflow(namespace="argo", body=request)


if __name__ == "__main__":
	main()
