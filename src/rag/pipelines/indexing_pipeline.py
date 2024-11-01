from pathlib import Path
from uuid import uuid4

from hera.workflows import (
	Container,
	Resources,
	SecretEnv,
	Step,
	Steps,
	Workflow,
	WorkflowsService,
)

PIPELINE_IMAGE = "espymur/rag-doc-ingestion:latest"

pipeline_file = Path(__file__).parent.parent.parent.parent.joinpath("pipelines_files")
workflows_service = WorkflowsService(host="https://localhost:2746", verify_ssl=False)
resources = Resources(cpu_request=2, memory_request="2Gi")
postgres_host = SecretEnv(
	name="POSTGRES_HOST",
	secret_key="POSTGRES_HOST",
	secret_name="postgres-database-secrets",
)
postgres_password = SecretEnv(
	name="POSTGRES_PASSWORD",
	secret_key="POSTGRES_PASSWORD",
	secret_name="postgres-database-secrets",
)
postgres_user = SecretEnv(
	name="POSTGRES_USER",
	secret_key="POSTGRES_USER",
	secret_name="postgres-database-secrets",
)
postgres_port = SecretEnv(
	name="POSTGRES_PORT",
	secret_key="POSTGRES_PORT",
	secret_name="postgres-database-secrets",
)
postgres_database = SecretEnv(
	name="POSTGRES_DB",
	secret_key="POSTGRES_DB",
	secret_name="postgres-database-secrets",
)

secrets = [
	postgres_host,
	postgres_password,
	postgres_user,
	postgres_port,
	postgres_database,
]
with Workflow(
	generate_name="indexing-step-",
	entrypoint="ingestion",
	namespace="argo",
	workflows_service=workflows_service,
	resources=resources,
) as workflow:
	container_uuid = str(uuid4())[:6]
	ingestion_container = Container(
		name=f"ingestion-{container_uuid}",
		image=PIPELINE_IMAGE,
		command=["python", "ingestion/main.py"],
		env=secrets,
	)
	with Steps(name="ingestion"):
		Step(name="ingestion", template=ingestion_container)
	workflow.to_file(pipeline_file)
	workflow.create()
