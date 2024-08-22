from uuid import uuid4
from hera.workflows import Container, Workflow, Steps, Step, WorkflowsService, Resources
from pathlib import Path
PIPELINE_IMAGE = 'espymur/rag-doc-ingestion:latest'

pipeline_file = Path(__file__).parent.parent.parent.parent.joinpath(
    "pipelines_files")
workflows_service = WorkflowsService(host="https://localhost:2746",
                                     verify_ssl=False)
resources = Resources(cpu_request=2, memory_request='1Gi')
with Workflow(generate_name='indexing-step-',
              entrypoint='ingestion',
              namespace='argo',
              workflows_service=workflows_service,
              resources=resources) as workflow:
    container_uuid = str(uuid4())[:6]
    ingestion_container = Container(
        name=f'ingestion-{container_uuid}', image=PIPELINE_IMAGE, command=['python', 'ingestion/main.py'])
    with Steps(name='ingestion'):
        Step(name='ingestion', template=ingestion_container)
    workflow.to_file(pipeline_file)
    workflow.create()
