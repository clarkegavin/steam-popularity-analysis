# main.py
from dotenv import load_dotenv
from orchestrator.pipeline_orchestrator import PipelineOrchestrator
from pipelines.factory import PipelineFactory

load_dotenv()

def main():
    yaml_path = "config/pipelines.yaml"
    pipelines = PipelineFactory.build_pipelines_from_yaml(yaml_path)

    orchestrator = PipelineOrchestrator(pipelines=pipelines, parallel=False, max_retries=3)

    data = None  # If your first pipeline extracts data, this can be None

    X_train, X_test, y_train, y_test = orchestrator.run(data=data, target_column="Genre")


if __name__ == "__main__":
    main()