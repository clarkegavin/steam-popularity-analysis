# data/pipeline_factory.py
from typing import List, Type
from logs.logger import get_logger
import yaml
import importlib
from pipelines.base import Pipeline
logger = get_logger("PipelineFactory")

class PipelineFactory:
    """
    Factory for dynamically building pipelines from YAML.
    Supports arbitrary pipeline classes and parameters.
    """

    _registry = {}

    @classmethod
    def register_pipeline(cls, name: str, pipeline: Pipeline):
        cls._registry[name] = pipeline
        logger.info(f"Registered pipeline: {name}")

    @classmethod
    def get_pipeline(cls, name: str) -> Pipeline:
        pipeline = cls._registry.get(name)
        if not pipeline:
            cls.logger.warning(f"Pipeline '{name}' not found in registry")
        return pipeline

    @classmethod
    def build_pipelines_from_yaml(cls, yaml_path: str) -> List[Pipeline]:
        """Build pipelines dynamically from YAML config."""
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)

        pipelines: List[Pipeline] = []

        for entry in config.get("pipelines", []):
            if not entry.get("enabled", True):
                logger.info(f"Skipping disabled pipeline '{entry.get('name')}'")
                continue

            pipeline_type = entry.get("type")
            if not pipeline_type:
                logger.warning(f"Pipeline type missing for '{entry.get('name')}', skipping")
                continue

            try:
                # Import the class dynamically
                module_name, class_name = pipeline_type.rsplit(".", 1)
                module = importlib.import_module(module_name)
                klass: Type[Pipeline] = getattr(module, class_name)

                # Determine if class has from_config
                if hasattr(klass, "from_config") and callable(getattr(klass, "from_config")):
                    pipeline_instance = klass.from_config(entry)
                    logger.info(f"Used from_config() to create pipeline '{entry.get('name')}'")
                else:
                    # Pass only 'params' dict to constructor
                    params: Dict[str, Any] = entry.get("params", {})
                    pipeline_instance = klass(**params)
                    logger.info(f"Used __init__() to create pipeline '{entry.get('name')}'")

                pipelines.append(pipeline_instance)
                cls.register_pipeline(entry.get("name"), pipeline_instance)
                logger.info(f"Pipeline '{entry.get('name')}' ({pipeline_type}) created successfully")

            except Exception as e:
                logger.error(f"Failed to create pipeline '{entry.get('name')}': {e}")

        return pipelines