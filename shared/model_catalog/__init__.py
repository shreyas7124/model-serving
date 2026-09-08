"""Multi-model catalog shared by all apps (Switchyard-style JSON)."""
from .catalog import ModelCatalog, deploy_app_models, load_app_models

__all__ = [
    "ModelCatalog",
    "load_app_models",
    "deploy_app_models",
]
