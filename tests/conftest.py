"""Shared test fixtures and helpers."""

from skillmodels.model_spec import ModelSpec


def model_spec_from_yaml_dict(d: dict) -> ModelSpec:
    """Create a ModelSpec from a YAML-loaded dictionary.

    Args:
        d: A dictionary loaded from a YAML model specification file.

    Returns:
        A ModelSpec instance.

    """
    return ModelSpec.from_dict(d)
