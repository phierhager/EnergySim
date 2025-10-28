from dataclasses import dataclass, field

@dataclass
class Registry:
    """A central registry for components and their underlying models."""
    components: dict = field(default_factory=dict)
    models: dict = field(default_factory=dict)

registry = Registry()

def register_component(config_cls):
    """Decorator to register a component class with its config class."""
    def decorator(cls):
        registry.components[config_cls.__name__] = cls
        return cls
    return decorator

def register_model(config_cls):
    """Decorator to register a model class with its config class."""
    def decorator(cls):
        registry.models[config_cls.__name__] = cls
        return cls
    return decorator
