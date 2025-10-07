from dataclasses import dataclass, field


@dataclass
class Registry:
    components: dict = field(default_factory=dict)
    models: dict = field(default_factory=dict)
    connections: dict = field(default_factory=dict)


registry = Registry()


def register_local_component(config_cls):
    def decorator(cls):
        is_local = True
        registry.components[config_cls.__name__] = (cls, is_local)
        return cls

    return decorator


def register_remote_component(config_cls):
    def decorator(cls):
        is_local = False
        registry.components[config_cls.__name__] = (cls, is_local)
        return cls

    return decorator


def register_model(config_cls):
    def decorator(cls):
        registry.models[config_cls.__name__] = cls
        return cls

    return decorator


def register_connection(config_cls):
    def decorator(cls):
        registry.connections[config_cls.__name__] = cls
        return cls

    return decorator
