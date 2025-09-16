registry = {}


def register(config_cls):
    def decorator(cls):
        registry[config_cls.__name__] = cls
        return cls

    return decorator
