from abc import ABC

class ModelBase(ABC):
    """
    Abstract base class for all component models in the simulation framework.

    This class serves as a common 'marker' interface for all models,
    formally establishing the architectural pattern that every Component
    contains a Model.

    It does not enforce any specific abstract methods, because the fundamental
    responsibilities of different models (e.g., energy storage, generation,
    consumption) are too diverse to be captured in a single, universal contract.

    Specific model interfaces for families of models (e.g., IBatteryModel)
    should inherit from this base class to signal their role in the system.
    """
    pass