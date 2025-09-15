from bems_simulation.core.components.remote.shared.connection import (
    IRemoteConnection,
)
from abc import ABC, abstractmethod
from bems_simulation.core.components.shared.component_base import ComponentBase
from bems_simulation.core.components.shared.spaces import Space


class RemoteComponent(ComponentBase, ABC):
    @abstractmethod
    def __init__(self, connection: IRemoteConnection, action_space: dict[str, Space]):
        pass
