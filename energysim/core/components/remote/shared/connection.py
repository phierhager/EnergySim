from abc import ABC, abstractmethod
from typing import Any, Dict


class IRemoteConnection(ABC):
    """Generic interface for a remote connection."""

    @abstractmethod
    def connect(self) -> None:
        """Establish the connection."""
        pass

    @abstractmethod
    def send(self, data: Dict[str, Any]) -> None:
        """Send data to the remote endpoint."""
        pass

    @abstractmethod
    def receive(self) -> Dict[str, Any]:
        """Receive data from the remote endpoint."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Close the connection."""
        pass
