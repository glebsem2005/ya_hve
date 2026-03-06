from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class GpsPosition:
    lat: float
    lon: float
    alt: float = 50.0

@dataclass
class Photo:
    data: bytes
    lat: float
    lon: float

    @property
    def b64(self) -> str:
        import base64
        return base64.b64encode(self.data).decode()


class DroneInterface(ABC):

    @abstractmethod
    async def takeoff(self) -> None: ...

    @abstractmethod
    async def fly_to(self, lat: float, lon: float):
        """Async generator — yields GpsPosition while flying."""
        ...

    @abstractmethod
    async def capture_photo(self) -> Photo: ...

    @abstractmethod
    async def return_home(self) -> None: ...
