from enum import Enum


class ComponentType(int, Enum):


    FULL = 0
    SWA = 1
    MAMBA = 2
    C128 = 3

    def __str__(self) -> str:
        return self.name.lower()

    @property
    def is_full(self) -> bool:
        return self == ComponentType.FULL

    @property
    def is_swa(self) -> bool:
        return self == ComponentType.SWA

    @property
    def is_mamba(self) -> bool:
        return self == ComponentType.MAMBA


BASE_COMPONENT_TYPE = ComponentType.FULL
