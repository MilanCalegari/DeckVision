from enum import Enum


class OS(str, Enum):
    DARWIN = "Darwin" # Mac
    WINDOWS = "Windows"
    LINUX = "Linux"
