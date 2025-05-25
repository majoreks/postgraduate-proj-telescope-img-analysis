from enum import Enum
from pathlib import Path

class DataType(Enum):
    LABEL = 1
    IMAGE = 2

class FilePath():
    __IMAGE_PATH_POSTFIX = "_V_imc.fits.gz"
    __LABEL_PATH_POSTFIX = "_V_imc_trl.dat"

    __IMAGE_PATH_DIR = "RED"
    __LABEL_PATH_DIR = "CAT"

    def __init__(self, key: str, type: DataType):
        self.__key = key
        self.__dataType = type

    def __str__(self) -> str:
        return self.path

    @property
    def path(self) -> str:
        return f"{self.__key[3:].split('.')[0]}/{self.__dir}/{self.__key}{self.__postfix}"

    @property
    def __postfix(self) -> str:
        if self.__dataType == DataType.IMAGE:
            return self.__IMAGE_PATH_POSTFIX
        elif self.__dataType == DataType.LABEL:
            return self.__LABEL_PATH_POSTFIX
        else:
            raise Exception("Unknown data type", self.__dataType)
        
    @property
    def __dir(self) -> str:
        if self.__dataType == DataType.IMAGE:
            return self.__IMAGE_PATH_DIR
        elif self.__dataType == DataType.LABEL:
            return self.__LABEL_PATH_DIR
        else:
            raise Exception("Unknown data type", self.__dataType)

def get_basename_prefix(path: Path) -> str:
    return path.name.split("_V_")[0]

def build_path(key: str, type: DataType) -> FilePath:
    return FilePath(key, type)