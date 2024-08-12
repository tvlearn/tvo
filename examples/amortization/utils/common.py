from enum import Enum
import torch


class FloatPrecision(Enum):
    float16 = "float16"
    float32 = "float32"
    float64 = "float64"

    def __str__(self):
        return str(self.value)

    @staticmethod
    def from_string(s):
        try:
            return FloatPrecision[s]
        except KeyError:
            raise ValueError()
        
    def torch_dtype(self):
        if self.value == FloatPrecision.float16.value:
            return torch.float16
        elif self.value == FloatPrecision.float32.value:
            return torch.float32
        elif self.value == FloatPrecision.float64.value:
            return torch.float64
        else:
            raise ValueError()
        