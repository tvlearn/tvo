""" Custom serializer for pytorch tensors

    Author: Dmytro Velychko
    Institution: Carl von Ossietzky University of Oldenburg
    Email: dmytro.velychko@uni-oldenburg.de
"""

import json
import torch
import numpy as np


def tensor_to_dict(t):
    if isinstance(t, torch.Tensor):
        return {"type": "torch.Tensor",
                "dtype": str(t.dtype), 
                "device": t.device.type, 
                "requires_grad": t.requires_grad,
                "data": t.to("cpu").data.numpy().tolist()}
    else:
        raise ValueError("Expected pytorch tensor")


def ndarray_to_dict(t):
    if isinstance(t, np.ndarray):
        return {"type": "numpy.ndarray",
                "dtype": str(t.dtype), 
                "data": t.tolist()}
    else:
        raise ValueError("Expected numpy array")


def tensor_to_dict_hook(t):
    if isinstance(t, torch.Tensor):
        return tensor_to_dict(t)
    if isinstance(t, np.ndarray):
        return ndarray_to_dict(t)
    return t


def dict_to_tensor(d):
    if "type" in d and d["type"] == "torch.Tensor":
        data = np.array(d["data"])
        if d["dtype"] == str(torch.int64):
            dtype = torch.int64
        elif d["dtype"] == str(torch.int32):
            dtype = torch.int32
        elif d["dtype"] == str(torch.float64):
            dtype = torch.float64
        elif d["dtype"] == str(torch.float32):
            dtype = torch.float32
        res = torch.tensor(data, requires_grad=d["requires_grad"]).to(dtype)
        try:
            res = res.to(d["device"])
        except:
            pass
        return res
    else:
        raise ValueError("Expected serialized pytorch tensor")


def dict_to_ndarray(d):
    if "type" in d and d["type"] == "numpy.ndarray":
        if d["dtype"] == str(np.int64.__name__):
            dtype = np.int64
        elif d["dtype"] == str(np.int32.__name__):
            dtype = np.int32
        elif d["dtype"] == str(np.float64.__name__):
            dtype = np.float64
        elif d["dtype"] == str(np.float32.__name__):
            dtype = np.float32
        res = np.array(d["data"], dtype=dtype)
        return res
    else:
        raise ValueError("Expected serialized numpy array")


def dict_to_tensor_hook(d):
    if "type" in d and d["type"] == "torch.Tensor":
        return dict_to_tensor(d)
    elif "type" in d and d["type"] == "numpy.ndarray":
        return dict_to_ndarray(d)
    else:
        return d


class DataJSON(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, torch.Tensor):
            return tensor_to_dict(o)
        if isinstance(o, np.ndarray):
            return ndarray_to_dict(o)
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, o) 
    

def dumps(*args, **kwargs):
    return json.dumps(*args, cls=DataJSON, **kwargs)

def dump(*args, **kwargs):
    return json.dump(*args, cls=DataJSON, **kwargs)


def loads(*args, **kwargs):
    return json.loads(*args, object_hook=dict_to_tensor_hook, **kwargs)

def load(*args, **kwargs):
    return json.load(*args, object_hook=dict_to_tensor_hook, **kwargs)



