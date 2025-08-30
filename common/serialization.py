import io
import torch


def state_dict_to_bytes(state_dict) -> bytes:
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    return buffer.getvalue()


def bytes_to_state_dict(b: bytes):
    buffer = io.BytesIO(b)
    buffer.seek(0)
    return torch.load(buffer, map_location="cpu")