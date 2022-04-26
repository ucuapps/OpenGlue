from .model import SuperPointNet, SuperPointNetBn

methods = {
    'SuperPointNet': SuperPointNet,
    'SuperPointNetBn': SuperPointNetBn
    # register new methods here
}

__all__ = ['methods']
