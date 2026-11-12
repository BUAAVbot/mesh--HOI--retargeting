"""Exact last-input no-grad identity cache, installed by the CPU SOMA adapter."""
import torch


def snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key:snapshot(item) for key,item in value.items()}
    if isinstance(value, tuple):
        return tuple(snapshot(item) for item in value)
    if value is None or type(value) in (bool,int,float,str):
        return value
    raise TypeError(f'unsupported cache key type: {type(value)}')


def equal(a,b):
    if type(a) is not type(b):
        return False
    if isinstance(a,torch.Tensor):
        return a.dtype==b.dtype and a.device==b.device and a.shape==b.shape and torch.equal(a,b)
    if isinstance(a,dict):
        return a.keys()==b.keys() and all(equal(a[key],b[key]) for key in a)
    if isinstance(a,tuple):
        return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def install(layer):
    """Use only on an exclusively owned layer; changing identity always recomputes."""
    original = layer.prepare_identity
    last = None
    stats = dict(prepared=0,reused=0)
    def prepare(identity_coeffs,scale_params=None,repose_to_bind_pose=True,global_scale=1.,kwargs=None):
        nonlocal last
        key = (identity_coeffs,scale_params,repose_to_bind_pose,global_scale,kwargs)
        # Never reuse autograd state. Unsupported keys safely use the original call.
        if not torch.is_grad_enabled():
            try:
                current = snapshot(key)
            except TypeError:
                current = None
            if current is not None and last is not None and equal(current,last):
                stats['reused'] += 1
                return
        else:
            current = None
        last = None
        original(identity_coeffs,scale_params,repose_to_bind_pose,global_scale,kwargs)
        last = current
        stats['prepared'] += 1
    layer.prepare_identity = prepare
    def restore():
        layer.prepare_identity = original
    return stats,restore
