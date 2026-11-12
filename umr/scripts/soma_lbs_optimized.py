"""Equivalent dense LBS with per-vertex blended transforms, avoiding J*V positions."""
import torch


def lbs_vertex_chunked(bind_shape,skinning_weights,target_pose_world,chunk_vertices=1024):
    from soma.geometry.lbs import lbs
    vertices=bind_shape.shape[-2]
    if vertices<=chunk_vertices:
        return lbs(bind_shape,skinning_weights,target_pose_world)
    return torch.cat([lbs(bind_shape[...,start:start+chunk_vertices,:],
                          skinning_weights[start:start+chunk_vertices],target_pose_world)
                      for start in range(0,vertices,chunk_vertices)],dim=-2)


def lbs_blended_transform(bind_shape, skinning_weights, target_pose_world):
    if bind_shape.ndim<2 or bind_shape.shape[-1]!=3:
        raise ValueError('bind_shape must end in (V,3)')
    if target_pose_world.ndim<3 or target_pose_world.shape[-2:]!=(4,4):
        raise ValueError('target_pose_world must end in (J,4,4)')
    vertices=bind_shape.shape[-2]; joints=target_pose_world.shape[-3]
    if skinning_weights.shape!=(vertices,joints):
        raise ValueError('skinning_weights must have shape (V,J)')
    if bind_shape.ndim>2 and bind_shape.shape[:-2]!=target_pose_world.shape[:-3]:
        raise ValueError('batched bind/target leading dimensions must match')
    blended=torch.einsum('vj,...jmk->...vmk',skinning_weights,target_pose_world[...,:3,:])
    return torch.einsum('...vmk,...vk->...vm',blended[...,:3],bind_shape)+blended[...,3]
