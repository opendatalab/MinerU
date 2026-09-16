"""公式增量生成专用缓存；缓存仅由当前调用持有，不写入模型权重。"""

import torch


class GrowingLayerCache(tuple):
    """兼容四项层缓存协议，按几何增长容量保存 self-attention K/V。"""

    def __new__(cls, states, buffers=None):
        """保留普通元组索引协议，并携带当前调用独享的底层缓冲区。"""
        instance = super().__new__(cls, states)
        instance.buffers = buffers
        return instance

    def append(self, key, value):
        """仅写入新增 token，容量不足时扩容，避免逐轮复制完整历史。"""
        if torch.is_grad_enabled():
            raise RuntimeError("GrowingLayerCache is only valid during inference")
        length = self[0].shape[2]
        needed = length + key.shape[2]
        buffers = self.buffers
        if buffers is None or needed > buffers[0].shape[2]:
            capacity = max(64, 1 << (needed - 1).bit_length())
            shape = (*key.shape[:2], capacity, key.shape[3])
            buffers = (key.new_empty(shape), value.new_empty(shape))
            if length:
                buffers[0][:, :, :length].copy_(self[0])
                buffers[1][:, :, :length].copy_(self[1])
        buffers[0][:, :, length:needed].copy_(key)
        buffers[1][:, :, length:needed].copy_(value)
        return type(self)(
            (buffers[0][:, :, :needed], buffers[1][:, :, :needed]), buffers
        )

    def __add__(self, cross_states):
        """拼接 cross-attention 状态时保留 self-attention 缓冲区所有权。"""
        return type(self)(tuple(self) + tuple(cross_states), self.buffers)
