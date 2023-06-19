import mindspore as ms
import numpy as np

class AnchorGenerator():
    """
    Examples:
        >>> from mmdet.core import AnchorGenerator
        >>> self = AnchorGenerator(9, [1.], [1.])
        >>> all_anchors = self.grid_anchors((2, 2), device='cpu')
        >>> print(all_anchors)
        tensor([[ 0.,  0.,  8.,  8.],
                [16.,  0., 24.,  8.],
                [ 0., 16.,  8., 24.],
                [16., 16., 24., 24.]])
    """
    def __init__(self, base_size, scales, ratios, scale_major=True, ctr=None):
        self.base_size = base_size
        # Change here
        if len(scales) > 1:
            self.multi_scale = True
        else:
            self.multi_scale = False
        # Change over
        self.scales = ms.Tensor(scales, ms.float32)
        self.ratios = ms.Tensor(ratios, ms.float32)
        self.scale_major = scale_major
        self.ctr = ctr
        self.base_anchors = self.gen_base_anchors()

    @property
    def num_base_anchors(self):
        return self.base_anchors.shape[0]

    def gen_base_anchors(self):
        w = self.base_size
        h = self.base_size
        if self.ctr is None:
            x_ctr = 0.5 * (w - 1)
            y_ctr = 0.5 * (h - 1)
        else:
            x_ctr, y_ctr = self.ctr

        h_ratios = ms.sqrt(self.ratios)
        w_ratios = 1 / h_ratios
        # Change here
        # Multi Scale:
        if self.multi_scale:
            if self.scale_major:
                ws = (w * w_ratios * self.scales).reshape(-1)
                hs = (h * h_ratios * self.scales).reshape(-1)
            else:
                ws = (w * self.scales * w_ratios).reshape(-1)
                hs = (h * self.scales * h_ratios).reshape(-1)
        else:
            if self.scale_major:
                ws = (w * w_ratios[:, None] * self.scales[None, :]).reshape(-1)
                hs = (h * h_ratios[:, None] * self.scales[None, :]).reshape(-1)
            else:
                ws = (w * self.scales[:, None] * w_ratios[None, :]).reshape(-1)
                hs = (h * self.scales[:, None] * h_ratios[None, :]).reshape(-1)
        # Change over
        # yapf: disable
        base_anchors = ms.stack(
            [
                x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1),
                x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)
            ],
            axis=-1).round()
        # yapf: enable

        return base_anchors

    def _meshgrid(self, x, y, row_major=True):
        xx = x.repeat(len(y))
        yy = y.reshape(-1, 1).repeat(1, len(x)).reshape(-1)
        if row_major:
            return xx, yy
        else:
            return yy, xx

    def grid_anchors(self, featmap_size, stride=16, device='cuda'):
        base_anchors = self.base_anchors.to(ms.context(device))

        feat_h, feat_w = featmap_size
        shift_x = np.arange(0, feat_w, device=device) * stride
        shift_y = np.arange(0, feat_h, device=device) * stride
        shift_xx, shift_yy = np.meshgrid(shift_x, shift_y)
        shifts = ms.Tensor(np.stack([shift_xx, shift_yy, shift_xx, shift_yy], axis=-1), dtype=ms.float32)
        shifts = shifts.astype(base_anchors.dtype)
        # first feat_w elements correspond to the first row of shifts
        # add A anchors (1, A, 4) to K shifts (K, 1, 4) to get
        # shifted anchors (K, A, 4), reshape to (K*A, 4)

        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.view(-1, 4)
        # first A rows correspond to A anchors of (0, 0) in feature map,
        # then (0, 1), (0, 2), ...
        return all_anchors

    def valid_flags(self, featmap_size, valid_size, device='cuda'):
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = ms.zeros((feat_w,), ms.uint8)
        valid_y = ms.zeros((feat_h,), ms.uint8)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = np.meshgrid(valid_x.asnumpy(), valid_y.asnumpy())
        valid = (valid_xx & valid_yy).astype(ms.uint8)
        valid = valid[:, None].broadcast_to((valid.shape[0], self.num_base_anchors)).contiguous_view().reshape(-1)
        return valid
