import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.common.initializer import Normal

from mmdet.core.bbox import delta2bbox
from mmdet.ops.nms import batched_nms
from .anchor_head import AnchorHead
from ..registry import HEADS

@HEADS.register_module
class AABORPNHead(AnchorHead):
    def __init__(self, in_channels, **kwargs):
        super(AABORPNHead, self).__init__(2, in_channels, **kwargs)
        self.feat_channels = kwargs['feat_channels']
        self.num_anchors_list = kwargs['num_anchors_list']
        self.cls_out_channels = kwargs['cls_out_channels']
        self.target_means = Tensor(kwargs['target_means'])
        self.target_stds = Tensor(kwargs['target_stds'])
        self.use_sigmoid_cls = kwargs.get('use_sigmoid_cls', False)
        self.rpn_conv = nn.Conv2d(self.in_channels, self.feat_channels, kernel_size=3, stride=1, padding=1, has_bias=True)
        self.rpn_cls = nn.CellList()
        self.rpn_reg = nn.CellList()
        for i in range(5):
            self.rpn_cls.append(nn.Conv2d(self.feat_channels, self.num_anchors_list[i] * self.cls_out_channels, kernel_size=1, stride=1, has_bias=True, pad_mode='pad'))
            self.rpn_reg.append(nn.Conv2d(self.feat_channels, self.num_anchors_list[i] * 4, kernel_size=1, stride=1, has_bias=True, pad_mode='pad'))
        self.concat = ops.Concat(axis=0)
        self.reshape = ops.Reshape()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(axis=1)
        self.permute = ops.Transpose()
        self.nms = ops.NMSWithMask(0.5)
        self.batched_nms = batched_nms

    def construct(self, feats):
        rpn_cls_scores, rpn_bbox_preds = (), ()
        for i, feat in enumerate(feats):
            x = self.rpn_conv(feat)
            x = P.ReLU()(x)
            cls_scores = self.rpn_cls[i](x)
            bbox_preds = self.rpn_reg[i](x)
            rpn_cls_scores += (cls_scores,)
            rpn_bbox_preds += (bbox_preds,)
        return rpn_cls_scores, rpn_bbox_preds

    def loss(self, rpn_cls_scores, rpn_bbox_preds, gt_bboxes, img_metas, cfg, gt_bboxes_ignore=None):
        losses = super(AABORPNHead, self).loss(
            rpn_cls_scores,
            rpn_bbox_preds,
            gt_bboxes,
            None,
            img_metas,
            cfg,
            gt_bboxes_ignore=gt_bboxes_ignore)
        return dict(loss_rpn_cls=losses['loss_cls'], loss_rpn_bbox=losses['loss_bbox'])
      
    def get_bboxes_single(self, cls_scores, bbox_preds, mlvl_anchors, img_shape, scale_factor, cfg, rescale=False):
        mlvl_proposals = []
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            assert rpn_cls_score.shape[-2:] == rpn_bbox_pred.shape[-2:]
            anchors = mlvl_anchors[idx]
            rpn_cls_score = ops.Transpose()(rpn_cls_score, (1, 2, 0))
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = ops.Sigmoid()(rpn_cls_score)
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                scores = ops.Softmax()(rpn_cls_score, axis=1)[:, 1]
            rpn_bbox_pred = ops.Transpose()(rpn_bbox_pred, (1, 2, 0)).reshape(-1, 4)
            if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
                _, topk_inds = ops.TopK(sorted=True)(scores, cfg.nms_pre)
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]
                scores = scores[topk_inds]
            proposals = delta2bbox(anchors, rpn_bbox_pred, self.target_means, self.target_stds, img_shape)
            if cfg.min_bbox_size > 0:
                w = proposals[:, 2] - proposals[:, 0] + 1
                h = proposals[:, 3] - proposals[:, 1] + 1
                valid_inds = ops.NonZero()(ops.logical_and(w >= cfg.min_bbox_size, h >= cfg.min_bbox_size)).squeeze()
                proposals = proposals[valid_inds, :]
                scores = scores[valid_inds]
            proposals = ops.cat()(proposals, ops.ExpandDims()(scores, -1), axis=-1)
            proposals, _ = nms(proposals, cfg.nms_thr)
            proposals = proposals[:cfg.nms_post, :]
            mlvl_proposals.append(proposals)
        proposals = ops.cat()(mlvl_proposals, 0)
        if cfg.nms_across_levels:
            proposals, _ = nms(proposals, cfg.nms_thr)
            proposals = proposals[:cfg.max_num, :]
        else:
            scores = proposals[:, 4]
            num = min(cfg.max_num, proposals.shape[0])
            _, topk_inds = ops.TopK(sorted=True)(scores, num)
            proposals = proposals[topk_inds, :]
        return proposals
