# AABO_mindspore
Ported AABO with Huawei mindspore.

## 1. Introduction
Since MMDetection is used as the basis in the open source AABO project, only the original part of AABO was code ported for this project.
- Referred to the official manual [PyTorch and MindSpore API Mapping Table](https://www.mindspore.cn/docs/en/r2.0.0-alpha/note/api_mapping/pytorch_api_mapping.html) for code porting.
- The following code was ported:
  - `AABO_mindspore/aabo_rpn_head.py`
- The following code was ported but not used:
  - `AABO_mindspore/mindspore_anchor_generator.py`
  - `AABO_mindspore/mindspore_anchor_head.py`
  - `AABO_mindspore/mindspore_aabo_mask_rcnn_r101_fpn_2x.py`
  - `AABO_mindspore/mindspore_aabo_htc_dcov_x101_64x4d_fpn_24e.py`
- Code was tested with mindspore 2.0.0rc1, mindinsight 2.0.0rc1, Pytorch 1.8.2 and [MMdetection v1.0rc1](https://github.com/open-mmlab/mmdetection/tree/v1.0rc1).

## 2. Notes
- Mapping Table

  |  PyTorch 1.8.2 APIs   | MindSpore APIs |
  |  :----  | :----  |
  | torch.cat  | mindspore.ops.cat |
  | torch.transpose  | mindspore.Tensor.transpose |
  | torch.Tensor  | mindspore.Tensor |
  | torch.sqrt  | mindspore.sqrt |
  | torch.stack  | mindspore.stack |
  | torch.zeros  | mindspore.zeros |
  | torch.nn.Conv2d  | mindspore.nn.Conv2d |
  | torch.nn.ReLU  | mindspore.nn.ReLU |
  | torch.nn.Sigmoid  | mindspore.nn.Sigmoid |
  | torch.nn.Softmax  | mindspore.nn.Softmax |
  | torch.topk  | mindspore.ops.TopK |
  | torch.nonzero  | mindspore.ops.nonzero |
  | torch.arange  | mindspore.numpy.arange |

- The default value of PyTorch's Conv2d parameter bias is True, which means that bias parameters are added by default, while the default value of MindSpore's parameter has_bias is False, which means that bias functions are not added by default.

- Note that I've used np.meshgrid to create the meshgrid arrays since MindSpore doesn't have an equivalent function. I've used astype instead of type_as to convert the shifts tensor to the same datatype as base_anchors. And, I've used broadcast_to and contiguous_view instead of expand and contiguous.

## 3. Implementation
- Replace the original files in MMDetection with  new files:
  - `AABO_mindspore/__init__.py` ---> `mmdetection/mmdet/models/anchor_heads/__init__.py`  
  - `AABO_mindspore/anchor_generator.py` --->`mmdetection/mmdet/core/anchor/anchor_generator.py`
  - `AABO_mindspore/anchor_head.py`--->`mmdetection/mmdet/models/anchor_heads/anchor_head.py`
- Add these files to the corresponding directories:
  - Add `AABO_mindspore/aabo_rpn_head.py` to `mmdetection/mmdet/models/anchor_heads/`
  - Add `AABO_mindspore/aabo_mask_rcnn_r101_fpn_2x.py` to `mmdetection/configs/`
  - Add `AABO_mindspore/aabo_htc_dcov_x101_64x4d_fpn_24e.py` to `mmdetection/configs/`
