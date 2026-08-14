# SOURCE_MAP · DETR

| Paper concept | DETRLite | Official facebookresearch/detr |
|---------------|----------|--------------------------------|
| Backbone | `TinyBackbone` | ResNet + position encoding |
| Transformer enc/dec | `nn.Transformer*` 1 layer | `models/transformer.py` |
| Object queries | `query_embed` | `query_embed` |
| Hungarian match | `assignment.hungarian_match` | `models/matcher.py` HungarianMatcher |
| Loss CE+L1(+GIoU) | `detr_lite_loss` | `models/detr.py` SetCriterion |
| no-object class | last logit | `num_classes` eos |

## Read order
1. DETR paper matching cost definition
2. `matcher.py` cost_class / cost_bbox / cost_giou
3. This script matching dump + train
