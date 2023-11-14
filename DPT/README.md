# DPT notebooks

DPT (Dense Prediction Transformer) is a model that can be used for dense prediction tasks (meaning, predicting things per pixel), such as depth estimation and semantic segmentation.

DPT was recently updated to leverage the `AutoBackbone` class, which enables it to leverage any `xxxBackbone` class in the Transformers library. This means that one can combine DPT
with a DINOv2 backbone, for instance:

```
from transformers import Dinov2Config, DPTConfig, DPTForDepthEstimation

backbone_config = Dinov2Config.from_pretrained("facebook/dinov2-base", out_features=["stage1", "stage2", "stage3", "stage4"])
config = DPTConfig(backbone_config=backbone_config)

model = DPTForDepthEstimation(config=config)
```

DPT checkpoints are on the hub, e.g.: https://huggingface.co/Intel/dpt-large

DPT checkpoints with a DINOv2 backbone are also on the hub: https://huggingface.co/models?pipeline_tag=depth-estimation&other=dinov2&sort=trending.
