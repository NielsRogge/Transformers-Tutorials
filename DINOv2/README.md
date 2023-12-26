# DINOv2 notebooks

DINOv2 is a new Vision Transformer (ViT) by Meta AI trained in a self-supervised fashion on a highly curated dataset of 142 million images.

This folder contains demo notebooks to showcase how to fine-tune the model on custom data for [image classification](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/DINOv2/Fine_tune_DINOv2_for_image_classification_%5Bminimal%5D.ipynb) + [semantic segmentation](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/DINOv2/Train_a_linear_classifier_on_top_of_DINOv2_for_semantic_segmentation.ipynb).

Interested in doing depth estimation with DINOv2? See [this notebook](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/DPT/Inference_with_DPT_%2B_DINOv2_for_depth_estimation.ipynb), which adds a DPT head to a DINOv2 backbone. 

The DINOv2 docs can be found [here](https://huggingface.co/docs/transformers/main/model_doc/dinov2).

See also this thread for more info regarding using DINOv2: https://github.com/facebookresearch/dinov2/issues/153
