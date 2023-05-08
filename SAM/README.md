# SAM notebooks

This folder contains demo notebooks regarding Meta AI's [SAM](https://huggingface.co/docs/transformers/main/en/model_doc/sam) (segment anything model):

- [fine-tuning SAM on custom data](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/SAM/Fine_tune_SAM_(segment_anything)_on_a_custom_dataset.ipynb)
- [performing inference with MedSAM](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/SAM/Run_inference_with_MedSAM_using_HuggingFace_Transformers.ipynb)

Check also the official Hugging Face notebooks for [general usage](https://github.com/huggingface/notebooks/blob/main/examples/segment_anything.ipynb) and [automatic mask generation](https://github.com/huggingface/notebooks/blob/main/examples/automatic_mask_generation.ipynb).

## ONNX export

Note that SAM can also be exported to ONNX using the 🤗 [Optimum library](https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/export_a_model): 

```
optimum-cli export onnx --model path_to_your_checkpoint sam_onnx/
```

ONNX is useful when putting the model in production, as one can apply several quantization techniques to speed up inference. Note that there are also 2 custom flags, `--point_batch_size` and `--nb_points_per_image`. See [this PR](https://github.com/huggingface/optimum/pull/1025) which added support for it.
