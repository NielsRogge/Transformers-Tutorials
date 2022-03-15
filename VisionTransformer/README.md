# Vision Transformer (ViT) notebooks
In this directory, you can find several notebooks that illustrate how to use Google's [ViT](https://huggingface.co/docs/transformers/model_doc/vit) both for fine-tuning on custom data as well as inference. It currently includes the following notebooks:

- performing inference with ViT to illustrate image classification
- fine-tuning ViT on CIFAR-10 using HuggingFace's [Trainer](https://huggingface.co/transformers/main_classes/trainer.html)
- fine-tuning ViT on CIFAR-10 using [PyTorch Lightning](https://www.pytorchlightning.ai/)

Note that these notebooks work for any vision model in the library (i.e. any model supported by the `AutoModelForImageClassification` API). You can just replace the checkpoint name
(like `google/vit-base-patch16-224`) by another one (like `facebook/convnext-tiny-224`)

Just pick your favorite vision model from the [hub](https://huggingface.co/models?other=vision) and start fine-tuning it :)
