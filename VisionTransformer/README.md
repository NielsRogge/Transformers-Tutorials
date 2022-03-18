# Vision Transformer (ViT) 

## Notebooks
This directory contains several notebooks that illustrate how to use Google's [ViT](https://huggingface.co/docs/transformers/model_doc/vit) both for fine-tuning on custom data as well as inference. It currently includes the following notebooks:

- performing inference with ViT to illustrate image classification
- fine-tuning ViT on CIFAR-10 using HuggingFace's [Trainer](https://huggingface.co/transformers/main_classes/trainer.html)
- fine-tuning ViT on CIFAR-10 using [PyTorch Lightning](https://www.pytorchlightning.ai/)

There's also the official HuggingFace image classification notebook, which can be found [here](
https://github.com/huggingface/notebooks/blob/master/examples/image_classification.ipynb).

Note that these notebooks work for any vision model in the library (i.e. any model supported by the `AutoModelForImageClassification` API). You can just replace the checkpoint name
(like `google/vit-base-patch16-224`) by another one (like `facebook/convnext-tiny-224`)

Just pick your favorite vision model from the [hub](https://huggingface.co/models?other=vision) and start fine-tuning it :)

## Blog posts

Below, I list some great blog posts explaining how to use ViT:

PyTorch:
- [Fine-Tune ViT for Image Classification with 🤗 Transformers](https://huggingface.co/blog/fine-tune-vit)
- [A complete Hugging Face tutorial: how to build and train a vision transformer](https://theaisummer.com/hugging-face-vit/)
- [How to Train the Hugging Face Vision Transformer On a Custom Dataset](https://blog.roboflow.com/how-to-train-vision-transformer/)

Tensorflow/Keras:
- [Image Classification with Hugging Face Transformers and `Keras`](https://www.philschmid.de/image-classification-huggingface-transformers-keras)
