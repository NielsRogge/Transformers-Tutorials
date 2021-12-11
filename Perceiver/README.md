# Perceiver IO notebooks
In this directory, you can find several notebooks that illustrate how to use Deepmind's [Perceiver IO](https://arxiv.org/abs/2107.14795) both for fine-tuning on custom data as well as inference. They are based on the [official Colab notebooks](https://github.com/deepmind/deepmind-research/tree/master/perceiver/colabs) released by Deepmind, as well as some additional notebooks which I believe will be helpful for the community.

The notebooks which are available are:
- showcasing masked language modeling and image classification with the Perceiver [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/Perceiver/Perceiver_for_masked_language_modeling_and_image_classification.ipynb)
- fine-tuning the Perceiver for image classification [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/Perceiver/Fine_tune_the_Perceiver_for_image_classification.ipynb)
- fine-tuning the Perceiver for text classification [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/Perceiver/Fine_tune_Perceiver_for_text_classification.ipynb)
- predicting optical flow between a pair of images with `PerceiverForOpticalFlow`[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/Perceiver/Perceiver_for_Optical_Flow.ipynb)
- auto-encoding a video (images, audio, labels) with `PerceiverForMultimodalAutoencoding` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/Perceiver/Perceiver_for_Multimodal_Autoencoding.ipynb)

Note that these are just a few examples of what you can do with the Perceiver. There are many more possibilities with it, such as question-answering, named-entity recognition on text, object detection on images, audio classification,... Basically, anything you can do with BERT/ViT/Wav2Vec2/DETR/etc., you can do with the Perceiver too.

The [Perceiver](https://arxiv.org/abs/2103.03206) and its follow-up variant, [Perceiver IO](https://arxiv.org/abs/2107.14795) by Google Deepmind are one of my favorite works of 2021.

This model is quite elegant: it aims to solve the quadratic complexity of the self-attention mechanism by employing it on a (not-too large) set of latent variables, rather than on the inputs.
The inputs are only used for doing cross-attention with the latents. In that way, the inputs (which can be text, image, audio, video,...) don't have an impact on the memory and compute requirements of the self-attention operations.

In the Perceiver IO paper, the authors extend this to let the Perceiver also handle arbitrary outputs, next to arbitrary inputs. The idea is similar: one only employs the outputs for doing cross-attention with the latents.

The authors show that the model can achieve great results on a variety of modalities, including masked language modeling, image classification, optical flow, multimodal autoencoding and games.

The difference between the various models lies in their preprocessor, decoder and optional postprocessor. I've implemented all models that Deepmind [open-sourced](https://github.com/deepmind/deepmind-research/tree/master/perceiver) (originally written in JAX/Haiku) in PyTorch.
