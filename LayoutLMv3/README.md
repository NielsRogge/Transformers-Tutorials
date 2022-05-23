# LayoutLMv3 notebooks
In this directory, you can find notebooks that illustrate how to use LayoutLMv3 both for fine-tuning on custom data as well as inference. 

Note that LayoutLMv3 is identical to LayoutLMv2 in terms of training/inference, except that:
* images need to be resized and normalized, such that they are `pixel_values` of shape `(batch_size, num_channels, heigth, width)`. The channels need to be in RGB format. This was not the case for LayoutLMv2, which expected the channels in BGR format (due to its Detectron2 visual backbone), and normalized the images internally.
* tokenization of text is based on RoBERTa, hence byte-level Byte-Pair-Encoding. This in contrast to LayoutLMv2, which used BERT-like WordPiece tokenization.

Because of this, I've created a new `LayoutLMv3Processor`, which combines a `LayoutLMv3FeatureExtractor` (for the image modality) and a `LayoutLMv3TokenizerFast (for the text modality) into one. Usage is identical to its predecessor [`LayoutLMv2Processor`](https://huggingface.co/docs/transformers/model_doc/layoutlmv2#usage-layoutlmv2processor).

The full documentation can be found [here](https://huggingface.co/transformers/model_doc/layoutlmv3.html).

The models on the hub can be found [here](https://huggingface.co/models?search=layoutlmv3).
