# LayoutLMv3 notebooks
In this directory, you can find notebooks that illustrate how to use LayoutLMv3 both for fine-tuning on custom data as well as inference. 

## Important note

LayoutLMv3 models are capable of getting > 90% F1 on FUNSD. This is thanks to the use of segment position embeddings, as opposed to word-level position embeddings, inspired by [StructuralLM](https://arxiv.org/abs/2105.11210). This means that words belonging to the same "segment" (let's say, an address) get the same bounding box coordinates, and thus the same 2D position embeddings. 

Most OCR engines (like Google's Tesseract) are able to identify segments as explained in [this thread](https://github.com/microsoft/unilm/issues/838) by the LayoutLMv3 author.

For the FUNSD dataset, segments were created based on the labels as seen [here](https://huggingface.co/datasets/nielsr/funsd-layoutlmv3/blob/main/funsd-layoutlmv3.py#L140).

It's always advised to use segment position embeddings over word-level position embeddings, as it gives quite a boost in performance.

## Training tips

Note that LayoutLMv3 is identical to LayoutLMv2 in terms of training/inference, except that:
* images need to be resized and normalized, such that they are `pixel_values` of shape `(batch_size, num_channels, heigth, width)`. The channels need to be in RGB format. This was not the case for LayoutLMv2, which expected the channels in BGR format (due to its Detectron2 visual backbone), and normalized the images internally.
* tokenization of text is based on RoBERTa, hence byte-level Byte-Pair-Encoding. This in contrast to LayoutLMv2, which used BERT-like WordPiece tokenization.

Because of this, I've created a new `LayoutLMv3Processor`, which combines a `LayoutLMv3ImageProcessor` (for the image modality) and a `LayoutLMv3TokenizerFast` (for the text modality) into one. Usage is identical to its predecessor [`LayoutLMv2Processor`](https://huggingface.co/docs/transformers/model_doc/layoutlmv2#usage-layoutlmv2processor).

The full documentation can be found [here](https://huggingface.co/transformers/model_doc/layoutlmv3.html).

The models on the hub can be found [here](https://huggingface.co/models?search=layoutlmv3).
