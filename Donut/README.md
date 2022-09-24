# Donut 🍩 notebooks
In this directory, you can find several notebooks that illustrate how to use Donut both for fine-tuning on custom data as well as inference. I've split up the notebooks according to the different downstream datasets:

- CORD (form understanding)
- DocVQA (visual question answering on documents)
- RVL-DIP (document image classification)

I've implemented Donut as an instance of [`VisionEncoderDecoderModel`](https://huggingface.co/docs/transformers/main/model_doc/vision-encoder-decoder) in the Transformers library.

The full documentation can be found [here](https://huggingface.co/transformers/main/model_doc/donut.html).

The models on the hub can be found [here](https://huggingface.co/models?search=donut).

Note that there's also several Gradio demos available for Donut, hosted as HuggingFace Spaces:
- [DocVQA](https://huggingface.co/spaces/nielsr/donut-docvqa)
- [RVLCDIP](https://huggingface.co/spaces/nielsr/donut-rvlcdip)
- [CORD](https://huggingface.co/spaces/nielsr/donut-cord)

## Third-party resources

Also check out this [great blog](https://www.philschmid.de/fine-tuning-donut) by Philipp Schmid on fine-tuning Donut on the SROIE dataset :)
