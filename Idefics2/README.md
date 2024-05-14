# Idefics2 notebooks

This folder contains notebooks regarding Idefics2, a powerful vision-language model developed by Hugging Face.

- Idefics2 [docs](https://huggingface.co/docs/transformers/main/en/model_doc/idefics2)
- Idefics2 [blog post](https://huggingface.co/blog/idefics2)
- see also this nice blog post: https://medium.com/google-developer-experts/ml-story-fine-tune-vision-language-model-on-custom-dataset-8e5f5dace7b1.

## Notes

1. I just uploaded a [similar notebook for LLaVa](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/LLaVa/Fine_tune_LLaVa_on_a_custom_dataset_(with_PyTorch_Lightning).ipynb): it works just as well, and I removed the addition of special tokens to make the logic simpler. Can be done for Idefics2, too.

2. The notebook I currently include here is aimed for extraction use cases (image->text or JSON).

If you have a chatbot use case, I'd recommend taking a look at the experimental support for VLMs in the [TRL](https://huggingface.co/docs/trl/en/index) library:
- example script for fine-tuning Llava for chat: https://github.com/huggingface/trl/blob/main/examples/scripts/vsft_llava.py
- example script for fine-tuning Idefics2 for chat: https://gist.github.com/edbeeching/228652fc6c2b29a1641be5a5778223cb
