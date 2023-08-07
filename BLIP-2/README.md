# BLIP-2 notebooks

This folder contains notebooks to illustate Salesforce's BLIP-2 model in 🤗 Transformers.

This folder contains notebooks for inference, but if you're interested in fine-tuning the model on custom data I recommend the following notebooks:

- https://github.com/huggingface/notebooks/blob/main/examples/image_captioning_blip.ipynb (full fine-tuning)
- https://github.com/huggingface/notebooks/blob/main/peft/Fine_tune_BLIP2_on_an_image_captioning_dataset_PEFT.ipynb (parameter efficient fine-tuning or PEFT)

One can either update all the parameters of the model (full fine-tuning), or leverage newer methods like LoRa (available in the 🤗 PEFT library) to freeze the weights of the pre-trained model and only train a couple of linear layers.

BLIP-2 can be used to caption images, answer questions related to images, or even chat in a conversational manner similar to Deepmind's [Flamingo](https://www.deepmind.com/blog/tackling-multiple-tasks-with-a-single-visual-language-model).

BLIP-2 even beats Flamingo on several benchmarks, despite being a lot smaller, leveraging open-source large language models like [OPT](https://huggingface.co/docs/transformers/model_doc/opt) and [Flan-T5](https://huggingface.co/docs/transformers/model_doc/flan-t5).
