# TrOCR notebooks
In this directory, you can find several notebooks that illustrate how to use Microsoft's [TrOCR]() both for fine-tuning on custom data as well as inference. It currently includes the following notebooks:

- performing inference with TrOCR to illustrate optical character recognition with Transformers, as well as making a [Gradio](https://gradio.app/) demo
- fine-tuning TrOCR on the IAM dataset using HuggingFace's [Seq2SeqTrainer](https://huggingface.co/transformers/main_classes/trainer.html#seq2seqtrainer)
- fine-tuning TrOCR on the IAM dataset using native PyTorch

I also made a notebook that illustrates how to evaluate a TrOCR checkpoint in terms of CER (character-error rate) on the test set of the [IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database).

All models can be found on the [hub](https://huggingface.co/models?search=trocr).
