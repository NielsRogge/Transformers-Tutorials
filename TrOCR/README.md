# TrOCR notebooks
In this directory, you can find several notebooks that illustrate how to use Microsoft's [TrOCR](https://arxiv.org/abs/2109.10282) both for fine-tuning on custom data as well as inference. It currently includes the following notebooks:

- performing inference with TrOCR to illustrate optical character recognition with Transformers, as well as making a [Gradio](https://gradio.app/) demo
- fine-tuning TrOCR on the IAM dataset using HuggingFace's [Seq2SeqTrainer](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainer)
- fine-tuning TrOCR on the IAM dataset using native PyTorch

I also made a notebook that illustrates how to evaluate a TrOCR checkpoint in terms of CER (character-error rate) on the test set of the [IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database).

All models can be found on the [hub](https://huggingface.co/models?search=trocr).

Note that there's also a Gradio demo available for TrOCR, hosted as a HuggingFace Space [here](https://huggingface.co/spaces/nielsr/TrOCR-handwritten).
