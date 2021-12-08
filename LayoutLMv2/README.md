# LayoutLMv2 notebooks
In this directory, you can find several notebooks that illustrate how to use LayoutLMv2 both for fine-tuning on custom data as well as inference. I've split up the notebooks according to the different downstream datasets:

- CORD (form understanding)
- DocVQA (visual question answering on documents)
- FUNSD (form understanding)
- RVL-DIP (document image classification)

I've implemented LayoutLMv2 (and LayoutXLM) in the same way as other models in the Transformers library. You have:
- `LayoutLMv2ForSequenceClassification`, which you can use to classify document images (an example dataset is [RVL-CDIP](https://www.cs.cmu.edu/~aharley/rvl-cdip/)). This model adds a sequence classification head on top of the base `LayoutLMv2Model`, and returns `logits` of shape `(batch_size, num_labels)` (similar to `BertForSequenceClassification`).
- `LayoutLMv2ForTokenClassification`, which you can use to annotate words appearing in a document image (example datasets here are [CORD](https://github.com/clovaai/cord), [FUNSD](https://guillaumejaume.github.io/FUNSD/), [SROIE](https://rrc.cvc.uab.es/?ch=13), [Kleister-NDA](https://github.com/applicaai/kleister-nda)). This model adds a token classification head on top of the base `LayoutLMv2Model`, and treats form understanding as a sequence labeling/named-entity recognition (NER) problem. It returns `logits` of shape `(batch_size, sequence_length, num_labels)` (similar to `BertForTokenClassification`).
- `LayoutLMv2ForQuestionAnswering`, which you can use to perform extractive visual question answering on document images (an example dataset here is [DocVQA](https://docvqa.org/)). This model adds a question answering head on top of the base `LayoutLMv2Model`, and returns `start_logits` and `end_logits` (similar to `BertForQuestionAnswering`).

The full documentation (which also includes tips on how to use `LayoutLMv2Processor`) can be found [here](https://huggingface.co/transformers/model_doc/layoutlmv2.html).

The models on the hub can be found [here](https://huggingface.co/models?search=layoutlmv2).

Note that there's also a Gradio demo available for LayoutLMv2, hosted as a HuggingFace Space [here](https://huggingface.co/spaces/nielsr/LayoutLMv2-FUNSD).
