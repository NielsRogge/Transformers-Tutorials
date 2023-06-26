# LayoutXLM notebooks

This folder contains notebooks illustrating how to fine-tune LayoutXLM on the [XFUND](https://github.com/doc-analysis/XFUND) dataset:

- fine-tuning `LayoutLMv2ForTokenClassification` for semantic entity recognition (also known as NER)
- fine-tuning `LayoutLMv2ForRelationExtraction` for key-value extraction

## Important note

Note that LayoutXLM is a multilingual model trained on a variety of languages, some of which may not be relevant for your use case. In that case, it makes sense to remove unnecessary tokens from the model's embedding matrx (as the embedding layer oftentimes makes up a large portion of a given model's total size).

Check out this blog post for details: https://medium.com/@coding-otter/reduce-your-transformers-model-size-by-removing-unwanted-tokens-and-word-embeddings-eec08166d2f9?.
