# UDOP notebooks

UDOP is an exciting document AI model from Microsoft Research. It has an encoder-decoder architecture, hence it's a generative model. 

It relies on an OCR engine, like LayoutLMv3. However, unlike LayoutLMv3, it has a text decoder, which means that the model generates answers one token at a time.
Hence we can train it to generate whatever sequences we want given document images, like generating a JSON of the key fields in the document, generating the class
of the document, or an answer regarding a question.

Docs: https://huggingface.co/docs/transformers/main/en/model_doc/udop

Checkpoints: https://huggingface.co/collections/microsoft/udop-65e625124aee97415b88b513
