# Table Transformer notebooks

This folder contains a notebook to illustrate inference with Table Transformer to detect tables in PDFs and perform table structure recognition.

Note that the Table Transformer is identical to the DETR object detection model, which means that fine-tuning Table Transformer on custom data
can be done as shown in the notebooks found in [this folder](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/DETR).

The only difference is that the Table Transformer applies a "normalize before" operation, which means that layernorms are applied before,
rather than after MLPs/attention.
