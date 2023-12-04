# Table Transformer notebooks

This folder contains notebooks to illustrate inference with Table Transformer to detect tables in PDFs and perform table structure recognition.

Note that the Table Transformer is identical to the DETR object detection model, which means that fine-tuning Table Transformer on custom data
can be done as shown in the notebooks found in [this folder](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/DETR) - make sure to update the model and corresponding image processor.

The only difference is that the Table Transformer applies a "normalize before" operation, which means that layernorms are applied before,
rather than after MLPs/attention compared to DETR.

It's recommended to use the [last notebook](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Table%20Transformer/Inference_with_Table_Transformer_(TATR)_for_parsing_tables.ipynb) which leverages the right image processing settings.
