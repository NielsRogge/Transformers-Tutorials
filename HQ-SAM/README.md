# HQ-SAM notebook

HQ-SAM (or [SAM-HQ](https://huggingface.co/docs/transformers/main/en/model_doc/sam_hq)) does the same thing as Meta's Segment Anything (SAM), however it was fine-tuned on 44k high-quality segmentation masks. The authors used the pre-trained weights of SAM, added a couple of additional learnable parameters, and then fine-tuned the model altogether on those.

Hence, HQ-SAM can serve as a drop-in replacement to SAM.

This folder contains an inference notebook to show the various use cases of HQ-SAM (by simply copying the [SAM notebook](https://github.com/huggingface/notebooks/blob/main/examples/segment_anything.ipynb) and running it for HQ-SAM with some tiny tweaks).

For fine-tuning, see my [SAM notebook](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/SAM) (you can easily adapt it for HQ-SAM by just changing the classes and model names).