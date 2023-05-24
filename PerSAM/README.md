# PerSAM notebooks

PerSAM is a really cool method to personalize the [SAM](https://huggingface.co/docs/transformers/main/model_doc/sam) model by Meta AI.

Using just one example (image, mask) pair of a given concept (like "dog"), the method allows to make sure the SAM model is able to segment the concept in other images.

I've made 2 notebooks to illustrate the 2 approaches as proposed in the [original paper](https://arxiv.org/abs/2305.03048).

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/persam_overview.jpg"
alt="drawing" width="600"/>
