# LiLT notebooks

LiLT (Language-independent Layout Transformer) is a nice model as it allows to plug-and-play any pre-trained RoBERTa model with a layout module,
allowing to have a LayoutLM-like model for any language.

To combine LiLT with any pre-trained RoBERTa model from the 🤗 hub, please check out [this notebook](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/LiLT/Create_LiLT_%2B_XLM_RoBERTa_base.ipynb).

Next, it can be fine-tuned on a custom data as shown in [this notebook](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/LiLT/Fine_tune_LiLT_on_a_custom_dataset%2C_in_any_language.ipynb).

There are 2 other notebooks in which I leverage LayoutLMv3Processor, but note that that's only possible because I fine-tune a checkpoint that uses
the same vocabulary as LayoutLMv3. So it's recommended to use [this notebook](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/LiLT/Fine_tune_LiLT_on_a_custom_dataset%2C_in_any_language.ipynb).
