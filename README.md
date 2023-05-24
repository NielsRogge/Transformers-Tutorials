# Transformers-Tutorials

Hi there!

This repository contains demos I made with the [Transformers library](https://github.com/huggingface/transformers) by 🤗 HuggingFace. Currently, all of them are implemented in PyTorch.

NOTE: if you are not familiar with HuggingFace and/or Transformers, I highly recommend to check out our [free course](https://huggingface.co/course/chapter1), which introduces you to several Transformer architectures (such as BERT, GPT-2, T5, BART, etc.), as well as an overview of the HuggingFace libraries, including [Transformers](https://github.com/huggingface/transformers), [Tokenizers](https://github.com/huggingface/tokenizers), [Datasets](https://github.com/huggingface/datasets), [Accelerate](https://github.com/huggingface/accelerate) and the [hub](https://huggingface.co/).

For an overview of the ecosystem of HuggingFace for computer vision (June 2022), refer to [this notebook](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/HuggingFace_vision_ecosystem_overview_(June_2022).ipynb) with corresponding [video](https://www.youtube.com/watch?v=oL-xmufhZM8&t=2884s).

Currently, it contains the following demos:
* Audio Spectrogram Transformer ([paper](https://arxiv.org/abs/2104.01778)): 
  - performing inference with `ASTForAudioClassification` to classify audio. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/AST/Inference_with_the_Audio_Spectogram_Transformer_to_classify_audio.ipynb)
* BERT ([paper](https://arxiv.org/abs/1810.04805)): 
  - fine-tuning `BertForTokenClassification` on a named entity recognition (NER) dataset. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/BERT/Custom_Named_Entity_Recognition_with_BERT_only_first_wordpiece.ipynb)
  - fine-tuning `BertForSequenceClassification` for multi-label text classification. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb)
* BEiT ([paper]([https://arxiv.org/abs/2103.06874](https://arxiv.org/abs/2106.08254))):
  - understanding `BeitForMaskedImageModeling` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/BEiT/Understanding_BeitForMaskedImageModeling.ipynb)
* CANINE ([paper](https://arxiv.org/abs/2103.06874)):
  - fine-tuning `CanineForSequenceClassification` on IMDb [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/CANINE/Fine_tune_CANINE_on_IMDb_(movie_review_binary_classification).ipynb)
* CLIPSeg ([paper](https://arxiv.org/abs/2112.10003)):
  - performing zero-shot image segmentation with `CLIPSeg` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/CLIPSeg/Zero_shot_image_segmentation_with_CLIPSeg.ipynb)
* Conditional DETR ([paper](https://arxiv.org/abs/2108.06152)):
  - performing inference with `ConditionalDetrForObjectDetection` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/Conditional%20DETR/Run_inference_with_Conditional_DETR.ipynb)
  - fine-tuning `ConditionalDetrForObjectDetection` on a custom dataset (balloon) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/Conditional%20DETR/Fine_tuning_Conditional_DETR_on_custom_dataset_(balloon).ipynb)
* ConvNeXT ([paper](https://arxiv.org/abs/2201.03545)):
  - fine-tuning (and performing inference with) `ConvNextForImageClassification` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/ConvNeXT/Fine_tune_ConvNeXT_for_image_classification.ipynb)
* DINO ([paper](https://arxiv.org/abs/2104.14294)):
  - visualize self-attention of Vision Transformers trained using the DINO method [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/DINO/Visualize_self_attention_of_DINO.ipynb)
* DETR ([paper](https://arxiv.org/abs/2005.12872)):
  - performing inference with `DetrForObjectDetection` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/DETR/DETR_minimal_example_(with_DetrFeatureExtractor).ipynb)
  - fine-tuning `DetrForObjectDetection` on a custom object detection dataset [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/DETR/Fine_tuning_DetrForObjectDetection_on_custom_dataset_(balloon).ipynb)
  - evaluating `DetrForObjectDetection` on the COCO detection 2017 validation set [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/DETR/Evaluating_DETR_on_COCO_validation_2017.ipynb)
  - performing inference with `DetrForSegmentation` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/DETR/DETR_panoptic_segmentation_minimal_example_(with_DetrFeatureExtractor).ipynb)
  - fine-tuning `DetrForSegmentation` on COCO panoptic 2017 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/DETR/Fine_tuning_DetrForSegmentation_on_custom_dataset_end_to_end_approach.ipynb)
* DPT ([paper](https://arxiv.org/abs/2103.13413)):
  - performing inference with DPT for monocular depth estimation [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/DPT/DPT_inference_notebook_(depth_estimation).ipynb)
  - performing inference with DPT for semantic segmentation [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/DPT/DPT_inference_notebook_(semantic_segmentation).ipynb)
* Deformable DETR ([paper](https://arxiv.org/abs/2010.04159)):
  - performing inference with `DeformableDetrForObjectDetection` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/Deformable-DETR/Inference_with_Deformable_DETR_(CPU).ipynb)
* DiT ([paper](https://arxiv.org/abs/2203.02378)):
  - performing inference with DiT for document image classification [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/DiT/Inference_with_DiT_(Document_Image_Transformer)_for_document_image_classification.ipynb)
* Donut ([paper](https://arxiv.org/abs/2111.15664)):
  - performing inference with Donut for document image classification [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/Donut/RVL-CDIP/Quick_inference_with_DONUT_for_Document_Image_Classification.ipynb)
  - fine-tuning Donut for document image classification [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/Donut/RVL-CDIP/Fine_tune_Donut_on_toy_RVL_CDIP_(document_image_classification).ipynb)
  - performing inference with Donut for document visual question answering (DocVQA) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/Donut/DocVQA/Quick_inference_with_DONUT_for_DocVQA.ipynb)
  - performing inference with Donut for document parsing [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/Donut/CORD/Quick_inference_with_DONUT_for_Document_Parsing.ipynb)
  - fine-tuning Donut for document parsing with PyTorch Lightning [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/Donut/CORD/Fine_tune_Donut_on_a_custom_dataset_(CORD)_with_PyTorch_Lightning.ipynb)
* GIT ([paper](https://arxiv.org/abs/2205.14100)):
  - performing inference with GIT for image/video captioning and image/video question-answering [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/GIT/Inference_with_GIT_for_image_video_captioning_and_image_video_QA.ipynb)
  - fine-tuning GIT on a custom image captioning dataset [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/GIT/Fine_tune_GIT_on_an_image_captioning_dataset.ipynb)
* GLPN ([paper](https://arxiv.org/abs/2201.07436)):
  - performing inference with `GLPNForDepthEstimation` to illustrate monocular depth estimation [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/GLPN/GLPN_inference_(depth_estimation).ipynb)
* GPT-J-6B ([repository](https://github.com/kingoflolz/mesh-transformer-jax)):
  - performing inference with `GPTJForCausalLM` to illustrate few-shot learning and code generation [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/GPT-J-6B/Inference_with_GPT_J_6B.ipynb)
* GroupViT ([repository](https://github.com/NVlabs/GroupViT)):
  - performing inference with `GroupViTModel` to illustrate zero-shot semantic segmentation [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/GroupViT/Inference_with_GroupViT_for_zero_shot_semantic_segmentation.ipynb)
* ImageGPT ([blog post](https://openai.com/blog/image-gpt/)):
  - (un)conditional image generation with `ImageGPTForCausalLM` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/ImageGPT/(Un)conditional_image_generation_with_ImageGPT.ipynb)
  - linear probing with ImageGPT [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/ImageGPT/Linear_probing_with_ImageGPT.ipynb)
* LUKE ([paper](https://arxiv.org/abs/2010.01057)):
  - fine-tuning `LukeForEntityPairClassification` on a custom relation extraction dataset using PyTorch Lightning [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LUKE/Supervised_relation_extraction_with_LukeForEntityPairClassification.ipynb)
* LayoutLM ([paper](https://arxiv.org/abs/1912.13318)): 
  - fine-tuning `LayoutLMForTokenClassification` on the [FUNSD](https://guillaumejaume.github.io/FUNSD/) dataset [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLM/Fine_tuning_LayoutLMForTokenClassification_on_FUNSD.ipynb)
  - fine-tuning `LayoutLMForSequenceClassification` on the [RVL-CDIP](https://www.cs.cmu.edu/~aharley/rvl-cdip/) dataset [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLM/Fine_tuning_LayoutLMForSequenceClassification_on_RVL_CDIP.ipynb)
  - adding image embeddings to LayoutLM during fine-tuning on the [FUNSD](https://guillaumejaume.github.io/FUNSD/) dataset [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLM/Add_image_embeddings_to_LayoutLM.ipynb)
* LayoutLMv2 ([paper](https://arxiv.org/abs/2012.14740)):
  - fine-tuning `LayoutLMv2ForSequenceClassification` on RVL-CDIP [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv2/RVL-CDIP/Fine_tuning_LayoutLMv2ForSequenceClassification_on_RVL_CDIP.ipynb)
  - fine-tuning `LayoutLMv2ForTokenClassification` on FUNSD [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv2/FUNSD/Fine_tuning_LayoutLMv2ForTokenClassification_on_FUNSD.ipynb)
  - fine-tuning `LayoutLMv2ForTokenClassification` on FUNSD using the 🤗 Trainer [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv2/FUNSD/Fine_tuning_LayoutLMv2ForTokenClassification_on_FUNSD_using_HuggingFace_Trainer.ipynb)
  - performing inference with `LayoutLMv2ForTokenClassification` on FUNSD [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv2/FUNSD/Inference_with_LayoutLMv2ForTokenClassification.ipynb)
  - true inference with `LayoutLMv2ForTokenClassification` (when no labels are available) + Gradio demo [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv2/FUNSD/True_inference_with_LayoutLMv2ForTokenClassification_%2B_Gradio_demo.ipynb)
  - fine-tuning `LayoutLMv2ForTokenClassification` on CORD [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv2/CORD/Fine_tuning_LayoutLMv2ForTokenClassification_on_CORD.ipynb)
  - fine-tuning `LayoutLMv2ForQuestionAnswering` on DOCVQA [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv2/DocVQA/Fine_tuning_LayoutLMv2ForQuestionAnswering_on_DocVQA.ipynb)
* LayoutLMv3 ([paper](https://arxiv.org/abs/2204.08387)): 
  - fine-tuning `LayoutLMv3ForTokenClassification` on the [FUNSD](https://guillaumejaume.github.io/FUNSD/) dataset [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv3/Fine_tune_LayoutLMv3_on_FUNSD_(HuggingFace_Trainer).ipynb)
* LayoutXLM ([paper](https://arxiv.org/abs/2104.08836)): 
  - fine-tuning LayoutXLM on the [XFUND](https://github.com/doc-analysis/XFUND) benchmark for token classification [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutXLM/Fine_tuning_LayoutXLM_on_XFUND_for_token_classification_using_HuggingFace_Trainer.ipynb)
  - fine-tuning LayoutXLM on the [XFUND](https://github.com/doc-analysis/XFUND) benchmark for relation extraction [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutXLM/Fine_tune_LayoutXLM_on_XFUND_(relation_extraction).ipynb)
* MarkupLM ([paper](https://arxiv.org/abs/2110.08518)):
  - inference with MarkupLM to perform question answering on web pages [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/MarkupLM/Inference_with_MarkupLM_for_question_answering_on_web_pages.ipynb)
  - fine-tuning `MarkupLMForTokenClassification` on a toy dataset for NER on web pages [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/MarkupLM/Fine_tune_MarkupLMForTokenClassification_on_a_custom_dataset.ipynb)
* Mask2Former ([paper](https://arxiv.org/abs/2112.01527)):
  - performing inference with `Mask2Former` for universal image segmentation: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/Mask2Former/Inference_with_Mask2Former.ipynb)
* MaskFormer ([paper](https://arxiv.org/abs/2107.06278)):
  - performing inference with `MaskFormer` (both semantic and panoptic segmentation): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/MaskFormer/maskformer_minimal_example(with_MaskFormerFeatureExtractor).ipynb)
  - fine-tuning `MaskFormer` on a custom dataset for semantic segmentation [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/MaskFormer/Fine_tune_MaskFormer_on_custom_dataset.ipynb)
* OneFormer ([paper](https://arxiv.org/abs/2211.06220)):
  - performing inference with `OneFormer` for universal image segmentation: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/OneFormer/Inference_with_OneFormer.ipynb)
* Perceiver IO ([paper](https://arxiv.org/abs/2107.14795)):
  - showcasing masked language modeling and image classification with the Perceiver [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/Perceiver/Perceiver_for_masked_language_modeling_and_image_classification.ipynb)
  - fine-tuning the Perceiver for image classification [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/Perceiver/Fine_tune_the_Perceiver_for_image_classification.ipynb)
  - fine-tuning the Perceiver for text classification [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/Perceiver/Fine_tune_Perceiver_for_text_classification.ipynb)
  - predicting optical flow between a pair of images with `PerceiverForOpticalFlow`[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/Perceiver/Perceiver_for_Optical_Flow.ipynb)
  - auto-encoding a video (images, audio, labels) with `PerceiverForMultimodalAutoencoding` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/Perceiver/Perceiver_for_Multimodal_Autoencoding.ipynb)
* SAM ([paper]([https://arxiv.org/abs/2105.15203](https://arxiv.org/abs/2304.02643))):
  - performing inference with MedSAM [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/SAM/Run_inference_with_MedSAM_using_HuggingFace_Transformers.ipynb)
  - fine-tuning `SamModel` on a custom dataset [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/SAM/Fine_tune_SAM_(segment_anything)_on_a_custom_dataset.ipynb)
* SegFormer ([paper](https://arxiv.org/abs/2105.15203)):
  - performing inference with `SegformerForSemanticSegmentation` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/SegFormer/Segformer_inference_notebook.ipynb)
  - fine-tuning `SegformerForSemanticSegmentation` on custom data using native PyTorch [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/SegFormer/Fine_tune_SegFormer_on_custom_dataset.ipynb)
* T5 ([paper](https://arxiv.org/abs/1910.10683)):
  - fine-tuning `T5ForConditionalGeneration` on a Dutch summarization dataset on TPU using HuggingFace Accelerate [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/tree/master/T5)
  - fine-tuning `T5ForConditionalGeneration` (CodeT5) for Ruby code summarization using PyTorch Lightning [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/T5/Fine_tune_CodeT5_for_generating_docstrings_from_Ruby_code.ipynb)
* TAPAS ([paper](https://arxiv.org/abs/2004.02349)):  
  - fine-tuning `TapasForQuestionAnswering` on the Microsoft [Sequential Question Answering (SQA)](https://www.microsoft.com/en-us/download/details.aspx?id=54253) dataset [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/TAPAS/Fine_tuning_TapasForQuestionAnswering_on_SQA.ipynb)
  - evaluating `TapasForSequenceClassification` on the [Table Fact Checking (TabFact)](https://tabfact.github.io/) dataset [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/TAPAS/Evaluating_TAPAS_on_the_Tabfact_test_set.ipynb)
* Table Transformer ([paper](https://arxiv.org/abs/2110.00061)):
  - using the Table Transformer for table detection and table structure recognition [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/Table%20Transformer/Using_Table_Transformer_for_table_detection_and_table_structure_recognition.ipynb)
* TrOCR ([paper](https://arxiv.org/abs/2109.10282)):
  - performing inference with `TrOCR` to illustrate optical character recognition with Transformers, as well as making a Gradio demo [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/TrOCR/Inference_with_TrOCR_%2B_Gradio_demo.ipynb)
  - fine-tuning `TrOCR` on the IAM dataset using the Seq2SeqTrainer [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/TrOCR/Fine_tune_TrOCR_on_IAM_Handwriting_Database_using_Seq2SeqTrainer.ipynb)
  - fine-tuning `TrOCR` on the IAM dataset using native PyTorch [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/TrOCR/Fine_tune_TrOCR_on_IAM_Handwriting_Database_using_native_PyTorch.ipynb)
  - evaluating `TrOCR` on the IAM test set [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/TrOCR/Evaluating_TrOCR_base_handwritten_on_the_IAM_test_set.ipynb)
* UPerNet ([paper](https://arxiv.org/abs/1807.10221)):
  - performing inference with `UperNetForSemanticSegmentation` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/UPerNet/Perform_inference_with_UperNetForSemanticSegmentation_(Swin_backbone).ipynb)
* VideoMAE ([paper](https://arxiv.org/abs/2203.12602)):
  - performing inference with `VideoMAEForVideoClassification` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/VideoMAE/Quick_inference_with_VideoMAE.ipynb)
* ViLT ([paper](https://arxiv.org/abs/2102.03334)):
  - fine-tuning `ViLT` for visual question answering (VQA) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/ViLT/Fine_tuning_ViLT_for_VQA.ipynb)
  - performing inference with `ViLT` to illustrate visual question answering (VQA) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/ViLT/Inference_with_ViLT_(visual_question_answering).ipynb)
  - masked language modeling (MLM) with a pre-trained `ViLT` model [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/ViLT/Masked_language_modeling_with_ViLT.ipynb)
  - performing inference with `ViLT` for image-text retrieval [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/ViLT/Using_ViLT_for_image_text_retrieval.ipynb)
  - performing inference with `ViLT` to illustrate natural language for visual reasoning (NLVR) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/ViLT/ViLT_for_natural_language_visual_reasoning.ipynb)
* ViTMAE ([paper](https://arxiv.org/abs/2111.06377)):
  - reconstructing pixel values with `ViTMAEForPreTraining` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/ViTMAE/ViT_MAE_visualization_demo.ipynb)
* Vision Transformer ([paper](https://arxiv.org/abs/2010.11929)):
  - performing inference with `ViTForImageClassification` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/VisionTransformer/Quick_demo_of_HuggingFace_version_of_Vision_Transformer_inference.ipynb)
  - fine-tuning `ViTForImageClassification` on [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) using PyTorch Lightning [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/VisionTransformer/Fine_tuning_the_Vision_Transformer_on_CIFAR_10_with_PyTorch_Lightning.ipynb)
  - fine-tuning `ViTForImageClassification` on [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) using the 🤗 Trainer [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/VisionTransformer/Fine_tuning_the_Vision_Transformer_on_CIFAR_10_with_the_%F0%9F%A4%97_Trainer.ipynb)
* X-CLIP ([paper](https://arxiv.org/abs/2208.02816)):
  - performing zero-shot video classification with X-CLIP [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/X-CLIP/Video_text_matching_with_X_CLIP.ipynb)
  - zero-shot classifying a YouTube video with X-CLIP [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/X-CLIP/Zero_shot_classify_a_YouTube_video_with_X_CLIP.ipynb)
* YOLOS ([paper](https://arxiv.org/abs/2106.00666)):
  - fine-tuning `YolosForObjectDetection` on a custom dataset [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/YOLOS/Fine_tuning_YOLOS_for_object_detection_on_custom_dataset_(balloon).ipynb)
  - inference with `YolosForObjectDetection` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/YOLOS/YOLOS_minimal_inference_example.ipynb)

... more to come! 🤗 

If you have any questions regarding these demos, feel free to open an issue on this repository.

Btw, I was also the main contributor to add the following algorithms to the library:
- TAbular PArSing (TAPAS) by Google AI
- Vision Transformer (ViT) by Google AI
- DINO by Facebook AI
- Data-efficient Image Transformers (DeiT) by Facebook AI
- LUKE by Studio Ousia
- DEtection TRansformers (DETR) by Facebook AI
- CANINE by Google AI
- BEiT by Microsoft Research 
- LayoutLMv2 (and LayoutXLM) by Microsoft Research
- TrOCR by Microsoft Research 
- SegFormer by NVIDIA
- ImageGPT by OpenAI
- Perceiver by Deepmind
- MAE by Facebook AI
- ViLT by NAVER AI Lab
- ConvNeXT by Facebook AI
- DiT By Microsoft Research
- GLPN by KAIST
- DPT by Intel Labs
- YOLOS by School of EIC, Huazhong University of Science & Technology
- TAPEX by Microsoft Research
- LayoutLMv3 by Microsoft Research
- VideoMAE by Multimedia Computing Group, Nanjing University
- X-CLIP by Microsoft Research
- MarkupLM by Microsoft Research

All of them were an incredible learning experience. I can recommend anyone to contribute an AI algorithm to the library!

## Data preprocessing
Regarding preparing your data for a PyTorch model, there are a few options:
- a native PyTorch dataset + dataloader. This is the standard way to prepare data for a PyTorch model, namely by subclassing `torch.utils.data.Dataset`, and then creating a corresponding `DataLoader` (which is a Python generator that allows to loop over the items of a dataset). When subclassing the `Dataset` class, one needs to implement 3 methods: `__init__`, `__len__` (which returns the number of examples of the dataset) and `__getitem__` (which returns an example of the dataset, given an integer index). Here's an example of creating a basic text classification dataset (assuming one has a CSV that contains 2 columns, namely "text" and "label"):

```python
from torch.utils.data import Dataset

class CustomTrainDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get item
        item = df.iloc[idx]
        text = item['text']
        label = item['label']
        # encode text
        encoding = self.tokenizer(text, padding="max_length", max_length=128, truncation=True, return_tensors="pt")
        # remove batch dimension which the tokenizer automatically adds
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        # add label
        encoding["label"] = torch.tensor(label)
        
        return encoding
```

Instantiating the dataset then happens as follows:

```python
from transformers import BertTokenizer
import pandas as pd

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
df = pd.read_csv("path_to_your_csv")

train_dataset = CustomTrainDataset(df=df, tokenizer=tokenizer)
```

Accessing the first example of the dataset can then be done as follows:

```python
encoding = train_dataset[0]
```

In practice, one creates a corresponding `DataLoader`, that allows to get batches from the dataset:

```python
from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
```
I often check whether the data is created correctly by fetching the first batch from the data loader, and then printing out the shapes of the tensors, decoding the input_ids back to text, etc.

```python
batch = next(iter(train_dataloader))
for k,v in batch.items():
    print(k, v.shape)
# decode the input_ids of the first example of the batch
print(tokenizer.decode(batch['input_ids'][0].tolist())
```
-  [HuggingFace Datasets](https://huggingface.co/docs/datasets/). Datasets is a library by HuggingFace that allows to easily load and process data in a very fast and memory-efficient way. It is backed by [Apache Arrow](https://arrow.apache.org/), and has cool features such as memory-mapping, which allow you to only load data into RAM when it is required. It only has deep interoperability with the [HuggingFace hub](https://huggingface.co/datasets), allowing to easily load well-known datasets as well as share your own with the community.

Loading a custom dataset as a Dataset object can be done as follows (you can install datasets using `pip install datasets`):
```python
from datasets import load_dataset

dataset = load_dataset('csv', data_files={'train': ['my_train_file_1.csv', 'my_train_file_2.csv'] 'test': 'my_test_file.csv'})
```
Here I'm loading local CSV files, but there are other formats supported (including JSON, Parquet, txt) as well as loading data from a local Pandas dataframe or dictionary for instance. You can check out the [docs](https://huggingface.co/docs/datasets/loading.html#local-and-remote-files) for all details.

## Training frameworks
Regarding fine-tuning Transformer models (or more generally, PyTorch models), there are a few options:
- using native PyTorch. This is the most basic way to train a model, and requires the user to manually write the training loop. The advantage is that this is very easy to debug. The disadvantage is that one needs to implement training him/herself, such as setting the model in the appropriate mode (`model.train()`/`model.eval()`), handle device placement (`model.to(device)`), etc. A typical training loop in PyTorch looks as follows (inspired by [this great PyTorch intro tutorial]()):

```python
import torch
from transformers import BertForSequenceClassification

# Instantiate pre-trained BERT model with randomly initialized classification head
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

# I almost always use a learning rate of 5e-5 when fine-tuning Transformer based models
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

# put model on GPU, if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for batch in train_dataloader:
        # put batch on device
        batch = {k:v.to(device) for k,v in batch.items()}
        
        # forward pass
        outputs = model(**batch)
        loss = outputs.loss
        
        train_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print("Loss after epoch {epoch}:", train_loss/len(train_dataloader))
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in eval_dataloader:
            # put batch on device
            batch = {k:v.to(device) for k,v in batch.items()}
            
            # forward pass
            outputs = model(**batch)
            loss = outputs.logits
            
            val_loss += loss.item()
                  
    print("Validation loss after epoch {epoch}:", val_loss/len(eval_dataloader))
```

- [PyTorch Lightning (PL)](https://www.pytorchlightning.ai/). PyTorch Lightning is a framework that automates the training loop written above, by abstracting it away in a Trainer object. Users don't need to write the training loop themselves anymore, instead they can just do `trainer = Trainer()` and then `trainer.fit(model)`. The advantage is that you can start training models very quickly (hence the name lightning), as all training-related code is handled by the `Trainer` object. The disadvantage is that it may be more difficult to debug your model, as the training and evaluation is now abstracted away.
- [HuggingFace Trainer](https://huggingface.co/transformers/main_classes/trainer.html). The HuggingFace Trainer API can be seen as a framework similar to PyTorch Lightning in the sense that it also abstracts the training away using a Trainer object. However, contrary to PyTorch Lightning, it is not meant not be a general framework. Rather, it is made especially for fine-tuning Transformer-based models available in the HuggingFace Transformers library. The Trainer also has an extension called `Seq2SeqTrainer` for encoder-decoder models, such as BART, T5 and the `EncoderDecoderModel` classes. Note that all [PyTorch example scripts](https://github.com/huggingface/transformers/tree/master/examples/pytorch) of the Transformers library make use of the Trainer.
- [HuggingFace Accelerate](https://github.com/huggingface/accelerate): Accelerate is a new project, that is made for people who still want to write their own training loop (as shown above), but would like to make it work automatically irregardless of the hardware (i.e. multiple GPUs, TPU pods, mixed precision, etc.).
