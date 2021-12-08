The [Perceiver](https://arxiv.org/abs/2103.03206) and its follow-up variant, [Perceiver IO](https://arxiv.org/abs/2107.14795) by Google Deepmind are one of my favorite works of 2021.

This model is quite elegant: it aims to solve the quadratic complexity of the self-attention mechanism by employing it on a (not-too large) set of latent variables, rather than on the inputs.
The inputs are only used for doing cross-attention with the latents. In that way, the inputs (which can be text, image, audio, video,...) don't have an impact on the memory and compute requirements of the self-attention operations.

In the Perceiver IO paper, the authors extend this to let the Perceiver also handle arbitrary outputs, next to arbitrary inputs. The idea is similar: one only employs the outputs for doing cross-attention with the latents.

The authors show that the model can achieve great results on a variety of modalities, including masked language modeling, image classification, optical flow, multimodal autoencoding and games.

The difference between the various models lies in their preprocessor, decoder and optional postprocessor. I've implemented all models that Deepmind [open-sourced](https://github.com/deepmind/deepmind-research/tree/master/perceiver) (originally written in JAX/Haiku) in PyTorch.
