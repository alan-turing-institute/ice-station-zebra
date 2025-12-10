# Adding models

## Background

An `ice-station-zebra` model must handle multiple datasets with different dimensions. Inputs and outputs use `NTCHW` format, where:

- `N` is the batch size
- `T` is the number of history (forecast) steps for inputs (outputs)
- `C` is the number of channels or variables
- `H` and `W` are the height and width dimensions

`N` and `T` are consistent across inputs, while `C`, `H`, and `W` may vary. For example, with batch size `N=2`, 3 history steps, and 4 forecast steps, inputs have shape `(2, 3, C_k, H_k, W_k)` and the output has shape `(2, 4, C_out, H_out, W_out)`.

## Standalone models

A standalone model accepts a `dict[str, TensorNTCHW]` mapping dataset names to `NTCHW` tensors. It can use one or more inputs for training and must produce output of shape `N, T, C_out, H_out, W_out`.

![Standalone model pipeline](assets/pipeline-standalone.png)

Pros:
- all input variables are available without transformation

Cons:
- hard to add new inputs
- hard to add new outputs

## Processor models

A processor model is part of an encode–process–decode pipeline. Define a latent space `(C_latent, H_latent, W_latent)` (e.g., `(10, 64, 64)`). Encoders convert each input to `(N, T, C_latent, H_latent, W_latent)`. Encoded datasets combine to `(N, T, k * C_latent, H_latent, W_latent)`, which the processor consumes and emits in the same shape. Decoders transform this to `(N, T, C_out, H_out, W_out)`.

![Encode-process-decode pipeline](assets/pipeline-encode-process-decode.png)

Pros:
- easy to add new inputs
- easy to add new outputs

Cons:
- input variables are transformed into latent space

