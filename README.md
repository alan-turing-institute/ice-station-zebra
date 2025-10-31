# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/alan-turing-institute/ice-station-zebra/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                                    |    Stmts |     Miss |   Cover |   Missing |
|------------------------------------------------------------------------ | -------: | -------: | ------: | --------: |
| ice\_station\_zebra/\_\_init\_\_.py                                     |        0 |        0 |    100% |           |
| ice\_station\_zebra/callbacks/\_\_init\_\_.py                           |        5 |        0 |    100% |           |
| ice\_station\_zebra/callbacks/ema\_weight\_averaging\_callback.py       |       13 |        8 |     38% |24-28, 34-40 |
| ice\_station\_zebra/callbacks/metadata.py                               |      243 |       25 |     90% |62, 66, 71, 89, 94, 112-113, 145-151, 192, 196-200, 228-229, 275, 328, 349, 436-437, 504, 506 |
| ice\_station\_zebra/callbacks/metric\_summary\_callback.py              |       28 |       15 |     46% |25-27, 39-45, 54-61 |
| ice\_station\_zebra/callbacks/plotting\_callback.py                     |      142 |       90 |     37% |71, 81, 100-101, 103-105, 113-117, 135-182, 191-211, 220-249, 272-287, 304-312, 323-337 |
| ice\_station\_zebra/callbacks/unconditional\_checkpoint.py              |       21 |       10 |     52% |17-19, 24, 29-30, 34-35, 39-40 |
| ice\_station\_zebra/callbacks/weight\_averaging.py                      |       95 |       60 |     37% |96-112, 133, 149-161, 191-197, 212-217, 232-233, 248-249, 264-265, 277, 289, 310-326, 349-376, 387-398, 407-416 |
| ice\_station\_zebra/cli/\_\_init\_\_.py                                 |        3 |        0 |    100% |           |
| ice\_station\_zebra/cli/hydra.py                                        |       22 |        3 |     86% |     39-41 |
| ice\_station\_zebra/cli/main.py                                         |       12 |        1 |     92% |        23 |
| ice\_station\_zebra/config/\_\_init\_\_.py                              |        0 |        0 |    100% |           |
| ice\_station\_zebra/data\_loaders/\_\_init\_\_.py                       |        3 |        0 |    100% |           |
| ice\_station\_zebra/data\_loaders/combined\_dataset.py                  |       44 |       27 |     39% |27-45, 64, 76, 87-88, 92, 99, 106-110, 115-119 |
| ice\_station\_zebra/data\_loaders/zebra\_data\_module.py                |       59 |       37 |     37% |25-72, 85, 93, 101-103, 109-128, 134-153, 159-178, 184-203 |
| ice\_station\_zebra/data\_loaders/zebra\_dataset.py                     |       78 |        0 |    100% |           |
| ice\_station\_zebra/data\_processors/\_\_init\_\_.py                    |        2 |        0 |    100% |           |
| ice\_station\_zebra/data\_processors/cli.py                             |       25 |       10 |     60% |21-25, 32-35, 39 |
| ice\_station\_zebra/data\_processors/filters/\_\_init\_\_.py            |        5 |        1 |     80% |         8 |
| ice\_station\_zebra/data\_processors/filters/doubling\_filter.py        |       11 |        4 |     64% |16-17, 21-22 |
| ice\_station\_zebra/data\_processors/preprocessors/\_\_init\_\_.py      |        4 |        0 |    100% |           |
| ice\_station\_zebra/data\_processors/preprocessors/icenet\_sic.py       |       63 |       46 |     27% |20-26, 30, 35-66, 74-131 |
| ice\_station\_zebra/data\_processors/preprocessors/ipreprocessor.py     |        9 |        2 |     78% |     10-11 |
| ice\_station\_zebra/data\_processors/preprocessors/null.py              |        5 |        1 |     80% |         9 |
| ice\_station\_zebra/data\_processors/zebra\_data\_processor.py          |       33 |       18 |     45% |25-31, 35-45, 49-51, 60-61 |
| ice\_station\_zebra/data\_processors/zebra\_data\_processor\_factory.py |       11 |        4 |     64% |     19-26 |
| ice\_station\_zebra/evaluation/\_\_init\_\_.py                          |        2 |        0 |    100% |           |
| ice\_station\_zebra/evaluation/cli.py                                   |       16 |        3 |     81% | 27-28, 32 |
| ice\_station\_zebra/evaluation/evaluator.py                             |       38 |       26 |     32% | 26-84, 89 |
| ice\_station\_zebra/exceptions.py                                       |        3 |        0 |    100% |           |
| ice\_station\_zebra/models/\_\_init\_\_.py                              |        4 |        0 |    100% |           |
| ice\_station\_zebra/models/common/\_\_init\_\_.py                       |        9 |        0 |    100% |           |
| ice\_station\_zebra/models/common/activations.py                        |        2 |        0 |    100% |           |
| ice\_station\_zebra/models/common/conv\_block\_common.py                |        8 |        0 |    100% |           |
| ice\_station\_zebra/models/common/conv\_block\_downsample.py            |       11 |        0 |    100% |           |
| ice\_station\_zebra/models/common/conv\_block\_upsample.py              |       13 |        0 |    100% |           |
| ice\_station\_zebra/models/common/conv\_block\_upsample\_naive.py       |        8 |        0 |    100% |           |
| ice\_station\_zebra/models/common/conv\_norm\_act.py                    |       20 |        4 |     80% | 41-46, 73 |
| ice\_station\_zebra/models/common/patchembed.py                         |       13 |        8 |     38% |19-24, 38-40 |
| ice\_station\_zebra/models/common/resizing\_interpolation.py            |       16 |        1 |     94% |        40 |
| ice\_station\_zebra/models/common/time\_embed.py                        |        9 |        4 |     56% | 26-30, 37 |
| ice\_station\_zebra/models/common/transformerblock.py                   |       12 |        7 |     42% |16-22, 40-42 |
| ice\_station\_zebra/models/decoders/\_\_init\_\_.py                     |        4 |        0 |    100% |           |
| ice\_station\_zebra/models/decoders/base\_decoder.py                    |       14 |        2 |     86% |     40-41 |
| ice\_station\_zebra/models/decoders/cnn\_decoder.py                     |       37 |        0 |    100% |           |
| ice\_station\_zebra/models/decoders/naive\_linear\_decoder.py           |       15 |        0 |    100% |           |
| ice\_station\_zebra/models/diffusion/\_\_init\_\_.py                    |        3 |        0 |    100% |           |
| ice\_station\_zebra/models/diffusion/gaussian\_diffusion.py             |       46 |       36 |     22% |41-74, 93-98, 117-137, 153-156, 177-182 |
| ice\_station\_zebra/models/diffusion/unet\_diffusion.py                 |       79 |       70 |     11% |56-176, 197-241, 257-272, 285-288 |
| ice\_station\_zebra/models/encode\_process\_decode.py                   |       21 |        0 |    100% |           |
| ice\_station\_zebra/models/encoders/\_\_init\_\_.py                     |        4 |        0 |    100% |           |
| ice\_station\_zebra/models/encoders/base\_encoder.py                    |       14 |        2 |     86% |     44-45 |
| ice\_station\_zebra/models/encoders/cnn\_encoder.py                     |       26 |        0 |    100% |           |
| ice\_station\_zebra/models/encoders/naive\_linear\_encoder.py           |       15 |        0 |    100% |           |
| ice\_station\_zebra/models/persistence.py                               |       16 |        0 |    100% |           |
| ice\_station\_zebra/models/processors/\_\_init\_\_.py                   |        6 |        0 |    100% |           |
| ice\_station\_zebra/models/processors/base\_processor.py                |       18 |        0 |    100% |           |
| ice\_station\_zebra/models/processors/ddpm.py                           |       27 |       17 |     37% |30-34, 46-50, 68-84 |
| ice\_station\_zebra/models/processors/null.py                           |       10 |        0 |    100% |           |
| ice\_station\_zebra/models/processors/unet.py                           |       53 |        0 |    100% |           |
| ice\_station\_zebra/models/processors/vit.py                            |       36 |       27 |     25% |36-69, 83-108 |
| ice\_station\_zebra/models/zebra\_model.py                              |       46 |        0 |    100% |           |
| ice\_station\_zebra/training/\_\_init\_\_.py                            |        2 |        0 |    100% |           |
| ice\_station\_zebra/training/cli.py                                     |       14 |        3 |     79% | 20-21, 25 |
| ice\_station\_zebra/training/trainer.py                                 |       31 |       17 |     45% |29-101, 104-105 |
| ice\_station\_zebra/types/\_\_init\_\_.py                               |        5 |        0 |    100% |           |
| ice\_station\_zebra/types/complex\_datatypes.py                         |       36 |       13 |     64% |37, 52-59, 63-65, 69 |
| ice\_station\_zebra/types/enums.py                                      |        7 |        0 |    100% |           |
| ice\_station\_zebra/types/simple\_datatypes.py                          |       37 |        0 |    100% |           |
| ice\_station\_zebra/types/typedefs.py                                   |       11 |        0 |    100% |           |
| ice\_station\_zebra/utils.py                                            |       23 |       14 |     39% |9, 14, 19-22, 29-36 |
| ice\_station\_zebra/visualisations/\_\_init\_\_.py                      |        5 |        0 |    100% |           |
| ice\_station\_zebra/visualisations/convert.py                           |       40 |       21 |     48% |28-34, 47-79 |
| ice\_station\_zebra/visualisations/layout.py                            |      205 |       28 |     86% |180, 417, 423, 437, 533-536, 538, 593-610, 622, 646, 661, 669-671, 724-730 |
| ice\_station\_zebra/visualisations/plotting\_core.py                    |      139 |       59 |     58% |52-58, 87-88, 111, 122-123, 168, 178-205, 229-230, 232-233, 255-256, 258-259, 297-300, 326-330, 360-361, 364, 381, 396, 423-425, 428-429, 437 |
| ice\_station\_zebra/visualisations/plotting\_maps.py                    |      220 |       40 |     82% |170-172, 188, 196-198, 254-257, 265-266, 289-292, 328-330, 338-339, 343-364, 478-493, 597, 603-604, 622-630, 689, 786, 819, 826, 838 |
| ice\_station\_zebra/visualisations/range\_check.py                      |       79 |       16 |     80% |32, 36-39, 46-47, 56, 61-63, 101, 110, 147, 168, 174 |
|                                                               **TOTAL** | **2464** |  **780** | **68%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/alan-turing-institute/ice-station-zebra/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/alan-turing-institute/ice-station-zebra/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/alan-turing-institute/ice-station-zebra/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/alan-turing-institute/ice-station-zebra/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Falan-turing-institute%2Fice-station-zebra%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/alan-turing-institute/ice-station-zebra/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.