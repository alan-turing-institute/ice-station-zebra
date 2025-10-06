# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/alan-turing-institute/ice-station-zebra/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                                    |    Stmts |     Miss |   Cover |   Missing |
|------------------------------------------------------------------------ | -------: | -------: | ------: | --------: |
| ice\_station\_zebra/\_\_init\_\_.py                                     |        0 |        0 |    100% |           |
| ice\_station\_zebra/callbacks/\_\_init\_\_.py                           |        5 |        0 |    100% |           |
| ice\_station\_zebra/callbacks/ema\_weight\_averaging\_callback.py       |       13 |        8 |     38% |24-28, 34-40 |
| ice\_station\_zebra/callbacks/metric\_summary\_callback.py              |       28 |       15 |     46% |25-27, 39-45, 54-61 |
| ice\_station\_zebra/callbacks/plotting\_callback.py                     |      102 |       80 |     22% |53-59, 73-103, 112-132, 141-170, 193-208, 225-233, 244-258 |
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
| ice\_station\_zebra/data\_processors/preprocessors/icenet\_sic.py       |       30 |       16 |     47% |18-24, 28, 33-57 |
| ice\_station\_zebra/data\_processors/preprocessors/ipreprocessor.py     |        9 |        2 |     78% |     10-11 |
| ice\_station\_zebra/data\_processors/preprocessors/null.py              |        5 |        1 |     80% |         9 |
| ice\_station\_zebra/data\_processors/zebra\_data\_processor.py          |       33 |       18 |     45% |25-31, 35-45, 49-51, 60-61 |
| ice\_station\_zebra/data\_processors/zebra\_data\_processor\_factory.py |       11 |        4 |     64% |     19-26 |
| ice\_station\_zebra/evaluation/\_\_init\_\_.py                          |        2 |        0 |    100% |           |
| ice\_station\_zebra/evaluation/cli.py                                   |       16 |        3 |     81% | 27-28, 32 |
| ice\_station\_zebra/evaluation/evaluator.py                             |       33 |       21 |     36% | 26-81, 86 |
| ice\_station\_zebra/exceptions.py                                       |        3 |        0 |    100% |           |
| ice\_station\_zebra/models/\_\_init\_\_.py                              |        4 |        0 |    100% |           |
| ice\_station\_zebra/models/common/\_\_init\_\_.py                       |       10 |        0 |    100% |           |
| ice\_station\_zebra/models/common/activations.py                        |        2 |        0 |    100% |           |
| ice\_station\_zebra/models/common/conv\_block\_common.py                |        8 |        0 |    100% |           |
| ice\_station\_zebra/models/common/conv\_block\_downsample.py            |       11 |        0 |    100% |           |
| ice\_station\_zebra/models/common/conv\_block\_upsample.py              |       13 |        0 |    100% |           |
| ice\_station\_zebra/models/common/conv\_block\_upsample\_naive.py       |        8 |        0 |    100% |           |
| ice\_station\_zebra/models/common/conv\_norm\_act.py                    |       20 |        4 |     80% | 41-46, 73 |
| ice\_station\_zebra/models/common/patchembed.py                         |       13 |        8 |     38% |19-24, 38-40 |
| ice\_station\_zebra/models/common/resizing\_average\_pool\_2d.py        |       13 |        6 |     54% | 24-47, 53 |
| ice\_station\_zebra/models/common/resizing\_interpolation.py            |        9 |        0 |    100% |           |
| ice\_station\_zebra/models/common/time\_embed.py                        |        9 |        4 |     56% | 26-30, 37 |
| ice\_station\_zebra/models/common/transformerblock.py                   |       12 |        7 |     42% |16-22, 40-42 |
| ice\_station\_zebra/models/decoders/\_\_init\_\_.py                     |        4 |        0 |    100% |           |
| ice\_station\_zebra/models/decoders/base\_decoder.py                    |       13 |        2 |     85% |     39-40 |
| ice\_station\_zebra/models/decoders/cnn\_decoder.py                     |       22 |        0 |    100% |           |
| ice\_station\_zebra/models/decoders/naive\_linear\_decoder.py           |       14 |        0 |    100% |           |
| ice\_station\_zebra/models/diffusion/\_\_init\_\_.py                    |        3 |        0 |    100% |           |
| ice\_station\_zebra/models/diffusion/gaussian\_diffusion.py             |       46 |       36 |     22% |41-74, 93-98, 117-137, 153-156, 177-182 |
| ice\_station\_zebra/models/diffusion/unet\_diffusion.py                 |       79 |       70 |     11% |56-176, 197-241, 257-272, 285-288 |
| ice\_station\_zebra/models/encode\_process\_decode.py                   |       21 |        0 |    100% |           |
| ice\_station\_zebra/models/encoders/\_\_init\_\_.py                     |        4 |        0 |    100% |           |
| ice\_station\_zebra/models/encoders/base\_encoder.py                    |       14 |        2 |     86% |     44-45 |
| ice\_station\_zebra/models/encoders/cnn\_encoder.py                     |       20 |        0 |    100% |           |
| ice\_station\_zebra/models/encoders/naive\_linear\_encoder.py           |       14 |        0 |    100% |           |
| ice\_station\_zebra/models/persistence.py                               |       16 |        0 |    100% |           |
| ice\_station\_zebra/models/processors/\_\_init\_\_.py                   |        6 |        0 |    100% |           |
| ice\_station\_zebra/models/processors/base\_processor.py                |       18 |        0 |    100% |           |
| ice\_station\_zebra/models/processors/ddpm.py                           |       27 |       17 |     37% |30-34, 46-50, 68-84 |
| ice\_station\_zebra/models/processors/null.py                           |       10 |        0 |    100% |           |
| ice\_station\_zebra/models/processors/unet.py                           |       53 |        0 |    100% |           |
| ice\_station\_zebra/models/processors/vit.py                            |       31 |       22 |     29% |38-62, 76-99 |
| ice\_station\_zebra/models/zebra\_model.py                              |       45 |        0 |    100% |           |
| ice\_station\_zebra/training/\_\_init\_\_.py                            |        2 |        0 |    100% |           |
| ice\_station\_zebra/training/cli.py                                     |       14 |        3 |     79% | 20-21, 25 |
| ice\_station\_zebra/training/trainer.py                                 |       31 |       17 |     45% |29-101, 104-105 |
| ice\_station\_zebra/types/\_\_init\_\_.py                               |        5 |        0 |    100% |           |
| ice\_station\_zebra/types/complex\_datatypes.py                         |       42 |       13 |     69% |37, 52-59, 63-65, 69 |
| ice\_station\_zebra/types/enums.py                                      |        7 |        0 |    100% |           |
| ice\_station\_zebra/types/simple\_datatypes.py                          |       49 |        0 |    100% |           |
| ice\_station\_zebra/types/typedefs.py                                   |       11 |        0 |    100% |           |
| ice\_station\_zebra/utils.py                                            |       12 |        6 |     50% |9, 14, 19-22 |
| ice\_station\_zebra/visualisations/\_\_init\_\_.py                      |        3 |        0 |    100% |           |
| ice\_station\_zebra/visualisations/convert.py                           |       37 |       22 |     41% |20-23, 28-34, 44-66 |
| ice\_station\_zebra/visualisations/layout.py                            |      158 |      132 |     16% |115-318, 338-347, 363-365, 379-381, 430-503, 536-553, 571-575, 610-626 |
| ice\_station\_zebra/visualisations/plotting\_core.py                    |      102 |       90 |     12% |23-25, 45-53, 79-115, 152-190, 213-219, 239-245, 252-272, 279-289 |
| ice\_station\_zebra/visualisations/plotting\_maps.py                    |       88 |       69 |     22% |82-121, 169-262, 307-409, 430, 446-447 |
|                                                               **TOTAL** | **1827** |  **851** | **53%** |           |


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