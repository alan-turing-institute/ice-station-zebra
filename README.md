# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/alan-turing-institute/ice-station-zebra/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                                    |    Stmts |     Miss |   Cover |   Missing |
|------------------------------------------------------------------------ | -------: | -------: | ------: | --------: |
| ice\_station\_zebra/\_\_init\_\_.py                                     |        0 |        0 |    100% |           |
| ice\_station\_zebra/callbacks/\_\_init\_\_.py                           |        4 |        0 |    100% |           |
| ice\_station\_zebra/callbacks/metric\_summary\_callback.py              |       28 |       15 |     46% |25-27, 39-45, 54-61 |
| ice\_station\_zebra/callbacks/plotting\_callback.py                     |       42 |       29 |     31% |32-36, 49-87 |
| ice\_station\_zebra/callbacks/unconditional\_checkpoint.py              |       21 |       10 |     52% |17-19, 24, 29-30, 34-35, 39-40 |
| ice\_station\_zebra/cli/\_\_init\_\_.py                                 |        3 |        0 |    100% |           |
| ice\_station\_zebra/cli/hydra.py                                        |       22 |        3 |     86% |     39-41 |
| ice\_station\_zebra/cli/main.py                                         |       12 |        1 |     92% |        23 |
| ice\_station\_zebra/config/\_\_init\_\_.py                              |        0 |        0 |    100% |           |
| ice\_station\_zebra/data\_loaders/\_\_init\_\_.py                       |        3 |        0 |    100% |           |
| ice\_station\_zebra/data\_loaders/combined\_dataset.py                  |       44 |       27 |     39% |27-45, 66, 78, 89-90, 94, 101, 108-112, 117-121 |
| ice\_station\_zebra/data\_loaders/zebra\_data\_module.py                |       50 |       30 |     40% |25-66, 77, 86, 96-116, 122-142, 148-168 |
| ice\_station\_zebra/data\_loaders/zebra\_dataset.py                     |       45 |        0 |    100% |           |
| ice\_station\_zebra/data\_processors/\_\_init\_\_.py                    |        2 |        0 |    100% |           |
| ice\_station\_zebra/data\_processors/cli.py                             |       23 |        9 |     61% |20-23, 30-33, 37 |
| ice\_station\_zebra/data\_processors/preprocessors/\_\_init\_\_.py      |        3 |        0 |    100% |           |
| ice\_station\_zebra/data\_processors/preprocessors/base.py              |        8 |        1 |     88% |         9 |
| ice\_station\_zebra/data\_processors/preprocessors/icenet\_sic.py       |       30 |       16 |     47% |18-24, 28, 33-57 |
| ice\_station\_zebra/data\_processors/zebra\_data\_processor.py          |       33 |       18 |     45% |25-31, 35-45, 49-51, 60-61 |
| ice\_station\_zebra/data\_processors/zebra\_data\_processor\_factory.py |       11 |        4 |     64% |     19-26 |
| ice\_station\_zebra/evaluation/\_\_init\_\_.py                          |        2 |        0 |    100% |           |
| ice\_station\_zebra/evaluation/cli.py                                   |       16 |        3 |     81% | 27-28, 32 |
| ice\_station\_zebra/evaluation/evaluator.py                             |       31 |       20 |     35% | 25-69, 80 |
| ice\_station\_zebra/models/\_\_init\_\_.py                              |        4 |        4 |      0% |       1-5 |
| ice\_station\_zebra/models/common/\_\_init\_\_.py                       |        4 |        4 |      0% |       1-5 |
| ice\_station\_zebra/models/common/bottleneckblock.py                    |        7 |        7 |      0% |      1-28 |
| ice\_station\_zebra/models/common/convblock.py                          |       11 |       11 |      0% |      1-45 |
| ice\_station\_zebra/models/common/upconvblock.py                        |        7 |        7 |      0% |      1-16 |
| ice\_station\_zebra/models/decoders/\_\_init\_\_.py                     |        3 |        3 |      0% |       1-4 |
| ice\_station\_zebra/models/decoders/base\_decoder.py                    |        9 |        9 |      0% |      1-24 |
| ice\_station\_zebra/models/decoders/naive\_latent\_space\_decoder.py    |       20 |       20 |      0% |      1-67 |
| ice\_station\_zebra/models/encode\_process\_decode.py                   |       22 |       22 |      0% |      1-90 |
| ice\_station\_zebra/models/encoders/\_\_init\_\_.py                     |        3 |        3 |      0% |       1-4 |
| ice\_station\_zebra/models/encoders/base\_encoder.py                    |       10 |       10 |      0% |      1-25 |
| ice\_station\_zebra/models/encoders/naive\_latent\_space\_encoder.py    |       20 |       20 |      0% |      1-65 |
| ice\_station\_zebra/models/persistence.py                               |       16 |       16 |      0% |      1-34 |
| ice\_station\_zebra/models/processors/\_\_init\_\_.py                   |        3 |        3 |      0% |       1-4 |
| ice\_station\_zebra/models/processors/null.py                           |        9 |        9 |      0% |      1-29 |
| ice\_station\_zebra/models/processors/unet.py                           |       40 |       40 |      0% |      1-86 |
| ice\_station\_zebra/models/zebra\_model.py                              |       39 |       39 |      0% |     1-153 |
| ice\_station\_zebra/training/\_\_init\_\_.py                            |        2 |        0 |    100% |           |
| ice\_station\_zebra/training/cli.py                                     |       14 |        3 |     79% | 20-21, 25 |
| ice\_station\_zebra/training/trainer.py                                 |       28 |       15 |     46% | 28-95, 98 |
| ice\_station\_zebra/types.py                                            |       69 |       14 |     80% |62, 68, 83-90, 94-96, 100 |
| ice\_station\_zebra/utils.py                                            |       12 |        6 |     50% |9, 14, 19-22 |
| ice\_station\_zebra/visualisations/\_\_init\_\_.py                      |        2 |        0 |    100% |           |
| ice\_station\_zebra/visualisations/convert.py                           |        8 |        4 |     50% |      9-12 |
| ice\_station\_zebra/visualisations/sea\_ice\_concentration.py           |       20 |       13 |     35% |     20-39 |
|                                                               **TOTAL** |  **815** |  **468** | **43%** |           |


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