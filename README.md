# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/alan-turing-institute/ice-station-zebra/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                                    |    Stmts |     Miss |   Cover |   Missing |
|------------------------------------------------------------------------ | -------: | -------: | ------: | --------: |
| ice\_station\_zebra/\_\_init\_\_.py                                     |        0 |        0 |    100% |           |
| ice\_station\_zebra/callbacks/\_\_init\_\_.py                           |        4 |        0 |    100% |           |
| ice\_station\_zebra/callbacks/metric\_summary\_callback.py              |       28 |       15 |     46% |24-26, 38-44, 49-56 |
| ice\_station\_zebra/callbacks/plotting\_callback.py                     |       43 |       29 |     33% |29-33, 46-84 |
| ice\_station\_zebra/callbacks/unconditional\_checkpoint.py              |       21 |       10 |     52% |16-18, 23, 28-29, 33-34, 38-39 |
| ice\_station\_zebra/cli/\_\_init\_\_.py                                 |        3 |        0 |    100% |           |
| ice\_station\_zebra/cli/hydra.py                                        |       22 |        3 |     86% |     38-40 |
| ice\_station\_zebra/cli/main.py                                         |       12 |        1 |     92% |        23 |
| ice\_station\_zebra/config/\_\_init\_\_.py                              |        0 |        0 |    100% |           |
| ice\_station\_zebra/data\_loaders/\_\_init\_\_.py                       |        3 |        0 |    100% |           |
| ice\_station\_zebra/data\_loaders/combined\_dataset.py                  |       44 |       27 |     39% |22-40, 61, 72, 83-84, 88, 95, 102-106, 111-115 |
| ice\_station\_zebra/data\_loaders/zebra\_data\_module.py                |       49 |       29 |     41% |20-60, 71, 80, 90-110, 116-136, 142-162 |
| ice\_station\_zebra/data\_loaders/zebra\_dataset.py                     |       45 |        0 |    100% |           |
| ice\_station\_zebra/data\_processors/\_\_init\_\_.py                    |        2 |        0 |    100% |           |
| ice\_station\_zebra/data\_processors/cli.py                             |       23 |        9 |     61% |20-23, 30-33, 37 |
| ice\_station\_zebra/data\_processors/preprocessors/\_\_init\_\_.py      |        3 |        0 |    100% |           |
| ice\_station\_zebra/data\_processors/preprocessors/base.py              |        9 |        2 |     78% |     8, 11 |
| ice\_station\_zebra/data\_processors/preprocessors/icenet\_sic.py       |       30 |       16 |     47% |17-23, 27, 31-55 |
| ice\_station\_zebra/data\_processors/zebra\_data\_processor.py          |       34 |       18 |     47% |22-28, 32-40, 44-46, 55-56 |
| ice\_station\_zebra/data\_processors/zebra\_data\_processor\_factory.py |       10 |        4 |     60% |     14-21 |
| ice\_station\_zebra/evaluation/\_\_init\_\_.py                          |        2 |        0 |    100% |           |
| ice\_station\_zebra/evaluation/cli.py                                   |       16 |        3 |     81% | 27-28, 32 |
| ice\_station\_zebra/evaluation/evaluator.py                             |       34 |       22 |     35% | 21-66, 77 |
| ice\_station\_zebra/models/\_\_init\_\_.py                              |        4 |        0 |    100% |           |
| ice\_station\_zebra/models/decoders/\_\_init\_\_.py                     |        3 |        3 |      0% |       1-4 |
| ice\_station\_zebra/models/decoders/base\_decoder.py                    |        9 |        9 |      0% |      1-24 |
| ice\_station\_zebra/models/decoders/naive\_latent\_space\_decoder.py    |       20 |       20 |      0% |      1-67 |
| ice\_station\_zebra/models/encode\_process\_decode.py                   |       23 |       13 |     43% |23-54, 73-87 |
| ice\_station\_zebra/models/encoders/\_\_init\_\_.py                     |        3 |        0 |    100% |           |
| ice\_station\_zebra/models/encoders/base\_encoder.py                    |       10 |        3 |     70% |     20-22 |
| ice\_station\_zebra/models/encoders/naive\_latent\_space\_encoder.py    |       20 |       12 |     40% | 25-53, 65 |
| ice\_station\_zebra/models/persistence.py                               |       16 |        6 |     62% |17-19, 23, 32-33 |
| ice\_station\_zebra/models/processors/\_\_init\_\_.py                   |        3 |        3 |      0% |       1-4 |
| ice\_station\_zebra/models/processors/null.py                           |        9 |        9 |      0% |      1-28 |
| ice\_station\_zebra/models/processors/unet.py                           |       44 |       44 |      0% |      1-82 |
| ice\_station\_zebra/models/zebra\_model.py                              |       39 |       22 |     44% |26-45, 57, 68, 87-90, 109-111, 133-137 |
| ice\_station\_zebra/training/\_\_init\_\_.py                            |        2 |        0 |    100% |           |
| ice\_station\_zebra/training/cli.py                                     |       14 |        3 |     79% | 20-21, 25 |
| ice\_station\_zebra/training/trainer.py                                 |       29 |       15 |     48% | 23-87, 90 |
| ice\_station\_zebra/types.py                                            |       68 |       13 |     81% |61, 67, 81-87, 90-92, 95 |
| ice\_station\_zebra/utils.py                                            |       12 |        6 |     50% |9, 13, 18-21 |
| ice\_station\_zebra/visualisations/\_\_init\_\_.py                      |        2 |        0 |    100% |           |
| ice\_station\_zebra/visualisations/convert.py                           |        8 |        4 |     50% |      9-12 |
| ice\_station\_zebra/visualisations/sea\_ice\_concentration.py           |       20 |       13 |     35% |     17-36 |
|                                                               **TOTAL** |  **795** |  **386** | **51%** |           |


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