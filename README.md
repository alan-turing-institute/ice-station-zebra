# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/alan-turing-institute/ice-station-zebra/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                                  |    Stmts |     Miss |   Cover |   Missing |
|---------------------------------------------------------------------- | -------: | -------: | ------: | --------: |
| ice\_station\_zebra/\_\_init\_\_.py                                   |        0 |        0 |    100% |           |
| ice\_station\_zebra/cli/\_\_init\_\_.py                               |        3 |        0 |    100% |           |
| ice\_station\_zebra/cli/hydra.py                                      |       22 |        3 |     86% |     38-40 |
| ice\_station\_zebra/cli/main.py                                       |       12 |        1 |     92% |        23 |
| ice\_station\_zebra/config/\_\_init\_\_.py                            |        0 |        0 |    100% |           |
| ice\_station\_zebra/data/anemoi/\_\_init\_\_.py                       |        2 |        0 |    100% |           |
| ice\_station\_zebra/data/anemoi/cli.py                                |       23 |        9 |     61% |20-23, 30-33, 37 |
| ice\_station\_zebra/data/anemoi/preprocessors/\_\_init\_\_.py         |        3 |        0 |    100% |           |
| ice\_station\_zebra/data/anemoi/preprocessors/base.py                 |        9 |        2 |     78% |     8, 11 |
| ice\_station\_zebra/data/anemoi/preprocessors/icenet\_sic.py          |       30 |       16 |     47% |17-23, 27, 31-55 |
| ice\_station\_zebra/data/anemoi/zebra\_dataset.py                     |       34 |       18 |     47% |21-27, 31-39, 43-45, 54-55 |
| ice\_station\_zebra/data/anemoi/zebra\_dataset\_factory.py            |       10 |        4 |     60% |     14-21 |
| ice\_station\_zebra/data/lightning/\_\_init\_\_.py                    |        3 |        0 |    100% |           |
| ice\_station\_zebra/data/lightning/combined\_dataset.py               |       44 |       27 |     39% |21-39, 60, 71, 82-83, 87, 94, 101-105, 110-114 |
| ice\_station\_zebra/data/lightning/zebra\_data\_module.py             |       49 |       29 |     41% |20-60, 71, 80, 90-110, 116-136, 142-162 |
| ice\_station\_zebra/data/lightning/zebra\_dataset.py                  |       45 |        0 |    100% |           |
| ice\_station\_zebra/evaluation/\_\_init\_\_.py                        |        2 |        0 |    100% |           |
| ice\_station\_zebra/evaluation/callbacks/\_\_init\_\_.py              |        3 |        3 |      0% |       1-4 |
| ice\_station\_zebra/evaluation/callbacks/metric\_summary\_callback.py |       20 |       20 |      0% |      1-45 |
| ice\_station\_zebra/evaluation/callbacks/plotting\_callback.py        |       33 |       33 |      0% |      1-76 |
| ice\_station\_zebra/evaluation/cli.py                                 |       16 |        3 |     81% | 27-28, 32 |
| ice\_station\_zebra/evaluation/evaluator.py                           |       34 |       22 |     35% | 21-66, 77 |
| ice\_station\_zebra/models/\_\_init\_\_.py                            |        3 |        0 |    100% |           |
| ice\_station\_zebra/models/decoders/\_\_init\_\_.py                   |        3 |        3 |      0% |       1-4 |
| ice\_station\_zebra/models/decoders/base\_decoder.py                  |        9 |        9 |      0% |      1-24 |
| ice\_station\_zebra/models/decoders/naive\_latent\_space\_decoder.py  |       20 |       20 |      0% |      1-66 |
| ice\_station\_zebra/models/encode\_process\_decode.py                 |       23 |       13 |     43% |23-54, 73-87 |
| ice\_station\_zebra/models/encoders/\_\_init\_\_.py                   |        3 |        0 |    100% |           |
| ice\_station\_zebra/models/encoders/base\_encoder.py                  |       10 |        3 |     70% |     19-21 |
| ice\_station\_zebra/models/encoders/naive\_latent\_space\_encoder.py  |       20 |       12 |     40% | 24-52, 64 |
| ice\_station\_zebra/models/processors/\_\_init\_\_.py                 |        3 |        3 |      0% |       1-4 |
| ice\_station\_zebra/models/processors/null.py                         |        9 |        9 |      0% |      1-27 |
| ice\_station\_zebra/models/processors/unet.py                         |       44 |       44 |      0% |      1-82 |
| ice\_station\_zebra/models/zebra\_model.py                            |       39 |       22 |     44% |24-43, 55, 65, 79-82, 96-98, 112-116 |
| ice\_station\_zebra/training/\_\_init\_\_.py                          |        2 |        0 |    100% |           |
| ice\_station\_zebra/training/cli.py                                   |       14 |        3 |     79% | 20-21, 25 |
| ice\_station\_zebra/training/trainer.py                               |       25 |       13 |     48% | 21-81, 84 |
| ice\_station\_zebra/types.py                                          |       49 |        2 |     96% |    61, 67 |
| ice\_station\_zebra/utils.py                                          |       12 |        6 |     50% |9, 13, 18-21 |
| ice\_station\_zebra/visualisations/\_\_init\_\_.py                    |        2 |        2 |      0% |       1-3 |
| ice\_station\_zebra/visualisations/convert.py                         |        8 |        8 |      0% |      1-12 |
| ice\_station\_zebra/visualisations/sea\_ice\_concentration.py         |       17 |       17 |      0% |      1-28 |
|                                                             **TOTAL** |  **712** |  **379** | **47%** |           |


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