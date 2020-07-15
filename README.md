# Robust and Accurate Deconvolution (RAD)

## Introduction

RAD is a toolkit that unmixes bulk tumor samples.

It has the following features and advantages:
* `compress_module`: Integrate gene module knowledge to reduce noise.
* `estimate_number`: Estimate the number of cell populations automatically.
* `estimate_clones`: Utilize core RAD algorithm to unmix the cell populations accurately and robustly.
* `estimate_marker`: Estimate other biomarkers of cell populations given bulk marker data.


## Prerequisites

The code runs on Python 3. You will need to install the additional Python package `cvxopt`. Most other packages are available in the Anaconda.


## Tutorial

You can find a brief tutorial in `tutorial.py`.


## Citation

If you find RAD helpful, please cite the following paper: 
Yifeng Tao, Haoyun Lei, Xuecong Fu, Adrian V. Lee, Jian Ma, and Russell Schwartz. [**Robust and accurate deconvolution of tumor populations uncovers evolutionary mechanisms of breast cancer metastasis**](https://academic.oup.com/bioinformatics/article-pdf/36/Supplement_1/i407/33488922/btaa396.pdf). *Bioinformatics*, 36(Supplement_1): i407-i416. jul 2020.
```
@article{tao2020rad,
  title = {Robust and Accurate Deconvolution of Tumor Populations Uncovers Evolutionary Mechanisms of Breast Cancer Metastasis},
  author = {Tao, Yifeng and Lei, Haoyun and Fu, Xuecong and Lee, Adrian V and Ma, Jian and Schwartz, Russell},
  journal = {Bioinformatics},
  volume = {36},
  number = {Supplement_1},
  pages = {i407-i416},
  year = {2020},
  month = {jul},
  issn = {1367-4803},
  doi = {10.1093/bioinformatics/btaa396},
  url = {https://doi.org/10.1093/bioinformatics/btaa396},
  eprint = {https://academic.oup.com/bioinformatics/article-pdf/36/Supplement\_1/i407/33488922/btaa396.pdf},
}
```