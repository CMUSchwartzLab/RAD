# Robust and Accurate Deconvolution (RAD)

## Introduction

RAD is a toolkit that unmixes bulk tumor samples. Given a non-negative bulk RNA expression matrix $$B \in R_{+}^{m \times n}$$, where each row $$i$$ is a gene, each column $$j$$ is a tumor sample, our goal is to infer an expression profile matrix $$C \in R_{+}^{m \times k}$$, where each column $$l$$ is a cell community, and a fraction matrix $$F \in R_{+}^{k \times n}$$, such that:
$$B \approx C F$$.
To be more specific, RAD solves the following problem:
$$\min_{C, F} \| B - C F \|_{Fr}^2,$$ such that:
$$C_{i,l} \geq 0, i = 1,...,m, l=1,...,k,$$
 $$F_{l,j} \geq 0, l=1,...,k, j = 1,...,n,$$
 $$\sum_{lj} = 1, j = 1,...,n.$$
 
RAD has the following features and advantages:
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
Yifeng Tao, Haoyun Lei, Xuecong Fu, Adrian V. Lee, Jian Ma, and Russell Schwartz. [**Robust and accurate deconvolution of tumor populations uncovers evolutionary mechanisms of breast cancer metastasis**](https://academic.oup.com/bioinformatics/article-pdf/36/Supplement_1/i407/33488922/btaa396.pdf). *Bioinformatics*, 36(Supplement_1):i407-i416. jul 2020.
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

We compared RAD with a few other methods in the paper, you can find the links to these algorithms below:
* Geometric Unmixing: [Paper](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-11-42), [Code](http://www.cs.cmu.edu/~russells/software/unmixing/)
* LinSeed: [Paper](https://www.nature.com/articles/s41467-019-09990-5), [Code](https://github.com/ctlab/LinSeed)
* NND: [Paper](https://link.springer.com/chapter/10.1007%2F978-3-030-35210-3_1), [Code](https://github.com/CMUSchwartzLab/BrM-Phylo)

