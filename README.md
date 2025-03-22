# User disambiguation learning for precise shared-account marketing: A hierarchical self-attentive sequential recommendation method

## Introduction

This repository contains the code of the HierSASRec model for shared-account recommendations, which is proposed in:

> Duan, W., Liang, D., 2025. User disambiguation learning for precise shared-account marketing: A hierarchical self-attentive sequential recommendation method. Knowledge-Based Systems 315, 113328. https://doi.org/10.1016/j.knosys.2025.113328

## Function

- [main.py](./main.py): It is the main body of the HierSASRec model for training and testing as the result.
- [model.py](./model.py): The architecture of the HierSASRec model.
- [data_utils.py](./model.py): It includes some processing process and the calculation of the evaluation metrics.
- [utils.py](./utils.py): The utils for implementing the user disambiguation through time-aware DBSCAN.
- [split.py](./split.py): The process of time-aware DBSCAN to realize user disambiguation.

## Reference

If you are interested in our researches, please cite our paper:

> @article{Duan_Liang_2025, title={User disambiguation learning for precise shared-account marketing: A hierarchical self-attentive sequential recommendation method}, volume={315}, ISSN={09507051}, DOI={10.1016/j.knosys.2025.113328}, journal={Knowledge-Based Systems}, author={Duan, Weiyi and Liang, Decui}, year={2025}, month=apr, pages={113328}, language={en} }

