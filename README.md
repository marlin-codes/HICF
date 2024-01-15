## [HICF: Hyperbolic Informative Graph Collaborative Filtering [PDF] ](https://arxiv.org/abs/2207.09051)

## 1. Overview
This repository is an official PyTorch Implementation for "[Hyperbolic Informative Collaborative Filtering(KDD2022)](https://arxiv.org/abs/2207.09051)"

**Authors**: Menglin Yang, Zhihao Li, Min Zhou, Jiahong Liu, Irwin King \
**Codes**: https://github.com/marlin-codes/HICF

Note: this project is built upon [HRCF](https://github.com/marlin-codes/HRCF), [HGCF](https://github.com/layer6ai-labs/HGCF) and [HGCN](https://github.com/HazyResearch/hgcn), [LightGCN](https://github.com/gusye1234/LightGCN-PyTorch). HRCF is also of our work, but we didn't list its results in the HICF report because the HRCF was still under review when we submitted the HICF. We will make a full comparions in an extended version. By the way, if you would like to list HICF as a baseline, please follow the parameter's setting.

<a name="Environment"/>

## 2. Environment:

The code was developed and tested on the following python environment: 
```
python 3.7.7
pytorch 1.11.0
scikit-learn 0.23.2
numpy 1.20.2
scipy 1.6.2
tqdm 4.60.0
```
<a name="instructions"/>

## 3. Instructions:

Train and evaluate HICF:

- To evaluate HICF on Amazon_CD 
  - `bash ./examples/Amazon-CD/run_cd.sh`
- To evaluate HICF on Amazon_Book
   - `bash ./examples/Amazon-Book/run_book.sh`
- To evaluate HICF on Yelp2020
    - `bash ./examples/yelp/run_yelp.sh`

<a name="citation"/>

## 4. Citation

If you find this code useful in your research, please cite the following paper:

@inproceedings{yang2022hicf, \
  title={{HICF}: Hyperbolic informative collaborative filtering}, \
  author={Yang, Menglin and Li, Zhihao and Zhou, Min and Liu, Jiahong and King, Irwin}, \
  booktitle={Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD)},\
  pages={2212--2221},\
  year={2022}\
}
