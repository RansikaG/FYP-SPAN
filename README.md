![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)
# Semantics-guided Part Attention Network
This is the pytorch implementatin of Semantics-guided Part Attention Network **(SPAN)**

## Paper
Orientation-aware Vehicle Re-identification with Semantics-guided Part Attention Network <br/>
[Tsai-Shien Chen](https://tsaishien-chen.github.io/), [Chih-Ting Liu](https://jackie840129.github.io/), Chih-Wei Wu, and [Shao-Yi Chien](http://www.ee.ntu.edu.tw/profile?id=101), <br/>
European Conference on Computer Vision (**ECCV**), 2020 <br/>
[[Paper Website]](http://media.ee.ntu.edu.tw/research/SPAN/) [[arXiv]](https://arxiv.org/abs/2008.11423)

### Citation
If you use SPAN, please cite this paper:
```
@article{SPAN,
  	title   = {Orientation-aware Vehicle Re-identification with Semantics-guided Part Attention Network},
  	author  = {Chen, Tsai-Shien and Liu, Chih-Ting and Wu, Chih-Wei and Chien, Shao-Yi},
  	journal = {arXiv preprint arXiv:2008.11423},
  	year    = {2020}
}
```

## Visualization Example
We show some examples of
- original image (VeRi-776 Dataset)
- foreground mask generated by Grabcut
- foreground mask generated by Deep Learning
- front, rear and side attention mask
<p align="center"><img src='figures/example.png'></p>