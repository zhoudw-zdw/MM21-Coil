
# Co-Transport for Class-Incremental Learning  (Coil)

The code repository for "Co-Transport for Class-Incremental Learning
" [[paper]](http://arxiv.org/abs/2107.12654) (ACM MM'21) in PyTorch. If you use any content of this repo for your work, please cite the following bib entry:

    @inproceedings{zhou2021transport,
    author = {Zhou, Da-Wei and Ye, Han-Jia and Zhan, De-Chuan},
    title = {Co-Transport for Class-Incremental Learning},
    booktitle = {ACM MM},
    pages = {1645-1654},
    year = {2021}
    }

Update: We release a code repo for [Class-Incremental Learning](https://github.com/G-U-N/PyCIL) with more than 10 algorithms, and the code of Coil is also contained in it.

## Co-Transport for Class-Incremental Learning


Traditional learning systems are trained in closed-world for a fixed number of classes, and need pre-collected datasets in advance. However, new classes often emerge in real-world applications and should be learned incrementally. For example, in electronic commerce, new types of products appear daily, and in a social media community, new topics emerge frequently. Under such circumstances, incremental models should learn several new classes at a time without forgetting. We find a strong correlation between old and new classes in incremental learning, which can be applied to relate and facilitate different learning stages mutually. As a result, we propose CO-transport for class Incremental Learning (COIL), which learns to relate across incremental tasks with the class-wise semantic relationship. In detail, co-transport has two aspects: prospective transport tries to augment the old classifier with optimal transported knowledge as fast model adaptation. Retrospective transport aims to transport new class classifiers backward as old ones to overcome forgetting. With these transports, COIL efficiently adapts to new tasks, and stably resists forgetting. Experiments on benchmark and real-world multimedia datasets validate the effectiveness of our proposed method.

<img src='imgs/coil.png' width='700' height='320'>

## Prerequisites

The following packages are required to run the scripts:

- [PyTorch-1.4 and torchvision](https://pytorch.org)

- [POT](https://github.com/PythonOT/POT)

- Dataset: the code will automatically download CIFAR100 for training. 




## Code Structures
There are four parts in the code.
 - `models`: It contains the core algorithm of Coil.
 - `data`: Images and splits for the data sets.
 - `utlis`: The flow and pre-processing code for class-incremental learning.
- `convs`: It contains the backbone used for training
 
## Class-Incremental Learning for CIFAR100
We provide the code to reproduce results on CIFAR 100.
  ```
  python main.py --dataset cifar100 --shuffle 1 --init_cls 20 --increment 20 --model_name COIL --convnet_type cosine_resnet32 --seed 1993
  ```
  
  
## Acknowledgment
We thank the following repos providing helpful components/functions in our work.
- [continual-learning-reproduce](https://github.com/zhchuu/continual-learning-reproduce)

- [Proser](https://github.com/zhoudw-zdw/CVPR21-Proser)



## Contact 
If there are any questions, please feel free to contact with the author:  Da-Wei Zhou (zhoudw@lamda.nju.edu.cn). Enjoy the code.
