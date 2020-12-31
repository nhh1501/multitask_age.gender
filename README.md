# Multitask Age and Gender

The combination of age and gender tasks are trained in the same pineline


Dependencies
-

```
python 3
Pytorch 
```

Data
-
The dataset was preprocessed from the IMDB and WIKI dataset, and then labeled into age (2 classes) and gender (4 classes). Dataset these are available [here](https://drive.google.com/file/d/1A8TtUgF7nXglKIyajVe3SGvyzEsal0vN/view?usp=sharing/).

Running
-

+ ### Train<br>
  python main.py<br>

Citation
-
Model was created based on the concept of Cross-stitch Networks:

    @inproceedings{Ishan2016Cross-stitchN,
      title={Cross-stitch Networks for Multi-task Learning},
      author={Ishan Misra, Abhinav Shrivastava, Abhinav Gupta, Martial Hebert},
      booktitle={Proceedings of (CVPR) Computer Vision and Pattern Recognition},
      year={2016}
    }
