# Personalized Image Enhancement Featuring Masked Style Modeling
This is the official implementation of the paper in TCSVT. [[paper]](https://ieeexplore.ieee.org/abstract/document/10149499)

<p align="left">
<img src="figs/method.jpg" alt="architecture" width="875px">
</p>

## Requirements
- Python3.7.5

To install the Python libraries,
```Shell
pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset
The details of the dataset are described in the [data/](./data/) directory.

## Training
Start the first training step.
```Shell
python train1.py
```
Start the second training step.
```Shell
python train2.py
```

## Test
Test our model.
```Shell
python test.py
```
To test our model with the pretrained models, please download the pretrained models from [here](https://drive.google.com/file/d/1NVkmB8ADiux1rImRuQXgH1e9g4EU8F8Z/view).
```Shell
python test.py --save_dir pretrained_models
```

## Reference
Our implementation is based on [StarEnhancer](https://github.com/IDKiro/StarEnhancer). We would like to thank them.


## Citation
If you find our research useful in your research, please consider citing:

    @article{kosugi2023personalized,
      title={Personalized Image Enhancement Featuring Masked Style Modeling},
      author={Kosugi, Satoshi and Yamasaki, Toshihiko},
      journal={IEEE Transactions on Circuits and Systems for Video Technology},
      year={2023}
    }
