## Dataset

We use two types of data: the FiveK dataset and images downloaded from Flickr.

## FiveK dataset

We provide pre-processed images from the FiveK dataset.
These can be downloaded using the following command:
```Shell
sh download_fivek.sh
```

This should result in the following directory structure.
```
masked-style-modeling/data/
  ├── test/
    ├── ExpertA/
    ├── ExpertA_input/
    ...
  ├── test_ref/
    ├── ExpertA/
    ├── ExpertA_input/
    ...
  └── train/
    ├── CDSA/
    ├── CDSA_input/
    ...
```

The FiveK dataset is licensed by the provider; please adhere to its terms.
Details can be found at [this link](https://data.csail.mit.edu/graphics/fivek/).



## Images downloaded from Flickr

Images can be downloaded using the following command from Flickr:
```Shell
python download_from_flickr.py
```

This should result in the following directory structure.
```
masked-style-modeling/data/
  ├── test/
  ├── test_ref/
  ├── train/
    ├── 100015017_N04/
    ...
  ├── valid/
    ├── 101295391_N07/
    ...
  └── valid_ref/
    ├── 101295391_N07/
    ...
```

By applying a degrading model to the downloaded images, we create pairs of pseudo original and retouched images.
We use [Deep Symmetric Network](https://github.com/lin-zhao-resoLve/Deep-Symmetric-Network-Enhancement) as the degrading model.

Run the degrading process by executing the following code.
Before proceeding, ensure that you have installed all libraries required by Deep Symmetric Network.
```Shell
git clone https://github.com/lin-zhao-resoLve/Deep-Symmetric-Network-Enhancement.git
cp degrade.py Deep-Symmetric-Network-Enhancement/codes/
wget https://www.hal.t.u-tokyo.ac.jp/~kosugi/masked-style-modeling/data/degrading_model.pth
cd Deep-Symmetric-Network-Enhancement/codes/
python degrade.py --opt options/test/test_for_fiveK.yml --path_to_data ../../
```

This should result in the following directory structure.
```
masked-style-modeling/data/
  ├── test/
  ├── test_ref/
  ├── train/
    ├── 100015017_N04/
    ├── 100015017_N04_input/
    ...
  ├── valid/
    ├── 101295391_N07/
    ├── 101295391_N07_input/
    ...
  └── valid_ref/
    ├── 101295391_N07/
    ├── 101295391_N07_input/
    ...
```

All images uploaded to Flickr are subject to the licensing terms set by the uploader. Please adhere to these terms when using images downloaded from Flickr.
