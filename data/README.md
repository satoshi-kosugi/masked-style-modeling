## Dataset

We use two types of data: the FiveK dataset and images downloaded from Flickr.

## FiveK dataset

We provide pre-processed images from the FiveK dataset.
These can be downloaded from the following links.

[[test.tar.gz]](https://drive.google.com/file/d/1D6F45yyO_ukk64Ym3n5iTF7wF9Sr1wu3/view?usp=sharing)
[[test_ref.tar.gz]](https://drive.google.com/file/d/1Gi4nFlMvXH7djAsOMaBuGQKtjfxjerRq/view?usp=sharing)
[[train_part1.tar.gz]](https://drive.google.com/file/d/1OQlfPrwWGXObEj9L_96NoNcVF1v7dG4I/view?usp=sharing)
[[train_part2.tar.gz]](https://drive.google.com/file/d/1QPlpeRwLb2EnQanILkhag_ZXETxzZynO/view?usp=sharing)

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

Download the weights of the degradation model [[degrading_model.pth]](https://drive.google.com/file/d/18h_0xruPTnEWEzJou3AG04IRxWR_KVMn/view?usp=sharing) and place it in `masked-style-modeling/data/`.

Run the degrading process by executing the following code.
Before proceeding, ensure that you have installed all libraries required by Deep Symmetric Network.
```Shell
git clone https://github.com/lin-zhao-resoLve/Deep-Symmetric-Network-Enhancement.git
cp degrade.py Deep-Symmetric-Network-Enhancement/codes/
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
