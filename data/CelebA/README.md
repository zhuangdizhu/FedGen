#### To generated CelebA dataset
##### Step 1.
follow the [LEAF instructions](https://github.com/TalwalkarLab/leaf/tree/master/data/celeba) to download the raw celeb data, and generate `train` and `test` subfolders.

#### Step 2.
change [`LOAD_PATH`](https://github.com/zhuangdizhu/FedGen/blob/05625ef130f681075fb04b804322e33ef31f6dea/data/CelebA/generate_niid_agg.py#L15) in the `generate_niid_agg.py` to point to the folder of the raw celeba downloaded in step 1.

#### Step 3.
run `generate_niid_agg.py` to generate FL training and testing dataset.
For example, to generate data for 25 FL devices, where each device contains images of 10 celebrities:
```
python generate_niid_agg.py --agg_user 10 --ratio 250
```
