# Dense Trajectory Fields: Consistent and Efficient Spatio-Temporal Pixel Tracking

This repository contains code for DTF-Net, and experiments presented in the
ACCV 2024 publication
[*Dense Trajectory Fields: Consistent and Efficient Spatio-Temporal Pixel Tracking*](https://mtournadre.github.io/DTFNet/)

## Install

The conda environment is defined in `env.yml`.

```
conda env create --file env.yml
conda activate dtfnet
```

## Kubric-DTF Dataset

Data generation is done through the `generate_movi_dataset.py` script:
it enables to either build the DTF ground-truth on an existing
Kubric dataset, or generate it from scratch, using the original
[Kubric repository](https://github.com/google-research/kubric).

### From existing data

Easier and faster, you can generate the DTF ground-truth on
existing Kubric data.
By default, it will download samples from the public Google Storage.

```
python generate_movi_dataset.py from_data \
    --output_folder /path/to/kubric_dtf/movi_e/train \
    --dataset_name 'movi_e/256x256' \
    --subset train
python generate_movi_dataset.py from_data \
    --output_folder /path/to/kubric_dtf/movi_e/test \
    --dataset_name 'movi_e/256x256' \
    --subset test
```

### From scratch

You can also render all training & test data from scratch
using the Kubric source code.
This method is slower, but more reliable.
You need to download the
[Kubric repository](https://github.com/google-research/kubric) first,
and use their docker image.

```
git clone https://github.com/google-research/kubric
docker pull kubricdockerhub/kubruntu
python generate_movi_dataset.py from_source \
    --output_folder /path/to/kubric_dtf/movi_e/test \
    --dataset_name 'movi_e/256x256' \
    --resolution 256x256 \
    --kubric_source_path $PWD/kubric \
    --split test
python generate_movi_dataset.py from_source \
    --output_folder /path/to/kubric_dtf/movi_e/train \
    --dataset_name 'movi_e/256x256' \
    --resolution 256x256 \
    --kubric_source_path $PWD/kubric \
    --split train
```

## Training

Training is done by the `train.py` script.
`train.sh` contains the arguments used for training all 3 stages of DTF-Net.
You will need to adapt it to your environment.

A lighter model can be quickly trained with `train_light.sh`, for tests.

## Evaluation

The presented checkpoint is accessible [here](https://drive.google.com/file/d/1SLH5NxiHJzbimOldKZPR7o2qkOHkctpg/view?usp=sharing).

`evaluate_model.py` contains all the necessary code for inference,
and to evaluate diverse methods on the presented datasets.
You *need* the [TAPNet-TAPIR repository](https://github.com/google-deepmind/tapnet)
(Apache 2.0 License) for dataloading and evaluation.
To test other methods than DTF, you will need to clone
their repositories:
 - [RAFT](https://github.com/princeton-vl/RAFT.git)
 - [FlowFormer](https://github.com/drinkingcoder/FlowFormer-Official)
 - [PIPs](https://github.com/aharley/pips)
 - [PIPs++](https://github.com/aharley/pips2.git)
 - [MFT](https://github.com/serycjon/MFT.git)
 - [CoTracker](https://github.com/facebookresearch/co-tracker)

Most of the evaluations have used the `evaluate_all.sh` script
(adapt paths to your environment).
Videos can be generated using the `visu.sh` script.

## Acknowledgements

Some parts of this code come from
the [RAFT](https://github.com/princeton-vl/RAFT.git) repository.
Dataset generation relies on [Kubric](https://github.com/google-research/kubric).
Evaluation relies on the [TAPNet](https://github.com/google-deepmind/tapnet) benchmark.

### Other related works

 - [FlowFormer](https://github.com/drinkingcoder/FlowFormer-Official)
 - [PIPs](https://github.com/aharley/pips)
 - [PIPs++](https://github.com/aharley/pips2.git)
 - [MFT](https://github.com/serycjon/MFT.git)
 - [CoTracker](https://github.com/facebookresearch/co-tracker)

## BibTeX

```
@InProceedings{Tournadre_2024_DTF,
  author    = {Tournadre, Marc and Soladi\'e, Catherine and Stoiber, Nicolas and Richard, Pierre-Yves},
  title     = {Dense Trajectory Fields: Consistent and Efficient Spatio-Temporal Pixel Tracking},
  booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
  month     = {December},
  year      = {2024},
  pages     = {2212-2230}
}
```

