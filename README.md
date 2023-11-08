# PortraitNet
Reproduction of paper: "[PortraitNet: Real-time Portrait Segmentation Network for Mobile Device](https://www.sciencedirect.com/science/article/abs/pii/S0097849319300305)"

## Quick Start
1. Download datasets ([link](https://github.com/dong-x16/PortraitNet))
    - EG1800
    - Supervise-Portrait
2. Extract the downloaded dataset into the data folder.
3. Run the `split.ipynb` file to split the dataset into `train`, `validation`, `test` set.
4. Configure experimental parameters in `config.yaml` according to your needs.
5. Run the command: `python train.py`


## Implemented Features
- [x] train code
- [x] inference code
- [x] multiple auxiliary losses
- [x] data augmentation
- [x] wandb
- [ ] multi-gpu parallel training


# Acknowledgement
- [PortraitNet](https://github.com/dong-x16/PortraitNet)
