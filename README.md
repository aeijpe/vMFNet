# vMFNet: Compositionality Meets Domain-generalised Segmentation
Large parts of the code are directly taken from the [original repo](https://github.com/vios-s/vMFNet/tree/main).

Some small adjustments are made in the code to account for our dataset and to have the same metrics as out proposed method for evaluation. 
This implementation is used as baseline for the project with [this repo](https://github.com/aeijpe/CrossModal-DRL).
Please refer to the README in the baselines folder there to understand how to run this baseline.


## Code structure

- In `checkpoints`, all checkpoints to the vMFNet model for the different folds are stored.
- `composition` contains all classes and functions for the compositional layer.
- `models` contains all classes to make the vMFNet model.


# License
All scripts in the [original repo](https://github.com/vios-s/vMFNet/tree/main) are released under the MIT License. Therefore the same applies to this code. 
