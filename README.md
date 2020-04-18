# MXGNet
Implementation for MXGNet  [Abstract Diagrammatic Reasoning with Multiplex Graph Networks](https://openreview.net/forum?id=ByxQB1BKwH)

## Dependencies:
Tested in these versions but newer versions might work as well.

* Python == 3.6.1
* Pytorch == 1.2.0
* NumPy == 1.12.1
* SciPy == 0.19.0

## Testing with pretrained model
To test pretrained model (in 'pretrained_models').
For model trained on PGM dataset run: 
```
python test_PGM.py --model-path-name 'pretrained_models/PGM_best.tar' 'PATH-TO-DATA'
```
For model trained on RAVEN dataset run: 
```
python test_PGM.py --model-path-name 'pretrained_models/RAVEN_best.tar' 'PATH-TO-DATA'
```
You can check for validation split result as well with '--valid-result' option. Multiple GPU testing and training can be enabled with '--multi-gpu' option.

## Training:
To train on PGM dataset:
```
python train_PGM.py --save-model --model-save-path 'PATH-TO-SAVE-MODEL' 'PATH-TO-DATA'
```
To train on RAVEN dataset:
```
python train_RAVEN.py --save-model --model-save-path 'PATH-TO-SAVE-MODEL' 'PATH-TO-DATA'
```
The default data loader downsample image to size 80 * 80 on the fly. This can be quite slow. Optionally you can preprocess all data with downsampling to avoid downsampling on the fly, and save multiple data samples in the same '.npz' file to making loading faster. You use 'dataset_8s' class in 'data_utility.py' to load such preprocessed '.npz' files (This loader assume each '.npz' file contain 8 data samples, but you can modify the code for abitrary number of samples).

