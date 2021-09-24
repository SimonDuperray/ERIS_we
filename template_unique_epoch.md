# GRU: 32 - Epochs: 10

## Learning curves
![accuracy plot](./analysis/compar_epochs/4350/acc_valacc_10.png)

|          |      Training data      |  Validation data |
|:--------:|:-----------------------:|:----------------:|
| minimum  | 0.27                    | 0.26             |
| maximum  | 0.67                    | 0.69             |
|   mean   | 0.51                    | 0.52             |

![loss plot](./analysis/compar_epochs/4350/loss_valloss_10.png)

|          |      Training data      |  Validation data |
|:--------:|:-----------------------:|:----------------:|
| minimum  | 0.27                    | 0.26             |
| maximum  | 0.67                    | 0.69             |
|   mean   | 0.51                    | 0.52             |

## Fitting observations
* Underfitting: Yes
* Overfitting: No
* Good fitting: Partially

## Prediction
* Sequence to predict: \<authors author:"John Da">
* Expected prediction: author: 'John Da'
* Predicted sequence: author: 'John Bu'

## Bilan
For 32 neurons in GRU and 10 epochs, the training accuracy is gapped at 0.67, the training loss at 0.67. The model is underfitted and the prediction is correct at 53%.