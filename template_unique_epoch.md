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
|        | Undefitting | Good fitting | Overfitting |
|:------:|:-----------:|:------------:|:-----------:|
| Status | Yes | Partially | No |

## Prediction

|        | Seq. to predict | Expected seq. | Predicted seq. |
|:------:|:---------------:|:-------------:|:--------------:|
|        | <authors author:'John Du'/> | author: 'John Du' | author: 'Mr. John Da'|


## Bilan
For 32 neurons in GRU and 10 epochs, the training accuracy is gapped at 0.67, the training loss at 0.67. The model is underfitted and the prediction is correct at 53%.

|             Metric            | In vocab | Out vocab |
|:-----------------------------:|:---------------:|:-------------:|
| Total nb. of words            | 32054 | 23345 | 
| Nb. of unique words           | 3090 | 3078 | 
| Max seq. length               | 9    | 8 |
| Vocab size                    | 3089 | 3077 | 
| % True Negatives              | 24.1 | 24.1 |
| % authors names without title | None | None |