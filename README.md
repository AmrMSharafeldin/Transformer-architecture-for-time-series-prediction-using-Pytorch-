# Transformer-architecture-for-time-series-prediction-using-Pytorch-

 # **Introduction**
 **Overview**
Predicting stock market behavior represents a serious challenge for financial analysts due to the random nature of stock market features. These features are often unpredictable and difficult to generalize, making accurate modeling of stock market behavior an extremely difficult problem.

The category under which this problem lies is referred to as technical analysis, which involves the interpretation of price actions in the stock market. This type of analysis focuses primarily on determining the likelihood of a trend continuing or reversing. Technical analysis aims to reflect all available information that could affect the behavior of the trend and assumes that prices tend to follow the same past tendencies. 


# **Architecture**

The proposed architecture is a hybrid of an encoder block adapted from transformers and 1D convolution layers. 
Encoder Layer:
The Core of the network is an encoder block that is primarily composed of a multi-head self-attention layer followed by 2 convolution layers. The multi-head layer is designed to create a contextually aware vector of shape (sequence_lenght, num_afeatures) for each sentence. Then Conv1D increases the dimensionality of the num_features to detect more complex features of the contextual aware vector. Normalization, dropouts, and residual connect layers are added to retain the stability of the backpropagation 
Transformers:
The transformer block stacks N encoder blocks to obtain contextual awareness from higher to lower features. Then the outputs of this operation through map layers to increase the learnability of the model and also to match the outputs to the desired output shape (examples, output sequence length) 

It’s important to note that the input and ground truth sequences are normalized before feedforward. Then the inverse transformation is applied to the predictions for plotting and computing the Relative error, to be later used in Model Performance Comparison. Also do to excessive ram usage by the model. It wasn’t possible to run feedforward on the full test data locally, Thus  the analysis.csv for this model was computed using Google Collab

HyperParameters and Loss Function :
"batch_size":32,"epochs" :25,"lr":0.001, Loss function used is MAE



# **Running the Main Script with Command-Line Flags**
## Running Modes:
* --train: Run the script in training mode. This flag allows you to retrain the model.
Generates model.tbh
* --test: Run the script in testing mode. This flag allows you to evaluate the model on a test dataset.
* --analyze: Run the script in analysis mode. This flag allows you to analyze the performance of the model.
Generates Model-analysis.csv that contains the average error for each entry in the test data 
* --viz: Run the script with visualization. This flag plots how the evaluation of the model improves as the model learns.
Generates a plot in model/visualizations



## Usage:
**To run the main script with desired flags, execute the following command:**

```
python3 main.py [--train] [--test] [--analyze] [--viz] 
```
### **Examples**:

**To train the model and visualize the performance:**
```
python3 main.py --train --viz
```

**To run the Transformer model on an evaluation dataset:**
```
python3 main.py --test
```
