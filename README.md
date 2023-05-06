# Stock Prediction 

---

## Objective:
To use Machine learning models to make stock market predictions of an unstable stock like bitcoin. For which we make use of the following  approaches. 
* ARIMA
* Moving Averages (MA)
* CNN
* Transformers 
The goal is to optimize and compare performance of above models for limited and volatile data. Also to finally end up with a model that is a relatively fair predictor of bitcoin stock price.

[//]: # (Image References)

[image1]: ./output_images/image1.png "Transformer Output"
[image2]: ./output_images/image2.png "Transformer Error"
[image3]: ./output_images/image3.png "Moving Average Table"
[image4]: ./output_images/image4.png "Moving average graph"
[image5]: ./output_images/image5.png "Moving average error"
[image6]: ./output_images/image6.png "CNN model error"
[image7]: ./output_images/image7.png "ARIMA-Model"
[image8]: ./output_images/image8.png "ensamble-model"

## Background:

There is significant literature available regarding this subject, with approaches varying from statistical metrics to current Deep Learning models using CNN.
Initially, stock-related sentiments are collected from Twitter, then pre-processing steps are performed on the collected data.Then, classical algorithms like linear regression, Random Walk Theory (RWT), Moving Average Convergence / Divergence (MACD) (Chong & Ng, 2008) and also using some linear models like Autoregressive Moving Average (ARMA), Autoregressive Integrated Moving Average (ARIMA) (Zhang, 2003) have been used for predicting stock prices. Ruan et al (Ruan et al., 2018) proposed a Linear Regression model to perform a comparative analysis of the relation between people's sentiments expressed and the financial stock market using microblogs. In more recent times, machine learning techniques involving Support Vector Machine (SVM), Random Forest (RF) have been used. Mei et al. employed RF for accurately forecasting real time prices on the New York electricity market (Mei et al., 2014).
Currently, deep Neural modeling architectures have performed significantly better with attention mechanisms in the aspect of predicting stock market prices. Neural networks based approaches such as Artificial Neural Network (ANN), Convolutional Neural Network (CNN), Recurrent Neural Network (RNN) and deep neural networks like Long Short Term Memory (LSTM) also have shown promising results (Li et al., 2017) (Oyeyemi et al., 2007). LSTM-attention models implemented by Zhang and Zhang (Zhang & Zhang, 2020) by filtering out the high-value information from the large dataset, helps give attention to the features at different locations to capture the influence of the time period. Zhang et al. (Zhang et al., 2017) presented a LSTM based model to analyze and evaluate the impact of social media and the internet on the financial market. 
Transformer with a causal convolutional network for feature extraction in combination with masked self-attention was employed by Wallbridge (Wallbridge, 2020) for the prediction of price movements from limit order books. The implemented model updated its learned features based on relevant contextual information from the data with a causal and relational inductive bias. With hyperparameter fine tuning, the algorithm is efficient at performing on very large datasets with specified window size. Apart from all the highlighted works, there is much more literature regarding this subject.

In spite of all the papers listed as reference, what makes this project different is the volatility of the bitcoin stock and the limited data we have to make such a prediction. We have researched the best strategy to go about the experiment and settled on a unique transformer approach along with Moving Average (MA), CNN and ARIMA approaches. It is an effort that may reveal an insight into how to improve on the methods of predicting volatile data.

## Results:

### Transformer:
Here we follow the steps specified in the methods section and obtained a linear prediction for a transformer. As shown below in Figures 5 and 6, even the most advanced model architectures are not able to extract non-linear stock predictions from historical stock prices and volumes. Also the distance between the MAPE and other performance graphs is too high. Where during the run of the model exponential error is too high to minimize.
 
![alt_text][image1]
![alt_text][image2]

### Moving average:
For this part of the experiment we took the linear values from the transformer and applied a simple moving average smoothing effect on the data (window size=10). The model was able to provide significantly better predictions (green line) and better performance as seen in the graphs below, Figure 7 and 8. The model yields a MAE of 0.0275 on the test set.
Table 1 displays the metrics generated while training and testing.

![alt_text][image3]
![alt_text][image4]
![alt_text][image5]

### CNN-Model:
The CNN model was trained for 100 epochs with a batch size of 8 and yielded the following results. The results were obtained on the scaled data. 
As seen from Table 1, while R2 value on the training set is quite high, that for the Test set is not as good, implying that the CNN model was not able to capture all the underlying features leading to the stock price fluctuations.

![alt_text][image6]

### ARIMA-Model:
After training the ARIMA model on the training data, it was used to make predictions on the test data to see how well the model performs on the test data.

![alt_text][image7]

The plot in figure 10 represents the predicted bitcoin prices from the ARIMA model vs the actual bitcoin closing prices. As shown in Table 2, the ARIMA model yielded an R2 score(Coefficient of determination) of 0.8552 and a MAE of 2.063 meaning the model was performing well on the test data.

### Ensemble Model:
Once we had all individual model results we proceeded to use an ensemble technique called stacking in order to improve the prediction.This method like bagging and boosting makes use of another predictor which is trained on the individual prediction of the train sets of each of the previous models.

The stacked model was trained on the predicted stock prices from the 2 models (the actual prices were obtained by scaling back) against the actual known stock prices using a Linear Regression model. To evaluate the model we used the coefficient of determination (R2 score). The stacked regression model yielded an R2-score of 0.9535. This is better than the other models as shown in Table 3.

![alt_text][image8]

## Conclusion:
In the case of the transformer, even the most advanced model architectures are not able to extract non-linear stock predictions from historical stock prices and volumes. However, when applying a simple moving average smoothing effect on the data (window size=10), the model is able to provide significantly better predictions (green line). Instead of predicting the linear trend of the Bitcoin  stock, the model is able to predict the ups and downs, too. However, when observing carefully you can still see that the model has a large prediction delta on days with extreme daily change rate, hence we can conclude that we still have issues with outliers. 
For the ARIMA model, from the predictions on Figure 7, the ARIMA model performed well when the stock price has seen little change from the previous day but when there is a significant increase or decrease in the stock price, ARIMA model fails to make a good prediction. This shows that the ARIMA model is good for making short-term predictions of the prices but not so good in generalizing and predicting long-term prices.
For the CNN model, as we observed, performance on the testing data was not as good as expected. Since the CNN model itself consists of only a few layers the results are not that disappointing. But when we ensemble the predictions from the CNN model and Transformer with moving average model, the final predictor actually performs significantly better. This implies that the CNN model and the Moving Averages models each learned complimentary features. The CNN model being shallow would have learned the high level features while the other model, being deeper would have learned the more intricate features for price changes. When they are used together, a simple Linear Regression model outperforms all the other models including the ARIMA model while predicting stock prices with an R2 score of 0.9535.
