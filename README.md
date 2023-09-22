# SparePart_Predictor
![alt text](https://github.com/harunakcay/SparePart_Predictor/blob/main/app/static/robot.png) 
## Spare Part Predictor is a smart assistant for after-sales customer services that can make predictions for possible defective parts causing the error in products.
After-sales customer services often fail to detect the component causing the problem of their customers' products. 
Research shows that the probability of services detecting the defective part of a product based on its fault report is 40%, meaning there occurs average of 2.5 visits per any malfunction. 
This inefficiency does not only lead to increasing sales operational costs, but it also results in unsatisfied customers and hence tarnishing reputations of companies. 
In order to combat this issue, we propose you Spare Part Predictor: a smart model that can anticipate the problematic part of a product given its fault report. 
The model reads the entries of the fault report and suggests two possible spare parts that the faulty product may need, in only few seconds. 
Test results show that the model is able to predict the correct spare part in its first attempt with 68% probability, decreasing the average number of visits per malfunction to 1.5 visits/malfunction.
In addition, when the model's secondary prediction is also considered, the accuracy becomes more than 85. 
To sum up, Spare Part Predictor aims to provide a fast and good quality service to customers having problems with thier products.

# How Does The Predictor Work?

The Predictor was trained with a dataset containing 3500 fault reports. Each attribute of the dataset was analyzed and its correlation with the corresponding spare part is examined. 
Based on the results, it was observed that the defective part of a product has a strong correlation with the product's detailed expression (ZPRDHYR8), 
product's code (ZCRMPRD) and the customer's complaint about the product (ZSIKAYET). Additionally, relatively weaker but not neglectable correlations with product's usage duration (USAGE), 
production date (PRD_YEAR), cost group (ZRPRGRP), price group (PRICE_GRP), brand (ZZMARKA) and product type (ZURNTIP) were seen. 
Using these attributes, an XGB Classifier model was trained and tuned with the optimal hyperparameters. 
Eventually, when a user enters the desired information of a fault report, the Predictor makes a prediction based on its trained model. 
The prediction is displayed on interface of customer service, and the customer's problem is hopefully solved within one service visit.

- For more information about the model or project itself:
  - see https://colab.research.google.com/drive/1P5gy1lcj_DZ28hO6OF_s0eh87OIMLRSo?usp=sharing

# Deployment

1. Download the content and run it.
2. Install Flask
```
!pip install flask
```
4. Run the following code in terminal:
```
flask run
```
3. Fill the boxes with intended info
4. See the Predictions
