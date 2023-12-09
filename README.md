# Random_Forests
** from sklearn.ensemble import RandomForestClassifier

### Summary:


In machine learning, a "Random Forest" is like a large group of these friends, where each friend (or small decision tree) looks at the data (like restaurant options) and makes a decision (or prediction). However, instead of each friend deciding on their own, they all vote, and the restaurant with the most votes is chosen.

So, a Random Forest is a collection of many decision trees. Each tree makes its own prediction, and the final output of the Random Forest is decided based on the majority vote of all these trees for Classification & the average for Regression. This makes the Random Forest a strong and reliable method in machine learning because it combines the decisions of many different models, reducing the chance of making a poor decision based on just one model's view.

Therefore in this example:

| Who & What                  | Random Forest     |
|-----------------------------|-------------------|
| Each Person                 | Decision Tree     |
| Restaurants                 | Predictions       |








---



### How to Tune:


If we compare Random Forest to a single Decision Tree, there are several key hyperparameters that are different or unique to Random Forests. These hyperparameters help in controlling the behavior of the Random Forest model.

 **Number of Trees (n_estimators -> Default = 100):**
    This is perhaps the most important hyperparameter for a Random Forest. It specifies the number of trees in the forest. More trees usually mean better performance but also longer training time and chance of overfitting grows.

**Maximum Features (max_features  --> Default = 'auto'):**
   This parameter determines the maximum number of features that are considered for splitting a node. It can be set as a number, a percentage, or different heuristics like 'sqrt' or 'log2'. In a single Decision Tree,  usually, all features are considered for splitting a node, but in a Random Forest, limiting the number of features can lead to more diverse trees and reduce overfitting.
Recommend: Start with sqrt(number of features) then use a grid search for other possible values

**Bootstrap Samples[data rows] (bootstrap -> Default =TRUE):**
     This parameter decides whether or not to use bootstrap sampling when building trees. Bootstrap sampling means randomly selecting a subset of the data(rows from the data) with replacement for training each tree. This means, we are taking a subset of the features & a subset of the rows of data AKA Bootstrapped. This helps reduce correlation betwen trees because each tree is trained on a different subset rows of data & features which will likely better generalize.

**Out-of-Bag Error (oob_score -> Default =FALSE):**
     This is a method for estimating the generalization accuracy of the Random Forest. It uses the bootstrap samples not included in the training of each tree (the 'out-of-bag' samples) to estimate the model's performance. This is unique to     Random Forests and isn't a concept in a single Decision Tree. This will not impact the trees and only provides a way to measure the performance of the trees on the untrained data set similar to train/test split.

**Minimum Samples for Splitting (min_samples_split -> Default =2):**
    While this is also a hyperparameter for Decision Trees, it often plays a more crucial role in Random Forests because it affects each tree in the forest and thus has a compounded effect. This means that a node will be split if it contains 2 or more samples.

**Minimum Samples for a Leaf Node (min_samples_leaf -> Default =1):**
    This setting allows each leaf node to have as few as 1 sample.
 

---



### Data:


This bank customers churn dataset can be found at [Kaggle Bank Customer Churn data set](https://www.kaggle.com/mathchi/churn-for-bank-customers). It contains customers and their characteristics as well as if they churned. Each observation represents a unique customer and information such as age, gender, name, location, tenure, balance and many more. To access and view a detailed description of the dataset, [CLICK HERE](https://www.kaggle.com/mathchi/churn-for-bank-customers)


#### Data Download From Kaggle [CSV Data](https://www.kaggle.com/mathchi/churn-for-bank-customers)



---




### Algorithm & Tools:


#### **Measuring Metric:**


      F2_Beta was used because it puts more attention on minimizing false negatives than minimizing false positives. 



#### **Model Testing:**

     The data was split using stratified train/test with 10 K-Folds. 


#### The Algorithms used for this analysis include:
- XGB 
- AdaBoost 
- RandomForest 
- ExtraTrees 
- Bagging 
- DecisionTree 
- LogisticRegressionCV
- KNeighbors 
- SVC
- Bernoulli
- Gaussian



#### **The top 5 model results:**

| Machine Learning Algorithm  | F_Beta 2 Score    |
|-----------------------------|-------------------|
| XGB                         | 52.09155          |
| Decision Tree               | 49.85373          |
| Gradient Boosting           | 49.75858          |
| Random Forest               | 49.01696          |
| AdaBoost                    | 48.94638          |




#### **TOOLS:

The following tools were used in this project:
1.	Python & Pandas to: 
                  •	        Clean & Explore
                  •	      Feature Engineering 
                  
                  
2.	SKLearn to implement various classification models.
3.	Matplotlib and Seaborn to visualize the data and model outputs...
