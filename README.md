# DataMining_WorldBank_Project (In Work)
 DataMining Project 1

Team3 final project report (GPboost)
Team members: Akshay Verma, Yuchuan Feng, Chupeng Yue

Data


We are using data provided by the World Bank Group. We have imported the data using the wbg API provided by the World Bank.


We are using data from 58 countries, 
"Albania", "Argentina", "Armenia", "Australia", "Austria", "Azerbaijan", "Belarus", "Belgium", "Bosnia and Herzegovina", "Bulgaria", "Canada", "Chile", "China", "Croatia", "Cyprus",  "Czech Republic", "Denmark", "Estonia", "Finland", 	"France", "Georgia", "Germany", "Greece", "Hungary", "Iceland", "India",  "Ireland", "Israel",  "Italy", "Japan", "Kazakhstan", "Korea, Rep.",   "Latvia",   "Lithuania", "Luxembourg", "Mexico",   "Moldova", "Montenegro", "Netherlands", "North Macedonia", "Norway", "Poland", "Portugal", "Romania", "Russian Federation",   "Serbia",  "Singapore",   "Slovak Republic", "Slovenia", "South Africa", "Spain", "Sweden",  "Switzerland", "Turkey",  "Turkiye", "Ukraine", "United Kingdom", and "United States".

And have selected 7 macroeconomic indexes  GDP PPP,  Research and development expenditure,  Renewable Energy Consumption, Urban Population, Medium and high-tech manufacturing value-added, &  Trade(% of GDP), for each country measured from 1994-2021

Our response is CO2 emission per capita, and all other indexes are covariates.  

Data Cleaning

There was too many NAs in the last two years, so we dropped them. The observations in these three countries, [“ALB”,” CHE”,” MNE”], are inadequate, so we dropped them as well.

Then we did Log transformation for this data frame to transform skewed data to approximately conform to normality

Finally we get our main data frame with 55 countries, 6 covariates, and 1 response, measured from 1994-2019.


Split Training and Testing Data

We have taken the last year, 2019, of each country as the testing data. The rest is taken as Training Data.





GPboost Introduction
Model Introduction

GPboost model combines boosting with the Gaussian process and mixed effects models. This allows for relaxing, first, the zero or linearity assumption for the prior mean function in the Gaussian process and grouped random effects models in a flexible non-parametric way and, second, the independence assumption made in most boosting algorithms. The former is better for model misspecification prevention and forecast accuracy. The latter is crucial for producing probabilistic predictions and for learning the fixed effects predictor function effectively.


Tree-boosting with its well-known implementations such as XGBoost, and CatBoost, is widely used in applied data science. Besides state-of-the-art predictive accuracy, tree-boosting has the following advantages:

Automatic modeling of non-linearities, discontinuities, and complex high-order interactions
Robust to outliers in and multicollinearity among predictor variables
Scale-invariance to monotone transformations of the predictor variables
Automatic handling of missing values in predictor variables

Mixed effects models are a modeling approach for clustered, grouped, longitudinal, or panel data. Among other things, they have the advantage that they allow for more efficient learning of the chosen model for the regression function.

combined gradient tree-boosting and mixed effects models often perform better than (i) plain vanilla gradient boosting, (ii) standard linear mixed effects models, and (iii) alternative approaches for combing machine learning or statistical models with mixed effects models.

Modeling grouped data

Grouped data occurs naturally in many applications when there are multiple measurements for different units of a variable of interest. Examples:

Investigate the impact of some factors (e.g. learning technique, nutrition, sleep, etc.) on students’ test scores and every student does several tests. In this case, the units, i.e. the grouping variable, are the students and the variable of interest is the test score.

A company gathers transaction data about its customers. For every customer, there are several transactions. The units are then the customers and the variable of interest can be any attribute of the transactions such as prices.


Such grouped data can be modeled using four different approaches:

Ignore the grouping structure. This is rarely a good idea since important information is neglected.
Model each group separately. This is also rarely a good idea as the number of measurements per group is often small relative to the number of different groups.
Include the grouping variable (e.g. student or customer ID) in your model of choice and treat it as a categorical variable. While this is a viable approach, there are the following disadvantages. Often, the number of measurements per group is relatively small and the number of different groups or clusters is large. In this situation, the model must learn numerous parameters based on scant data, which could lead to ineffective learning. High cardinality categorical variables can also be challenging for trees.
Model the grouping variable using so-called random effects in a mixed-effects model. This is often a sensible compromise between the approaches above, which is beneficial in the case of tree-boosting.
Methodological Background

For the GPBoost algorithm, it is assumed that the response variable y is the sum of a non-linear mean function F(X) and so-called random effects Zb:

y = F(X)+ Zb + e,
where y is the response variable; X contains the predictor variables and F() is a potentially non-linear function. In linear mixed effects models, this is simply a linear function. In the GPBoost algorithm, this is an ensemble of trees; Zb is the random effects that are assumed to follow a multivariate normal distribution; e is an error term

The model is trained using the GPBoost algorithm, where training means learning the hyper-parameters of the random effects and the regression function F(X) using a tree ensemble. After the model has been trained, the random effects of Zb can be estimated. In a nutshell, the GPBoost algorithm is a boosting technique that adds a tree to the ensemble of trees using a gradient and a Newton boosting step while iteratively learning hyper-parameters. The main difference to existing boosting algorithms is that it accounts for dependency among the data due to clustering and, second, it learns the (co-)variance of the random effects. Fisher scoring or (accelerated) gradient descent can be used to learn (co-)variance parameters in the GPBoost library, and the LightGBM library is used to train trees. This indicates explicitly that LightGBM's entire functionality is available.

LightGBM
Introduction
LightGBM is a gradient-boosting framework that uses tree-based learning algorithms. In terms of tree-boosting, there is no difference between LightGBM and GPB. So we use LightGBM to verify whether the Gaussian process added by GPboosting is valuable, and how to improve the traditional gradient boosting.
Results
 Hyperparameters:

Objective
Regression
Learning Rate
 0.05
num_leaves
20
Min Data in Leaf
3
N_estimator
2258

In this step, we use blendsearch to optimize hyperparameters. BlendSearch combines local search with global search. It leverages the frugality of local search like CFO and the space exploration ability of global search like Bayesian optimization. This makes it possible to combine the advantages of both approaches to achieve the best results while saving resources.
Tree model：


The Root Mean Squared Error value for this model is 0.21.
Predicted value fitted graph：


Scattered graph of predicted value
Judging from the picture and the RMSE value, this model fits well. We will test the GPboost model below to see if the results are improved after adding the Gaussian process.
Of course, we also obtained the importance of features in this step. Important features affecting carbon dioxide emissions were identified.

From the figure, it can be concluded that renewable energy and R&D are the two factors that have the greatest impact on carbon dioxide emissions. In terms of impact, renewable energy is negatively correlated, and R&D is positively correlated. Since feature importance is calculated using the cutting method, shape value is calculated using game theory. So there are some small deviations in the results, but none of this affects the interpretability of the model.






GPboost Results
Our code is based on https://github.com/fabsig/GPBoost

We are using ‘Economy’ & ‘Years’ as mixed effects. First, we are going to train a model with untuned hyperparameters, then we are gonna find out the optimal hyperparameters through cross-validation, and then we are going to train an Auto-Regressive Model. We will compare the prediction accuracy of all three models and see how the factor importance changes in different models.

Untuned Model


For the first model we are using the following, untuned, hyperparameters.


Objective
Regression_L2
Learning Rate
0.1
Max Depth
5
Min Data in Leaf
5
N_estimator
400



The Root Mean Squared Error value for this model is 1.87.

Below are the two prediction plots for this model. In the plot on the left-hand side, the blue line is the actual test value while the orange line is the predicted value. 
On left we have a scatter plot between the true and predicted values, you can see that there is a lot of deviance from the regression line. 







Now, we will see this decision tree that this model uses for predictions. As you can see the tree is quite large, which is largely due to our choice of hyperparameters. In further models, you will see that with optimal hyperparameters the tree gets pruned a lot and is more intelligible.






Below is the Feature importance plot, Renewable Energy Consumption has the highest importance in our model, this is followed by GDP PPP, R&D expenditure & Trade, and the least important factor in our model is Urban Population. 

















To see how exactly the values of these factors affect our response we will have to look at the SHAP values (SHapley Additive exPlanations) plot.




The SHAP Value plot for this model is convoluted so it’s hard to clearly say which factors are directly or indirectly correlated. We can see that high renewable energy consumption values have a negative effect on our model, hence our response. Hopefully, in our further models, we can see a clearer trend from our SHAP plot.
Tuned Model

Now we tune our hyperparameters. We are using grid-search cross-validation to find the optimal hyperparameters. Our N-Fold value is 5.

The optimal hyperparameters are given below. The root mean square value for this model is 0.12. This is a great improvement from our previous model.

Objective
Regression_L2
Learning Rate
0.0001
Max Depth
4
Min Data in Leaf
8
N_estimator
907

















Below are our two prediction plots. We can see that in the plot on right, the predicted value line fits much closer to the actual value line, and in the plot on the left, the deviancy from the regression line is a lot lesser than in the previous model's plot.








Now we see the decision tree for this model. We can see that this tree is much smaller and pruned than the previous tree. 














Looking at the feature importance in this model, we see that renewable energy consumption is still the most important factor but the other features have changed their place. Urban Population % is no longer the least important feature but instead is the third most important feature. Trade, which was the third most important feature in the previous model is not the least important feature.









Looking at the SHAP value plot, we can see a huge difference from the previous model's plot. We can clearly see that renewable energy consumption is inversely correlated with our response. We can also see a clear demarcation between high and low values for the medium & high tech manufacturing value-added features in this plot.






Auto-Regressive Model with tuned Parameters 


Now we are going to see the results of an autoregressive model with tuned hyperparameters.   
The Root Mean Squared Error value for this model is 0.08, which is significantly lower than the previous models.


We can see in the prediction plots below that the predicted value fits much closer to the actual values.






The most important feature has remained the same in all three models, but the ranking of the other features has been changing. For the AR model, trade is the second most important feature while the urban population is the least important feature. 










In the SHAP value distribution plot, we can see that renewable energy consumption is inversely correlated with our response. We also notice that some high GDP value observations have a great positive impact on our response. This is not to say that we can conclude that GDP PPP is directly correlated to our response through this plot.







The decision tree for this model is below. We notice that this is even smaller than the previous tree and is a lot more pruned. We can also see that the root node of this decision tree is different from the previous trees. Instead of GDP PPP, this tree starts with R&D expenditure.













Conclusion

We have used GPboost to build a tree mode for such data with multiple groups and tuned the hyperparameters by cross-validation with 5 folds. It is shown that GPboost has better and more accurate predictive prowess than a similar but simpler LightGBM because the full functionality of LightGBM is available in the GPboost library. The autoregressive model performs better than all the other models, with a significantly lower root mean squared value. At the same time, there is still a flaw in this test process, that is, we removed a lot of missing data in the last year during data processing. The amount of data is reduced, so that the model may have the risk of overfitting. Missing values can be filled by regression through the random forest because prior probability and posterior probability can be converted to each other according to the Bayesian formula. At the same time, the judgment of feature importance is also a little flawed. Model interpretability does not imply causation. Neither the shape value nor the feature importance based on the segmentation method provides causality. We may be able to explain the reality of improving the feature through the Markov Boundary.
