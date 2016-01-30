---
output: pdf_document
---

# Practical Machine Learning: Prediction Assignment Writeup

Sagi Greenstine

## Introduction

Using devices such as *Jawbone Up*, *Nike FuelBand*, and *Fitbit* it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify **how much** of a particular activity they do, but they rarely quantify **how well** they do it.  
In this project, the goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants.  
Six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E). Class A corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes.  
More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).  
The goal of the project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. It is possible to use any of the other variables to predict with. It is necessary to create a report describing how was built the model, how was used cross validation, what the expected out of sample error is, and why were made the choices that were made. The prediction model also should be used to predict 20 different test cases.

## Data Processing

### Loading and viewing the data

#### The "Weight Lifting Exercises Dataset"

**Source:** Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13). Stuttgart, Germany: ACM SIGCHI, 2013.

Loading the dataset:

```{r loaddata}
if(!exists("training")){
        training <- read.csv("./pml-training.csv", row.names = NULL, check.names=F, stringsAsFactors=F, header=T, sep=',', comment.char="", quote='\"', na.strings = c("NA", ""))
}
if(!exists("testing")){
        testing <- read.csv("./pml-testing.csv", row.names = NULL, check.names=F, stringsAsFactors=F, header=T, sep=',', comment.char="", quote='\"', na.strings = c("NA", ""))
}
```

The structure of the data:  
- **training** dataset has 19622 observations of  160 variables;  
- **testing** dataset  has 20 observations of  160 variables.  
I'll try to predict the outcome of the variable **classe** in the *training* dataset.

### Cleaning the data
Firstly, let's remove the columns that filled with the NA values.
```{r remove NAs}
training <- training[, colSums(is.na(training)) == 0]
testing <- testing[, colSums(is.na(testing)) == 0]
```
Now, **training** and **testing** datasets have 19622 and 20 observations (appropriately) of 60 variables.

Also, there are the first seven variables related to the time-series or are not numeric and they are unnecessary for predicting.  
Let's remove these columns.
```{r removeclmns}
trainData <- training[, -c(1:7)]
testData <- testing[, -c(1:7)]
```
Now, **trainData** and **testData** datasets have 19622 and 20 observations (appropriately) of 53 variables, the first 52 variables are the same, and the 53-th variable is *classe* and *problem_id* appropriatelly.

### Slicing the data
Now we can split the cleaned training set into a training set (train, 70%) and a validation set (test, 30%). The validation set will be used to conduct cross validation.

```{r splitting, warning=FALSE, cache=TRUE}
library(caret); library(lattice); library(ggplot2)
set.seed(77777)
inTrain <- createDataPartition(trainData$classe, p=0.70, list=FALSE)
train <- trainData[inTrain, ]
test <- trainData[-inTrain, ]
```

## Predictive Models

We fit a predictive model for activity recognition using Classification Trees and Random Forest algorithms.  

### Classification Trees

We will use 5-fold cross validation when applying the algorithm.

```{r cv, warning=FALSE, cache=TRUE}
library(rpart)
library(rpart.plot)
controlrf <- trainControl(method = "cv", number = 5)
modelrf <- train(classe ~ ., data = train, method = "rpart", trControl = controlrf)
modelrf
```

```{r rpartplot, warning=FALSE, cache=TRUE}
library(rattle)
fancyRpartPlot(modelrf$finalModel)
```

Predicting outcomes with the validation set and viewing the prediction result:

```{r predict, warning=FALSE, cache=TRUE}
predictrp <- predict(modelrf, test)
confrp <- confusionMatrix(test$classe, predictrp)
confrp
```

```{r accuracy}
accurp <- confrp$overall[1]
accurp
```

The accuracy rate is 0.5 approximately, thus the out-of-sample error rate is about 0.5. Therefore, using the Classification Tree doesn't predict the outcome *classe* very well.

### Random Forests
Now, let's try the Random Forests algorithm.

```{r randomforests, warning=FALSE, cache=TRUE}
library(randomForest)
rfit <- train(classe ~ ., data = train, method = "rf",
              trControl = controlrf)
rfit
```

Predicting outcomes with the validation set and viewing the prediction result:

```{r predictrf, warning=FALSE, cache=TRUE}
predictrf <- predict(rfit, test)
confrf <- confusionMatrix(test$classe, predictrf)
confrf
```

```{r accuracyrf}
accurf <- confrf$overall[1]
accurf
```

Here, the Random Forest algorithm is better than Classification Tree algorithm. The accuracy rate is 0.994, thus,the out-of-sample error rate is 0.006.  
Therefore, we will use the Random Forests model to predict the outcome variable *classe* for the testing set.

## Predicting for Test Data Set

```{r predictest, warning=FALSE, cache=TRUE}
predict(rfit, testing)
```

