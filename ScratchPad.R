# http://groupware.les.inf.puc-rio.br/har
# https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv
# https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

# Six young health participants were asked to perform one set of 10 repetitions of the 
# Unilateral Dumbbell Biceps Curl in five different fashions: 
#   exactly according to the specification (Class A), 
#   throwing the elbows to the front (Class B), 
#   lifting the dumbbell only halfway (Class C), 
#   lowering the dumbbell only halfway (Class D) and 
#   throwing the hips to the front (Class E).

# Class A corresponds to the specified execution of the exercise, while the other 4 
# classes correspond to common mistakes. 

raw_train <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")
raw_eval <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv")

# Before stripping into train, test and validate, make sure we strip out columns that have > 70% NA values
na_pct <- apply(raw_train, 2, function(x) sum(is.na(x) | x == '' ) / length(x))
incl_cols <- na_pct < 0.3
raw_train <- raw_train[,incl_cols]   
raw_train <- raw_train[,-c(1,3:5)]                 # need to remove 'X' and all timestamps
raw_eval <- raw_eval[,incl_cols]
raw_eval <- raw_eval[,-c(1,3:5)]                   # need to remove 'X' and all timestamps    
# This reduces the number of features from 159 to 55 (so removed 104 immediately)

# Now we create our train, test and validation datasets from reduced datasets
library(caret)
set.seed(2903)
inTrain <- createDataPartition(y=raw_train$classe, p = 0.6, list = FALSE)
df_train <- raw_train[inTrain,]
df_test <- raw_train[-inTrain,]
 
set.seed(2903)
inTest <- createDataPartition(y = df_test$classe, p = 0.5, list = FALSE)
df_val <- df_test[-inTest,]
df_test <- df_test[inTest,]

# Centre and scale numeric data
# Note that we don't need to worry about our classe variable as this is categorical
# preProcess_model <- preProcess(df_train, method = c('center','scale'))
# df_train <- predict(preProcess_model, newdata = df_train)

# Do one-hot-encoding to transform categorical variables
train_y <- df_train$classe
ohe_model <- dummyVars(classe ~ ., data = df_train)
df_train <- data.frame(predict(ohe_model, newdata = df_train))
df_train$classe <- train_y

# View data and possible predictors
# featurePlot(x = df_train[,-62], 
featurePlot(x = df_train[, c(9,10,12,13,18,19,30,33:39,43:51,56,59:61)], 
            y = df_train$classe, 
            plot = "pairs",
            strip=strip.custom(par.strip.text=list(cex=.7)),
            scales = list(x = list(relation="free"), 
                          y = list(relation="free")))

featurePlot(x = df_train[, -62], 
            y = df_train$classe, 
            plot = "density",
            strip=strip.custom(par.strip.text=list(cex=.7)),
            scales = list(x = list(relation="free"), 
                          y = list(relation="free")))

# If we look at the data, there is a lot of highly correlated variables
M <- abs(cor(df_train[,-62]))
diag(M) <- 0
length(which(M > 0.8, arr.ind = T)) /2

# So high correlation between the variables
# Indicates that PCA is probably a good idea

preProcPCA <- preProcess(df_train,method='pca', pcaComp=10)
pca_train <- predict(preProcPCA, newdata = df_train[,-62])
pca_train$classe <- train_y


# Prepare Test and Validation datasets
test_y <- df_test$classe
df_test <- data.frame(predict(ohe_model, newdata = df_test))
pca_test <- predict(preProcPCA, newdata = df_test)
pca_test$classe <- test_y

val_y <- df_val$classe
df_val <- data.frame(predict(ohe_model, newdata = df_val))
pca_val <- predict(preProcPCA, newdata = df_val)
pca_val$classe <- val_y

# Look to tune cross validation parameters
mod_trControl <- trainControl(method = "cv", number = 5)

# Try Model Based approach - Naive Bayes
# Not expecting miracles as not a large number of binary (categorical) variables
set.seed(2903)
modFitnb <- train(classe ~ .
                  , data = pcat_train
                  , method = 'nb'
                  , trControl = mod_trControl
                  , tuneLength = 5)
pred_test_nbt <- predict(modFitnb,newdata = pcat_test)
confusionMatrix(pred_test_nbt, test_y)                            # Accuracy = 46.95%

# Try SVM
set.seed(2903)
modFitsvm <- train(classe ~ .
                   , data = pca_train
                   , method = 'svmRadial'
                   , trControl = mod_trControl
                   , tuneLength = 5)

pred_test_svm <- predict(modFitsvm, newdata = pca_test)
confusionMatrix(pred_test_svm, test_y)                            # Accuracy = 66.76%


# Now try Trees with Boosting (stochastic gradient boosting)
# Expect this to do better
set.seed(2903)
modFitgbm <- train(classe ~ .
                   , data = pca_train
                   , method = 'gbm'
                   , trControl = mod_trControl
                   , tuneLength = 5
                   , verbose = FALSE)
pred_test_gbm <- predict(modFitgbm,newdata = pca_test)
confusionMatrix(pred_test_gbm, test_y)                            # Accuracy = 81.21%


# Fit tree ensemble model (random forest)
# Expect this to do better again
set.seed(2903)
modFitrf <- train(classe ~ .
                  , data = pca_train
                  , method = 'rf'
                  , trControl = mod_trControl
                  , tuneLength = 5)

pred_test_rf <- predict(modFitrf,newdata = pca_test)
confusionMatrix(pred_test_rf, test_y)                            # Accuracy = 93.37%


# Try XGBoost
# Expect this to do pretty well
set.seed(2903)
modFitxgb <- train(classe ~ .
                   , data = pca_train
                   , method = 'xgbTree'
                   , trControl = mod_trControl)

pred_test_xgb <- predict(modFitxgb,newdata = pca_test)
confusionMatrix(pred_test_xgb, test_y)                            # Accuracy = 81.32%

set.seed(2903)
modFitrf <- train(classe ~ .
                  , data = pca_train
                  , method = 'rf'
                  , trControl = mod_trControl
                  , tuneLength = 5)

pred_test_rf <- predict(modFitrf,newdata = pca_test)
cm_rf <- confusionMatrix(pred_test_rf, test_y) 











