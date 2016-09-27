###############LR1###################

#Boosting GBM - LASSO  - BOOST
library(lattice)
library(ggplot2)
library(caret)
library(car)
library(randomForest)
set.seed(1234)

train <- read.csv("G:\\training_set_q1.csv")
test <- read.csv("G:\\test_set_q1_upload.csv")

dim(train)
dim(test)

names(train)

#Clean Data
trainG1 <- subset(train,select = -c(G2,G3))
trainG2 <- subset(train,select = -c(G1,G3))
trainG3 <- subset(train,select = -c(G1,G2))
#Data Slicing  
inTrain <- createDataPartition(y = trainG1$G1,p=0.75,list = FALSE)
training <- trainG1[inTrain,]
testing <- trainG1[-inTrain,]

#Modal Fit 
#modFit <- train(G1~.,methods = "gbm",data = training,verbose = FALSE)
modFitL1 <- train(G1~.,methods = "lasso",data = training,verbose = FALSE)
print(modFitL1)

#Plots 
qplot(predict(modFitL1,testing),G1,data = testing)

#pred <- predict(modFit,testing)
predL1 <- predict(modFitL1,testing)
rmse <- mean((testing$G1-predL1)^2)
print(rmse)
############################
inTrain <- createDataPartition(y = trainG2$G2,p=0.75,list = FALSE)
training <- trainG2[inTrain,]
testing <- trainG2[-inTrain,]

#modFitGBM2 <- train(G2~.,methods = "gbm",data = training,verbose = FALSE)
modFitL2 <- train(G2~.,methods = "lasso",data = training,verbose = FALSE)
print(modFitL2)

qplot(predict(modFitL2,testing),G2,data = testing)

#pred <- predict(modFit,testing)
predL2 <- predict(modFitL2,testing)
rmse <- mean((testing$G2-predL2)^2)
print(rmse)
##########################
inTrain <- createDataPartition(y = trainG3$G3,p=0.75,list = FALSE)
training <- trainG3[inTrain,]
testing <- trainG3[-inTrain,]

modFitL3 <- train(G3~.,methods = "lasso",data = training,verbose = FALSE)
print(modFitL3)

qplot(predict(modFitL3,testing),G3,data = testing)

#pred      <- predict(modFit,testing)
predL3 <- predict(modFitL3,testing)
rmse <- mean((testing$G3-predL3)^2)
print(rmse)

##Predicting the test set

test.org <- read.csv("G:\\test_set_q1_upload.csv")
pred <- predict(modFitL1,test.org)
test$G1 <- pred
pred <- predict(modFitL2,test.org)
test$G2 <- pred
pred <- predict(modFitL3,test.org)
test$G3 <- pred
dim(test)
View(test)
