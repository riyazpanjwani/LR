###################LR2###############################

library(ggplot2)
library(lattice)
library(caret)
library(car)
library(corrplot)
library(Rtsne)
library(xgboost)
library(stats)
library(knitr)

train <- read.csv("C:\\Users\\Aryan\\AppData\\Local\\Temp\\RtmpCyK3CX\\data13a413a12886", header=FALSE)
test <- read.csv("C:\\Users\\Aryan\\AppData\\Local\\Temp\\RtmpCyK3CX\\data13a413fb684a", header=FALSE)

dim(train)
dim(test)

out <- train[,31]
head(out)

train$V31 <- factor(train$V31,
                    levels = c(-1,1),
                    labels = c("0", "1"))
train.org <- read.csv("C:\\Users\\Aryan\\AppData\\Local\\Temp\\RtmpCyK3CX\\data13a413a12886", header=FALSE)

out <- train[,31] == 1
levels(out)
levels(out) <- 1 : length(levels(out))

train$V31 = NULL

cols.without.na <- colSums(is.na(test)) == 0
train <- train[,cols.without.na]
test <- test[,cols.without.na]

zero.var = nearZeroVar(train, saveMetrics=TRUE)
zero.var

corrplot.mixed(cor(train), lower="circle", upper="color", 
               tl.pos="lt", diag="n", order="hclust", hclust.method="complete")
# convert data to matrix
train.matrix <- as.matrix(train)
mode(train.matrix) <- "numeric"
test.matrix <- as.matrix(test)
mode(test.matrix) <- "numeric"
# convert outcome from factor to numeric matrix 
#   xgboost takes multi-labels in [0, numOfClass)
y <- as.matrix(as.integer(out))

param <- list("objective" = "multi:softprob",    # multiclass classification 
              "num_class" = length(levels(out)),    # number of classes 
              "eval_metric" = "merror",    # evaluation metric 
              "nthread" = 8,   # number of threads to be used 
              "max_depth" = 16,    # maximum depth of tree 
              "eta" = 0.3,    # step size shrinkage 
              "gamma" = 0,    # minimum loss reduction 
              "subsample" = 1,    # part of data instances to grow tree 
              "colsample_bytree" = 1,  # subsample ratio of columns when constructing each tree 
              "min_child_weight" = 12  # minimum sum of instance weight needed in a child 
)

set.seed(1234)
# k-fold cross validation, with timing
nround.cv <- 50
system.time( bst.cv <- xgb.cv(param=param, data=train.matrix, label=y, 
                              nfold=4, nrounds=nround.cv, prediction=TRUE, verbose=FALSE) )
tail(bst.cv$dt)

min.merror.idx <- which.min(bst.cv$dt[, test.merror.mean]) 
min.merror.idx 

bst.cv$dt[min.merror.idx,]

#predict 

pred.cv <- matrix(bst.cv$pred, nrow=length(bst.cv$pred)/length(levels(out)), ncol=length(levels(out)))
pred.cv <- max.col(pred.cv, "last")

#Confusion Matrix
confusionMatrix(factor(y+1), factor(pred.cv))

#this contains 1 and 2's where 1 -->FALSE and 2 --> TRUE
#Since I want to compare the data on equal ground lets convert the data 
train$V31 <- pred.cv
train$V31[train$V31 == 1] <- -1
train$V31[train$V31 == 2] <- 1

#library(compare)
#comparison <- compare(train$V31,train.org$V31,allowAll=TRUE)
#comparison$tM

mis <- train$V31 == train.org$V31
#Wrongly Predicted Value 
sum(mis == FALSE)
#Correctly Predicted Value 
sum(mis == TRUE )

system.time( bst <- xgboost(param=param, data=train.matrix, label=y, 
                            nrounds=min.merror.idx, verbose=0) )
pred <- predict(bst, test.matrix) 
head(pred,15)

pred <- matrix(pred, nrow=2, ncol=length(pred)/2)
pred <- t(pred)
pred <- max.col(pred, "last")
pred.ans <- toupper(letters[pred])
head(pred.ans,5)

pred.ans[pred.ans == "A"] <- -1
pred.ans[pred.ans == "B"] <-  1
head(pred.ans,5)

# get the trained model
model <- xgb.dump(bst, with.stats=TRUE)
# get the feature real names
names <- dimnames(train.matrix)[[2]]
# compute feature importance matrix
importance_matrix <- xgb.importance(names, model=bst)

# plot
gp <- xgb.plot.importance(importance_matrix)
print(gp)

############ Thank You ############################
names(train)