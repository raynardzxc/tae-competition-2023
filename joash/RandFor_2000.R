rm(list=ls())
library(randomForest)
library(caret)

safety <- read.csv("train1.csv")
pure <- safety
safety <- subset(safety, select=-c(No, Case, CC4,GN4,NS4,BU4,FA4,LD4,BZ4,FC4,FP4,RP4,PP4,KA4,SC4,TS4,NV4,MA4,LB4,AF4,HU4,Price4))

safety$Choice <- ifelse(safety$Ch1 == 1, 1,
                        ifelse(safety$Ch2 == 1, 2,
                               ifelse(safety$Ch3 == 1, 3,
                                      ifelse(safety$Ch4 == 1, 4, NA))))

set.seed(123)
trainingIndex <- createDataPartition(safety$Choice, p = 0.8, list = FALSE)

trainingSet <- safety[trainingIndex,]
testSet <- safety[-trainingIndex,]

set.seed(123)
model <- randomForest(as.factor(Choice) ~ ., data = trainingSet, mtry=12, importance=TRUE, ntree = 2000)

pred <- predict(model, testSet, type="prob")
fullpred <- predict(model, safety, type="prob")

colnames(pred) <- c("Ch1", "Ch2", "Ch3", "Ch4")
colnames(fullpred) <- c("Ch1", "Ch2", "Ch3", "Ch4")

logloss <- function(test_set, testpredict_df) {
  # Create one-hot encoding for each choice on-the-fly
  Ch1 <- as.integer(test_set$Choice == 1)
  Ch2 <- as.integer(test_set$Choice == 2)
  Ch3 <- as.integer(test_set$Choice == 3)
  Ch4 <- as.integer(test_set$Choice == 4)
  
  # Calculate logloss using these one-hot encoded variables
  result <- -1/nrow(test_set) * sum(Ch1 * log(testpredict_df$Ch1+.Machine$double.eps) +
                                      Ch2 * log(testpredict_df$Ch2+.Machine$double.eps) +
                                      Ch3 * log(testpredict_df$Ch3+.Machine$double.eps) +
                                      Ch4 * log(testpredict_df$Ch4+.Machine$double.eps))
  return(result)
}

loss <- logloss(testSet, as.data.frame(pred))
print(loss)
logloss(safety, as.data.frame(fullpred))


colnames(fullpred) <- c("Ch1","Ch2","Ch3","Ch4")
fullpred <- as.data.frame(fullpred)
fullpred$No <- pure$No
fullpred$Choice <- safety$Choice
fullpred <- fullpred[c("No","Ch1","Ch2","Ch3","Ch4", "Choice")]
write.csv(fullpred, file = "fulltestresults_rfor.csv", row.names = FALSE)


getNum <- read.csv("./test1.csv")

test <- subset(getNum, select = -c(No, Case, CC4,GN4,NS4,BU4,FA4,LD4,BZ4,FC4,FP4,RP4,PP4,KA4,SC4,TS4,NV4,MA4,LB4,AF4,HU4,Price4))

final_predict <- predict(model, test, type="prob")

colnames(final_predict) <- c("Ch1","Ch2","Ch3","Ch4")
final_predict_df <- as.data.frame(final_predict)
final_predict_df$No <- getNum$No

final_predict_df <- final_predict_df[c("No","Ch1","Ch2","Ch3","Ch4")]

write.csv(final_predict_df, file = "./randForest_2000.csv", row.names = FALSE)