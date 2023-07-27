
<!-- rnb-text-begin -->

---
title: "Random Forest"
output: html_notebook
---


<!-- rnb-text-end -->


<!-- rnb-chunk-begin -->


<!-- rnb-source-begin eyJkYXRhIjoiYGBgclxucm0obGlzdD1scygpKVxubGlicmFyeShyYW5kb21Gb3Jlc3QpXG5saWJyYXJ5KGNhcmV0KVxubGlicmFyeShtbHIpXG5saWJyYXJ5KGRhdGEudGFibGUpXG5gYGAifQ== -->

```r
rm(list=ls())
library(randomForest)
library(caret)
library(mlr)
library(data.table)
```

<!-- rnb-source-end -->

<!-- rnb-chunk-end -->


<!-- rnb-text-begin -->



<!-- rnb-text-end -->


<!-- rnb-chunk-begin -->


<!-- rnb-source-begin eyJkYXRhIjoiYGBgclxubGlicmFyeShwYXJhbGxlbE1hcClcbmxpYnJhcnkocGFyYWxsZWwpXG5wYXJhbGxlbFN0YXJ0U29ja2V0KGNwdXMgPSBkZXRlY3RDb3JlcygpKVxuYGBgIn0= -->

```r
library(parallelMap)
library(parallel)
parallelStartSocket(cpus = detectCores())
```

<!-- rnb-source-end -->

<!-- rnb-output-begin eyJkYXRhIjoiV2FybmluZzogUGFyYWxsZWxpemF0aW9uIHdhcyBub3Qgc3RvcHBlZCwgZG9pbmcgaXQgbm93LlN0b3BwZWQgcGFyYWxsZWxpemF0aW9uLiBBbGwgY2xlYW5lZCB1cC5cblN0YXJ0aW5nIHBhcmFsbGVsaXphdGlvbiBpbiBtb2RlPXNvY2tldCB3aXRoIGNwdXM9MTIuXG4ifQ== -->

```
Warning: Parallelization was not stopped, doing it now.Stopped parallelization. All cleaned up.
Starting parallelization in mode=socket with cpus=12.
```



<!-- rnb-output-end -->

<!-- rnb-chunk-end -->


<!-- rnb-text-begin -->



<!-- rnb-text-end -->


<!-- rnb-chunk-begin -->


<!-- rnb-source-begin eyJkYXRhIjoiYGBgclxuc2FmZXR5IDwtIHJlYWQuY3N2KFwidHJhaW4xX3RyeWluZy5jc3ZcIilcbnNhZmV0eSA8LSBzdWJzZXQoc2FmZXR5LCBzZWxlY3Q9LWMoTm8sIENhc2UsIENDNCxHTjQsTlM0LEJVNCxGQTQsTEQ0LEJaNCxGQzQsRlA0LFJQNCxQUDQsS0E0LFNDNCxUUzQsTlY0LE1BNCxMQjQsQUY0LEhVNCxQcmljZTQpKVxuXG4jIERlZmluZSB0aGUgYnJlYWtzIGZvciB0aGUgYmluc1xuI2JyZWFrcyA8LSBjKC0xLCAwLCAyOTk5OSwgMzk5OTksIDQ5OTk5LCA1OTk5OSwgNjk5OTksIDc5OTk5LCA4OTk5OSwgOTk5OTksIDEwOTk5OSwgMTE5OTk5LCAxMjk5OTksIDEzOTk5OSwgMTQ5OTk5LCAxNTk5OTksIDE2OTk5OSwgMTc5OTk5LCAxODk5OTksIDE5OTk5OSwgMjE5OTk5LCAyMzk5OTksIDI1OTk5OSwgMjc5OTk5LCAyOTk5OTksIEluZilcblxuIyBEZWZpbmUgdGhlIGxhYmVscyBmb3IgdGhlIGJpbnNcbiNsYWJlbHMgPC0gMToyNVxuXG4jIENvbnZlcnQgdGhlIGluY29tZWEgY29sdW1uIHRvIGJpbnMgYW5kIGxhYmVsIHRoZW1cbiNzYWZldHkkaW5jb21lYmlucyA8LSBjdXQoc2FmZXR5JGluY29tZWEsIGJyZWFrcyA9IGJyZWFrcywgbGFiZWxzID0gbGFiZWxzLCBpbmNsdWRlLmxvd2VzdCA9IFRSVUUpXG5cbiMgQ29udmVydCB0aGUgaW5jb21lX2JpbnMgY29sdW1uIHRvIGEgZmFjdG9yXG4jc2FmZXR5JGluY29tZWJpbnMgPC0gYXMuZmFjdG9yKHNhZmV0eSRpbmNvbWViaW5zKVxuXG4jIERlbGV0ZSBvdGhlciBpbmNvbWUgY29sdW1uc1xuI3NhZmV0eSA8LSBzdWJzZXQoc2FmZXR5LCBzZWxlY3QgPSAtYyhpbmNvbWVpbmQsaW5jb21lKSlcbmBgYCJ9 -->

```r
safety <- read.csv("train1_trying.csv")
safety <- subset(safety, select=-c(No, Case, CC4,GN4,NS4,BU4,FA4,LD4,BZ4,FC4,FP4,RP4,PP4,KA4,SC4,TS4,NV4,MA4,LB4,AF4,HU4,Price4))

# Define the breaks for the bins
#breaks <- c(-1, 0, 29999, 39999, 49999, 59999, 69999, 79999, 89999, 99999, 109999, 119999, 129999, 139999, 149999, 159999, 169999, 179999, 189999, 199999, 219999, 239999, 259999, 279999, 299999, Inf)

# Define the labels for the bins
#labels <- 1:25

# Convert the incomea column to bins and label them
#safety$incomebins <- cut(safety$incomea, breaks = breaks, labels = labels, include.lowest = TRUE)

# Convert the income_bins column to a factor
#safety$incomebins <- as.factor(safety$incomebins)

# Delete other income columns
#safety <- subset(safety, select = -c(incomeind,income))
```

<!-- rnb-source-end -->

<!-- rnb-chunk-end -->


<!-- rnb-text-begin -->



<!-- rnb-text-end -->


<!-- rnb-chunk-begin -->


<!-- rnb-source-begin eyJkYXRhIjoiYGBgclxuIyBGaW5kIHRoZSBwb3NpdGlvbiBvZiAnaW5jb21lYSdcbiNwb3MgPC0gd2hpY2gobmFtZXMoc2FmZXR5KSA9PSBcImluY29tZWFcIilcblxuIyBDcmVhdGUgYSBuZXcgY29sdW1uIG9yZGVyXG4jbmV3X29yZGVyIDwtIGMobmFtZXMoc2FmZXR5KVsxOihwb3MtMSldLCBcImluY29tZWJpbnNcIiwgbmFtZXMoc2FmZXR5KVtwb3M6KGxlbmd0aChzYWZldHkpLTEpXSlcblxuIyBSZWFycmFuZ2UgdGhlIGNvbHVtbnNcbiNzYWZldHkgPC0gc2FmZXR5WywgbmV3X29yZGVyXVxuXG4jaGVhZChzYWZldHkpXG5zZWVkIDwtIDEyM1xuYGBgIn0= -->

```r
# Find the position of 'incomea'
#pos <- which(names(safety) == "incomea")

# Create a new column order
#new_order <- c(names(safety)[1:(pos-1)], "incomebins", names(safety)[pos:(length(safety)-1)])

# Rearrange the columns
#safety <- safety[, new_order]

#head(safety)
seed <- 123
```

<!-- rnb-source-end -->

<!-- rnb-chunk-end -->


<!-- rnb-text-begin -->



<!-- rnb-text-end -->


<!-- rnb-chunk-begin -->


<!-- rnb-source-begin eyJkYXRhIjoiYGBgclxuc2V0LnNlZWQoc2VlZClcbnRyYWluaW5nSW5kZXggPC0gY3JlYXRlRGF0YVBhcnRpdGlvbihzYWZldHkkQ2hvaWNlLCBwID0gMC44LCBsaXN0ID0gRkFMU0UpXG5cbnRyYWluaW5nU2V0IDwtIHNhZmV0eVt0cmFpbmluZ0luZGV4LF1cbnRlc3RTZXQgPC0gc2FmZXR5Wy10cmFpbmluZ0luZGV4LF1cbmBgYCJ9 -->

```r
set.seed(seed)
trainingIndex <- createDataPartition(safety$Choice, p = 0.8, list = FALSE)

trainingSet <- safety[trainingIndex,]
testSet <- safety[-trainingIndex,]
```

<!-- rnb-source-end -->

<!-- rnb-chunk-end -->


<!-- rnb-text-begin -->



<!-- rnb-text-end -->


<!-- rnb-chunk-begin -->


<!-- rnb-source-begin eyJkYXRhIjoiYGBgclxuc2V0LnNlZWQoc2VlZClcbm10cnkgPC0gdHVuZVJGKHRyYWluaW5nU2V0WzE6bmNvbCh0cmFpbmluZ1NldCktMV0sIGFzLmZhY3Rvcih0cmFpbmluZ1NldCRDaG9pY2UpLFxuICAgICAgICAgICAgICAgc3RlcEZhY3Rvcj0xLjUsaW1wcm92ZT0wLjAxLCB0cmFjZT1UUlVFLCBwbG90PUZBTFNFKVxuYGBgIn0= -->

```r
set.seed(seed)
mtry <- tuneRF(trainingSet[1:ncol(trainingSet)-1], as.factor(trainingSet$Choice),
               stepFactor=1.5,improve=0.01, trace=TRUE, plot=FALSE)
```

<!-- rnb-source-end -->

<!-- rnb-output-begin eyJkYXRhIjoibXRyeSA9IDkgIE9PQiBlcnJvciA9IDUzLjgxJSBcblNlYXJjaGluZyBsZWZ0IC4uLlxubXRyeSA9IDYgXHRPT0IgZXJyb3IgPSA1NS40OCUgXG4tMC4wMzA5MTAwNyAwLjAxIFxuU2VhcmNoaW5nIHJpZ2h0IC4uLlxubXRyeSA9IDEzIFx0T09CIGVycm9yID0gNTIuNzYlIFxuMC4wMTk2MDE1MSAwLjAxIFxubXRyeSA9IDE5IFx0T09CIGVycm9yID0gNTMuMjklIFxuLTAuMDEwMTA2NTYgMC4wMSBcbiJ9 -->

```
mtry = 9  OOB error = 53.81% 
Searching left ...
mtry = 6 	OOB error = 55.48% 
-0.03091007 0.01 
Searching right ...
mtry = 13 	OOB error = 52.76% 
0.01960151 0.01 
mtry = 19 	OOB error = 53.29% 
-0.01010656 0.01 
```



<!-- rnb-output-end -->

<!-- rnb-source-begin eyJkYXRhIjoiYGBgclxuYmVzdC5tIDwtIG10cnlbbXRyeVssIDJdID09IG1pbihtdHJ5WywgMl0pLCAxXVxuYGBgIn0= -->

```r
best.m <- mtry[mtry[, 2] == min(mtry[, 2]), 1]
```

<!-- rnb-source-end -->

<!-- rnb-chunk-end -->


<!-- rnb-text-begin -->



<!-- rnb-text-end -->


<!-- rnb-chunk-begin -->


<!-- rnb-source-begin eyJkYXRhIjoiYGBgclxuc2V0LnNlZWQoc2VlZClcbm1vZGVsIDwtIHJhbmRvbUZvcmVzdChhcy5mYWN0b3IoQ2hvaWNlKSB+IC4sIGRhdGEgPSB0cmFpbmluZ1NldCwgbXRyeT1iZXN0Lm0sIGltcG9ydGFuY2U9VFJVRSwgbnRyZWUgPSAyMDAxKVxuI2ltcG9ydGFuY2UobW9kZWwpXG5tb2RlbFxuYGBgIn0= -->

```r
set.seed(seed)
model <- randomForest(as.factor(Choice) ~ ., data = trainingSet, mtry=best.m, importance=TRUE, ntree = 2001)
#importance(model)
model
```

<!-- rnb-source-end -->

<!-- rnb-output-begin eyJkYXRhIjoiXG5DYWxsOlxuIHJhbmRvbUZvcmVzdChmb3JtdWxhID0gYXMuZmFjdG9yKENob2ljZSkgfiAuLCBkYXRhID0gdHJhaW5pbmdTZXQsICAgICAgbXRyeSA9IGJlc3QubSwgaW1wb3J0YW5jZSA9IFRSVUUsIG50cmVlID0gMjAwMSkgXG4gICAgICAgICAgICAgICBUeXBlIG9mIHJhbmRvbSBmb3Jlc3Q6IGNsYXNzaWZpY2F0aW9uXG4gICAgICAgICAgICAgICAgICAgICBOdW1iZXIgb2YgdHJlZXM6IDIwMDFcbk5vLiBvZiB2YXJpYWJsZXMgdHJpZWQgYXQgZWFjaCBzcGxpdDogMTNcblxuICAgICAgICBPT0IgZXN0aW1hdGUgb2YgIGVycm9yIHJhdGU6IDQ3LjQyJVxuQ29uZnVzaW9uIG1hdHJpeDpcbiAgICAgMSAgICAyICAgIDMgICAgNCBjbGFzcy5lcnJvclxuMSAxNjY0ICA2NjcgIDQ5NyAgOTI4ICAgMC41NTY5NzU1XG4yICA1MjMgMjI0MSAgNjc1ICA5MjQgICAwLjQ4NjM2MjZcbjMgIDQ2MiAgNjgxIDE5MDAgIDg3NiAgIDAuNTE1MTgyNFxuNCAgNTc0ICA3MjkgIDY0NSAzMjY4ICAgMC4zNzM0NjYzXG4ifQ== -->

```

Call:
 randomForest(formula = as.factor(Choice) ~ ., data = trainingSet,      mtry = best.m, importance = TRUE, ntree = 2001) 
               Type of random forest: classification
                     Number of trees: 2001
No. of variables tried at each split: 13

        OOB estimate of  error rate: 47.42%
Confusion matrix:
     1    2    3    4 class.error
1 1664  667  497  928   0.5569755
2  523 2241  675  924   0.4863626
3  462  681 1900  876   0.5151824
4  574  729  645 3268   0.3734663
```



<!-- rnb-output-end -->

<!-- rnb-chunk-end -->


<!-- rnb-text-begin -->



<!-- rnb-text-end -->


<!-- rnb-chunk-begin -->


<!-- rnb-source-begin eyJkYXRhIjoiYGBgclxucHJlZCA8LSBwcmVkaWN0KG1vZGVsLCB0ZXN0U2V0LCB0eXBlPVwicHJvYlwiKVxuY29sbmFtZXMocHJlZCkgPC0gYyhcIkNoMVwiLCBcIkNoMlwiLCBcIkNoM1wiLCBcIkNoNFwiKVxuYGBgIn0= -->

```r
pred <- predict(model, testSet, type="prob")
colnames(pred) <- c("Ch1", "Ch2", "Ch3", "Ch4")
```

<!-- rnb-source-end -->

<!-- rnb-chunk-end -->


<!-- rnb-text-begin -->



<!-- rnb-text-end -->


<!-- rnb-chunk-begin -->


<!-- rnb-source-begin eyJkYXRhIjoiYGBgclxubG9nbG9zcyA8LSBmdW5jdGlvbih0ZXN0X3NldCwgdGVzdHByZWRpY3RfZGYpIHtcbiAgIyBDcmVhdGUgb25lLWhvdCBlbmNvZGluZyBmb3IgZWFjaCBjaG9pY2Ugb24tdGhlLWZseVxuICBDaDEgPC0gYXMuaW50ZWdlcih0ZXN0X3NldCRDaG9pY2UgPT0gMSlcbiAgQ2gyIDwtIGFzLmludGVnZXIodGVzdF9zZXQkQ2hvaWNlID09IDIpXG4gIENoMyA8LSBhcy5pbnRlZ2VyKHRlc3Rfc2V0JENob2ljZSA9PSAzKVxuICBDaDQgPC0gYXMuaW50ZWdlcih0ZXN0X3NldCRDaG9pY2UgPT0gNClcbiAgXG4gICMgQ2FsY3VsYXRlIGxvZ2xvc3MgdXNpbmcgdGhlc2Ugb25lLWhvdCBlbmNvZGVkIHZhcmlhYmxlc1xuICByZXN1bHQgPC0gLTEvbnJvdyh0ZXN0X3NldCkgKiBzdW0oQ2gxICogbG9nKHRlc3RwcmVkaWN0X2RmJENoMSsuTWFjaGluZSRkb3VibGUuZXBzKSArXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBDaDIgKiBsb2codGVzdHByZWRpY3RfZGYkQ2gyKy5NYWNoaW5lJGRvdWJsZS5lcHMpICtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIENoMyAqIGxvZyh0ZXN0cHJlZGljdF9kZiRDaDMrLk1hY2hpbmUkZG91YmxlLmVwcykgK1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgQ2g0ICogbG9nKHRlc3RwcmVkaWN0X2RmJENoNCsuTWFjaGluZSRkb3VibGUuZXBzKSlcbiAgcmV0dXJuKHJlc3VsdClcbn1cbmBgYCJ9 -->

```r
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
```

<!-- rnb-source-end -->

<!-- rnb-chunk-end -->


<!-- rnb-text-begin -->



<!-- rnb-text-end -->


<!-- rnb-chunk-begin -->


<!-- rnb-source-begin eyJkYXRhIjoiYGBgclxubG9zcyA8LSBsb2dsb3NzKHRlc3RTZXQsIGFzLmRhdGEuZnJhbWUocHJlZCkpXG5sb3NzXG5gYGAifQ== -->

```r
loss <- logloss(testSet, as.data.frame(pred))
loss
```

<!-- rnb-source-end -->

<!-- rnb-output-begin eyJkYXRhIjoiWzFdIDEuMTQ1MDU0XG4ifQ== -->

```
[1] 1.145054
```



<!-- rnb-output-end -->

<!-- rnb-chunk-end -->


<!-- rnb-chunk-begin -->


<!-- rnb-source-begin eyJkYXRhIjoiYGBgclxucmF5bmFyZF9kZiA8LSBhcy5kYXRhLmZyYW1lKHByZWQpXG5yYXluYXJkX2RmJE5vIDwtIHRlc3RTZXQkTm9cbnJheW5hcmRfZGYgPC0gcmF5bmFyZF9kZltjKFwiTm9cIixcIkNoMVwiLFwiQ2gyXCIsXCJDaDNcIixcIkNoNFwiKV1cbndyaXRlLmNzdihyYXluYXJkX2RmLCBmaWxlID0gXCIuL3Rlc3RyZXN1bHRzXzIwMDFmb3JfYWxsdmFyaWFibGVzLmNzdlwiLCByb3cubmFtZXMgPSBGQUxTRSlcbmBgYCJ9 -->

```r
raynard_df <- as.data.frame(pred)
raynard_df$No <- testSet$No
raynard_df <- raynard_df[c("No","Ch1","Ch2","Ch3","Ch4")]
write.csv(raynard_df, file = "./testresults_2001for_allvariables.csv", row.names = FALSE)
```

<!-- rnb-source-end -->

<!-- rnb-chunk-end -->


<!-- rnb-text-begin -->




<!-- rnb-text-end -->


<!-- rnb-chunk-begin -->


<!-- rnb-source-begin eyJkYXRhIjoiYGBgclxuI3NldCBhbGwgY2hhcmFjdGVyIHZhcmlhYmxlcyBhcyBmYWN0b3JcbnRyYWluaW5nU2V0IDwtIHN1YnNldCh0cmFpbmluZ1NldCwgc2VsZWN0PWMoQ2hvaWNlLCBQcmljZTMsIFByaWNlMiwgUHJpY2UxLCBhZ2VhLCBpbmNvbWVhLCBtaWxlc2EsIG5pZ2h0YSwgeWVhciwgeWVhcmluZCwgaW5jb21lYmlucywgbWlsZXMsIG1pbGVzaW5kICxuaWdodGluZCwgbmlnaHQsIHBwYXJraW5kLCBwcGFyaywgc2VnbWVudCwgc2VnbWVudGluZCwgcmVnaW9uLCByZWdpb25pbmQsIEJVMykpXG5cbnRlc3RTZXQgPC0gc3Vic2V0KHRlc3RTZXQsIHNlbGVjdD1jKENob2ljZSwgUHJpY2UzLCBQcmljZTIsIFByaWNlMSwgYWdlYSwgaW5jb21lYSwgbWlsZXNhLCBuaWdodGEsIHllYXIsIHllYXJpbmQsIGluY29tZWJpbnMsIG1pbGVzLCBtaWxlc2luZCAsbmlnaHRpbmQsIG5pZ2h0LCBwcGFya2luZCwgcHBhcmssIHNlZ21lbnQsIHNlZ21lbnRpbmQsIHJlZ2lvbiwgcmVnaW9uaW5kLCBCVTMpKVxuXG5mYWN0X2NvbCA8LSBjb2xuYW1lcyh0cmFpbmluZ1NldClbc2FwcGx5KHRyYWluaW5nU2V0LGlzLmNoYXJhY3RlcildXG5cbmZvcihpIGluIGZhY3RfY29sKVxuICAgICAgICBzZXQodHJhaW5pbmdTZXQsaj1pLHZhbHVlID0gZmFjdG9yKHRyYWluaW5nU2V0W1tpXV0pKVxuXG5mb3IoaSBpbiBmYWN0X2NvbClcbiAgICAgICAgc2V0KHRlc3RTZXQsaj1pLHZhbHVlID0gZmFjdG9yKHRlc3RTZXRbW2ldXSkpXG5gYGAifQ== -->

```r
#set all character variables as factor
trainingSet <- subset(trainingSet, select=c(Choice, Price3, Price2, Price1, agea, incomea, milesa, nighta, year, yearind, incomebins, miles, milesind ,nightind, night, pparkind, ppark, segment, segmentind, region, regionind, BU3))

testSet <- subset(testSet, select=c(Choice, Price3, Price2, Price1, agea, incomea, milesa, nighta, year, yearind, incomebins, miles, milesind ,nightind, night, pparkind, ppark, segment, segmentind, region, regionind, BU3))

fact_col <- colnames(trainingSet)[sapply(trainingSet,is.character)]

for(i in fact_col)
        set(trainingSet,j=i,value = factor(trainingSet[[i]]))

for(i in fact_col)
        set(testSet,j=i,value = factor(testSet[[i]]))
```

<!-- rnb-source-end -->

<!-- rnb-chunk-end -->


<!-- rnb-text-begin -->



<!-- rnb-text-end -->


<!-- rnb-chunk-begin -->


<!-- rnb-source-begin eyJkYXRhIjoiYGBgclxubGV2ZWxzKHRyYWluaW5nU2V0JGluY29tZWJpbnMpXG5gYGAifQ== -->

```r
levels(trainingSet$incomebins)
```

<!-- rnb-source-end -->

<!-- rnb-output-begin eyJkYXRhIjoiIFsxXSBcIjFcIiAgXCIyXCIgIFwiM1wiICBcIjRcIiAgXCI1XCIgIFwiNlwiICBcIjdcIiAgXCI4XCIgIFwiOVwiICBcIjEwXCIgXCIxMVwiIFwiMTJcIiBcIjEzXCIgXCIxNFwiIFwiMTVcIiBcIjE2XCIgXCIxN1wiIFwiMThcIiBcIjE5XCIgXCIyMFwiIFwiMjFcIiBcIjIyXCIgXCIyM1wiIFwiMjRcIiBcIjI1XCJcbiJ9 -->

```
 [1] "1"  "2"  "3"  "4"  "5"  "6"  "7"  "8"  "9"  "10" "11" "12" "13" "14" "15" "16" "17" "18" "19" "20" "21" "22" "23" "24" "25"
```



<!-- rnb-output-end -->

<!-- rnb-source-begin eyJkYXRhIjoiYGBgclxubGV2ZWxzKHRlc3RTZXQkaW5jb21lYmlucylcbmBgYCJ9 -->

```r
levels(testSet$incomebins)
```

<!-- rnb-source-end -->

<!-- rnb-output-begin eyJkYXRhIjoiIFsxXSBcIjFcIiAgXCIyXCIgIFwiM1wiICBcIjRcIiAgXCI1XCIgIFwiNlwiICBcIjdcIiAgXCI4XCIgIFwiOVwiICBcIjEwXCIgXCIxMVwiIFwiMTJcIiBcIjEzXCIgXCIxNFwiIFwiMTVcIiBcIjE2XCIgXCIxN1wiIFwiMThcIiBcIjE5XCIgXCIyMFwiIFwiMjFcIiBcIjIyXCIgXCIyM1wiIFwiMjRcIiBcIjI1XCJcbiJ9 -->

```
 [1] "1"  "2"  "3"  "4"  "5"  "6"  "7"  "8"  "9"  "10" "11" "12" "13" "14" "15" "16" "17" "18" "19" "20" "21" "22" "23" "24" "25"
```



<!-- rnb-output-end -->

<!-- rnb-chunk-end -->


<!-- rnb-text-begin -->




<!-- rnb-text-end -->


<!-- rnb-chunk-begin -->


<!-- rnb-source-begin eyJkYXRhIjoiYGBgclxudHJhaW50YXNrIDwtIG1ha2VDbGFzc2lmVGFzayhkYXRhID0gdHJhaW5pbmdTZXQsdGFyZ2V0ID0gXCJDaG9pY2VcIilcbmBgYCJ9 -->

```r
traintask <- makeClassifTask(data = trainingSet,target = "Choice")
```

<!-- rnb-source-end -->

<!-- rnb-output-begin eyJkYXRhIjoiV2FybmluZzogRW1wdHkgZmFjdG9yIGxldmVscyB3ZXJlIGRyb3BwZWQgZm9yIGNvbHVtbnM6IGluY29tZWJpbnNcbiJ9 -->

```
Warning: Empty factor levels were dropped for columns: incomebins
```



<!-- rnb-output-end -->

<!-- rnb-source-begin eyJkYXRhIjoiYGBgclxudGVzdHRhc2sgPC0gbWFrZUNsYXNzaWZUYXNrKGRhdGEgPSB0ZXN0U2V0LHRhcmdldCA9IFwiQ2hvaWNlXCIpXG5gYGAifQ== -->

```r
testtask <- makeClassifTask(data = testSet,target = "Choice")
```

<!-- rnb-source-end -->

<!-- rnb-output-begin eyJkYXRhIjoiV2FybmluZzogRW1wdHkgZmFjdG9yIGxldmVscyB3ZXJlIGRyb3BwZWQgZm9yIGNvbHVtbnM6IGluY29tZWJpbnNcbiJ9 -->

```
Warning: Empty factor levels were dropped for columns: incomebins
```



<!-- rnb-output-end -->

<!-- rnb-chunk-end -->


<!-- rnb-text-begin -->



<!-- rnb-text-end -->


<!-- rnb-chunk-begin -->


<!-- rnb-source-begin eyJkYXRhIjoiYGBgclxucmRlc2MgPC0gbWFrZVJlc2FtcGxlRGVzYyhcIkNWXCIsaXRlcnM9NUwpXG5gYGAifQ== -->

```r
rdesc <- makeResampleDesc("CV",iters=5L)
```

<!-- rnb-source-end -->

<!-- rnb-chunk-end -->


<!-- rnb-text-begin -->



<!-- rnb-text-end -->


<!-- rnb-chunk-begin -->


<!-- rnb-source-begin eyJkYXRhIjoiYGBgclxuI1JhbmRvbSBGb3Jlc3Qgd2l0aG91dCBDdXRvZmZcbnJmLmxybiA8LSBtYWtlTGVhcm5lcihcImNsYXNzaWYucmFuZG9tRm9yZXN0XCIsIHByZWRpY3QudHlwZSA9IFwicHJvYlwiKVxucmYubHJuJHBhci52YWxzIDwtIGxpc3QobnRyZWUgPSAyMDAxTCxcbiAgICAgICAgICAgICAgICAgICAgICAgIGltcG9ydGFuY2U9VFJVRSlcblxuciA8LSByZXNhbXBsZShsZWFybmVyID0gcmYubHJuXG4gICAgICAgICAgICAgICx0YXNrID0gdHJhaW50YXNrXG4gICAgICAgICAgICAgICxyZXNhbXBsaW5nID0gcmRlc2NcbiAgICAgICAgICAgICAgLG1lYXN1cmVzID0gbGlzdChhY2MpXG4gICAgICAgICAgICAgICxzaG93LmluZm8gPSBULFxuICAgICAgICAgICAgICBtdHJ5ID0gYmVzdC5tKVxuYGBgIn0= -->

```r
#Random Forest without Cutoff
rf.lrn <- makeLearner("classif.randomForest", predict.type = "prob")
rf.lrn$par.vals <- list(ntree = 2001L,
                        importance=TRUE)

r <- resample(learner = rf.lrn
              ,task = traintask
              ,resampling = rdesc
              ,measures = list(acc)
              ,show.info = T,
              mtry = best.m)
```

<!-- rnb-source-end -->

<!-- rnb-output-begin eyJkYXRhIjoiRXhwb3J0aW5nIG9iamVjdHMgdG8gc2xhdmVzIGZvciBtb2RlIHNvY2tldDogLm1sci5zbGF2ZS5vcHRpb25zXG5SZXNhbXBsaW5nOiBjcm9zcy12YWxpZGF0aW9uXG5NZWFzdXJlczogICAgICAgICAgICAgYWNjICAgICAgIFxuTWFwcGluZyBpbiBwYXJhbGxlbDogbW9kZSA9IHNvY2tldDsgbGV2ZWwgPSBtbHIucmVzYW1wbGU7IGNwdXMgPSAxMjsgZWxlbWVudHMgPSA1LlxuIn0= -->

```
Exporting objects to slaves for mode socket: .mlr.slave.options
Resampling: cross-validation
Measures:             acc       
Mapping in parallel: mode = socket; level = mlr.resample; cpus = 12; elements = 5.
```



<!-- rnb-output-end -->

<!-- rnb-chunk-end -->


<!-- rnb-text-begin -->



<!-- rnb-text-end -->


<!-- rnb-chunk-begin -->


<!-- rnb-source-begin eyJkYXRhIjoiYGBgclxuQ1Ztb2RlbCA8LSB0cmFpbihyZi5scm4sIHRyYWludGFzaylcbmBgYCJ9 -->

```r
CVmodel <- train(rf.lrn, traintask)
```

<!-- rnb-source-end -->

<!-- rnb-chunk-end -->


<!-- rnb-text-begin -->



<!-- rnb-text-end -->


<!-- rnb-chunk-begin -->


<!-- rnb-source-begin eyJkYXRhIjoiYGBgclxuQ1Ztb2RlbFxuYGBgIn0= -->

```r
CVmodel
```

<!-- rnb-source-end -->

<!-- rnb-output-begin eyJkYXRhIjoiTW9kZWwgZm9yIGxlYXJuZXIuaWQ9Y2xhc3NpZi5yYW5kb21Gb3Jlc3Q7IGxlYXJuZXIuY2xhc3M9Y2xhc3NpZi5yYW5kb21Gb3Jlc3RcblRyYWluZWQgb246IHRhc2suaWQgPSB0cmFpbmluZ1NldDsgb2JzID0gMTcyNTQ7IGZlYXR1cmVzID0gMjFcbkh5cGVycGFyYW1ldGVyczogbnRyZWU9MTAwLGltcG9ydGFuY2U9VFJVRVxuIn0= -->

```
Model for learner.id=classif.randomForest; learner.class=classif.randomForest
Trained on: task.id = trainingSet; obs = 17254; features = 21
Hyperparameters: ntree=100,importance=TRUE
```



<!-- rnb-output-end -->

<!-- rnb-chunk-end -->


<!-- rnb-text-begin -->



<!-- rnb-text-end -->


<!-- rnb-chunk-begin -->


<!-- rnb-source-begin eyJkYXRhIjoiYGBgclxuQ1ZSRiA8LSBwcmVkaWN0KENWbW9kZWwsIHRlc3R0YXNrLCB0eXBlPVwicHJvYlwiKVxuYGBgIn0= -->

```r
CVRF <- predict(CVmodel, testtask, type="prob")
```

<!-- rnb-source-end -->

<!-- rnb-chunk-end -->


<!-- rnb-text-begin -->



<!-- rnb-text-end -->


<!-- rnb-chunk-begin -->


<!-- rnb-source-begin eyJkYXRhIjoiYGBgclxuQ1ZSRlxuYGBgIn0= -->

```r
CVRF
```

<!-- rnb-source-end -->

<!-- rnb-output-begin eyJkYXRhIjoiUHJlZGljdGlvbjogNDMxMSBvYnNlcnZhdGlvbnNcbnByZWRpY3QudHlwZTogcHJvYlxudGhyZXNob2xkOiAxPTAuMjUsMj0wLjI1LDM9MC4yNSw0PTAuMjVcbnRpbWU6IDAuMjFcbiJ9 -->

```
Prediction: 4311 observations
predict.type: prob
threshold: 1=0.25,2=0.25,3=0.25,4=0.25
time: 0.21
```



<!-- rnb-output-end -->

<!-- rnb-frame-begin eyJtZXRhZGF0YSI6eyJjbGFzc2VzIjpbImRhdGEuZnJhbWUiXSwibnJvdyI6NiwibmNvbCI6Nywic3VtbWFyeSI6eyJEZXNjcmlwdGlvbiI6WyJkZiBbNiDDlyA3XSJdfX0sInJkZiI6Ikg0c0lBQUFBQUFBQUJwVlRQVWpEVUJCKytXdHR3RkxRV1ZlWEJrMnE2RkRlSXVyYW91RGdrclpSaXpHSlNhcWlxODZkZGRYRnhVblFVYXVDQ29JTzRpQU9kYTJycS9XOTVGMU1pd29HSHZudTdydnY3dDN4aXBQem1qd3ZJNFFFSkFvY0VpUUNrVFEzTzVVZFIwamtpY0VoRWFYb2Y1T1ErZ2hJa3ROTFRpSUlJc1FIMllSTk0wTy9BSEdlK2NVWVQrZ1NUcGpHdW1GNkJHVUNKaXMzQWtBRm9BSElkU2xJWlZQM1FDQ1NYZFRMdnUwUzlFbE9tbnJ3MDhCWm9iVTdpSysybXRQSC9kdjRWTTQ3Qjg4eVBtSDJ5MjVBd0xkNzlOdVA4aDVKZEtzNWd5KzE0TU03akgvZHFDOE1mVnppdTJaQUFMMG83NkpGQmQveEsvWG0xL0JScUl1YnpBOTl2RkdaUmozS2U2QjA1eERmczd3V3EzdFRvSWxGNkErZmg3ell3TG5ZTW9Sdm00NkxiLzkveEVMN3I2bDJMc0hTVncxWVFwSTUrV29Gd3I1Yjg1ZEJ4WEh0a2pMU1lha2RsdFpoNVpqVjR4cWVZMXVlMFZVNzVkb2JTcngrQXVxcnd4RWFBNlJwRVpvQWxGTWpOUHJMMWVXSzd1dktva3ZLL0hEOXBPMzRWZElhVWVoakR5R2V6TGxkamt6Tm9nMVhzdVhsbXJXU3BWV1J6QmJHc1JmR3NhVUNUb1kxUmVoTWdpRVoxbExWTW1EU3BsNHlUR2FreVdTQ3dTaU9XN1Y4dUFyeGVvcHYrenJ3NUxKdGdpZDhNWjlmRGpmWlh4d0VBQUE9In0= -->

<div data-pagedtable="false">
  <script data-pagedtable-source type="application/json">
{"columns":[{"label":[""],"name":["_rn_"],"type":[""],"align":["left"]},{"label":["id"],"name":[1],"type":["int"],"align":["right"]},{"label":["truth"],"name":[2],"type":["fctr"],"align":["left"]},{"label":["prob.1"],"name":[3],"type":["dbl"],"align":["right"]},{"label":["prob.2"],"name":[4],"type":["dbl"],"align":["right"]},{"label":["prob.3"],"name":[5],"type":["dbl"],"align":["right"]},{"label":["prob.4"],"name":[6],"type":["dbl"],"align":["right"]},{"label":["response"],"name":[7],"type":["fctr"],"align":["left"]}],"data":[{"1":"1","2":"2","3":"0.33","4":"0.27","5":"0.14","6":"0.26","7":"1","_rn_":"20"},{"1":"2","2":"4","3":"0.16","4":"0.15","5":"0.46","6":"0.23","7":"3","_rn_":"26"},{"1":"3","2":"4","3":"0.09","4":"0.01","5":"0.05","6":"0.85","7":"4","_rn_":"33"},{"1":"4","2":"2","3":"0.08","4":"0.17","5":"0.56","6":"0.19","7":"3","_rn_":"39"},{"1":"5","2":"3","3":"0.43","4":"0.21","5":"0.16","6":"0.20","7":"1","_rn_":"42"},{"1":"6","2":"3","3":"0.20","4":"0.09","5":"0.58","6":"0.13","7":"3","_rn_":"45"}],"options":{"columns":{"min":{},"max":[10],"total":[7]},"rows":{"min":[10],"max":[10],"total":[6]},"pages":{}}}
  </script>
</div>

<!-- rnb-frame-end -->

<!-- rnb-output-begin eyJkYXRhIjoiLi4uICgjcm93czogNDMxMSwgI2NvbHM6IDcpXG4ifQ== -->

```
... (#rows: 4311, #cols: 7)
```



<!-- rnb-output-end -->

<!-- rnb-chunk-end -->


<!-- rnb-text-begin -->



<!-- rnb-text-end -->


<!-- rnb-chunk-begin -->


<!-- rnb-source-begin eyJkYXRhIjoiYGBgclxuQ1ZSRiA8LSBDVlJGJGRhdGFbLDM6Nl1cbkNWUkZcbmBgYCJ9 -->

```r
CVRF <- CVRF$data[,3:6]
CVRF
```

<!-- rnb-source-end -->

<!-- rnb-frame-begin eyJtZXRhZGF0YSI6eyJjbGFzc2VzIjpbImRhdGEuZnJhbWUiXSwibnJvdyI6NDMxMSwibmNvbCI6NCwic3VtbWFyeSI6eyJEZXNjcmlwdGlvbiI6WyJkZiBbNCwzMTEgw5cgNF0iXX19LCJyZGYiOiJINHNJQUFBQUFBQUFCbldkVDR3ZHgzSEdWeExsd0FSaUNMRFB6bFU1U05nMzNUTTljeEQyRXNTNTJrZ0FIM3loWlRrUklwTUtKY09CYzEyZmZkbExjdUFsdlBDeUFJRUZhQUxFZ3BSV29DbXNGNkswQ2tXSFlVU0tmeVhLTi92b1pCN1p2NjZQSC9FSUNGTXowMTFkZjc2cXJxNTVwSDd3Tno5TVIzOTRkRzF0N1lXMUl5ODh0L2JDaXpPNTl1SS8vUDNmdmpLdXJSMTVmcjU1YnUzSTJqZVgxMytkQjMxN0pvN00vMzFycGg5c2ZQcmQzM3ovcTEvOTFjYmVMMjk5Yi9zNy83Wng3dWhyYi8vbnRhTWJPL1grdjMvMWVNREdsWDlmL3ZtUGpVL2YrL1dQWHY3ait4di85ZFBIZnpadUx0OSs5emRyOWMvRzJlWGQ5eDl0L00rU3kydi9zdkhSOHZMMjZZM3RKK00yTHJ6OHg1bkRYN2R4ckxkYjM5K3E2OXhlTHYvTFd4dnZwOGQvbW55TTR6blhqK3I2ZDZzK2pQdnQ0OHNQbWo2ZjEvdFRKamZ2VDlmcko1VXZldCtxOHFJM3p6K3IvQzVYUFQ1ZW11ZTlYMitjcis5L1gvVzQrdmp5ZDgzZWJtZld2Vmp0eFhQV08xUDU3WmlmbVBlN2FtL2svbnhwNXBmL3RIR2oycHY1ckgvZStPR25hMHN4djdmZCtHNmJ2ZUdIUDY1Vk9XNCtudmFkSmhmejBPZkRhZ2Z1d1FGWCtGZS9iR3hWL3RpVjU4ai9rZG5wVXZVUDczbk9QT3p6UWNVdnp6ZHRIUG96SGh5QjU4L3J1cDg4VVh6ajltUHhYdHM0ckh6UWgzWEFIMzRFSC9nSmU3QStjUUZPdHd3ZjZJVTh5TUY2cDAwTytGK3YvdWI1L1dwWDV1RlBjSDNleGhOZjZBbXV2Nmg2M05sZi92bGRzeGQ2TWg4N2dBdkh4MEc5M3pOOWtjUHpCRmZIRzgvQnpUbUx5eTlxUHRzMU9jaHpqRU5lY1BWWmpTZjBmbER2NGJOZjlTVE83MVYvWTdldnFwL3dMK3ZlcnVzaDUxZFZqanVWSC9iMi9JeDg0QmU1dHkxK2ZONEZpMS95QW5GODN2ei92NC9aL3FqeHhTODN6VzdFSGZ3YWpxdit5UE5GdGNPbStabjhpbDdnQXZ0Z2gxdlZ2c3dqRDdCdXMydU5TKzd2MWZ4MHcrTEM4eHQ2bnpiNUdFL2M3cGo5d0J2ckVzL0loMTdFRjdnaEw4THZnZTFYZTNZbEw5MDBmYkFQZG9Zdjh1SVA5RU5mY0lHZnVaSlh3RE4yK24zVjcycmx4M2pXMnpNL2tmZFlIN3V3enhOWCsxWVhZQy9mUng5WVhPSlg5TUFmNUVmMlovSWM4bUkvNUQyd3ZJRmR3VGY2RTJma3V5dUdGODhieEQ5NmdLTlc3OVJ4ck1ON3h4L3I0SC9pK0lyWjhiZW1yOWNqaHlZbitkRGZzdy82Zm8rZFBBOVhQTFI2aW5INEd4ejV2bnZKY0lHK3ZtNkw0NG9uOGdoNXgvY3ArSUpMNGdKY1U4Y3kzL0VPYnNpTE96YmYvUTAvOWxueUhyZ0VMOXh2bWw3ZzliTGxhZlRBajlnRGYxTW5XMTM3VFAxR3ZLM0tLMWZNLy9CdGVhVHFEWjhtaDlXL3JaNnMrb1B2aC9YS1BPSWNuTzZZUFlrSDdwRURmWHpmNXdwZmp4L3lNbjd5T2dVNThBTjVDejhoSDNHeVpYcVFWOGlyeEEvNWwzejV3UGFERzFZMzNLZzR3YTdrUmZSdmRlNFRlVnU4b1NmNmtLYzliN0Fmb1IvUDBRUGMzVE83WWpmR01jL3JDbkNESGJFWGNvQlRyOXNaaDEvd0IzWUFsMjRmOUVRdThpZDVEVC9CeitzYjE0L240Tm5yekJ1Vzk3bnkzdmM1OGhSeXVMekVDZnNiL3Z2WTZrS3ZHN2RYNUZYc2pqNStEZ0FuaDJZWHpqWGcxODlsZm00REgzZHIvbVQ5YW8rR1M2OC8wY1B0Q3Q5VmRhcmowKzNCT09SRlArVHljeUIrWS96bTAvWm84cDh5ZTd2OTdsRFgydmtWZkpPbjBjdHhRejVndjN4UTE3MXFPTUplOTJ1ZWFmbTE4aUd1cWJ0WWozdm00MmV2WCs1V0hLQW4rWWw2eXVzWFB6L3dITG14RSt0U3B4SC81TVc3MVIvb1NkenhISCtCYy9ZaDRtUFg3TTE4NU9IS1BrL2VRTTk5cS9QZ3cva0dmZkEzZWNYM0Y1NXpYc0V1Mk52ckIvWWpqeXV2eC9FUDc2M3YxdlI2Vk44akQrTjI3QjU3a29ld0gvc1Y5a04rNUNST3FHL0laOGgxdzlhblR1TTVlWVo2Q2o3SWRXRHhpWDljWC94RlhJRi83ME1nQjNhOFlYbUQ5WnZmcXA3TXUyNTJ4ODdrTytManNzVVpmTEFmdUVZTzdubFAzcmxqOG5uZGdwNnN5L01yWnNlUExPOTYvOVg3WCtCajMvQ0ZmUGlkZmRyUGpmY3JidkJmcThQdG5FUDhZMWYySjYvL3lEUEVpWjhyUFE4aEorczZqbmNzWHpNT25JQkg1RUZ1OGhTNGNqeVNGNUNUdkl5K3lMdGxmdURlNjFqOGdYN29SVjNBOHozTE84aUozanduUHNoMzN1L0FEdkRGVHQ2UEJGL2c4bUhGT1hLUXA2bjcvRHkzYjNrTnUzc2ZqcnFseFp2aEFwenNXenlDTzNCeXhmUUJyOVJoNE1EamcvMmw5VGZxdmRkaGZuN0Mzc2pqL1hiMkZmaGRYMkZ2cjA4WlJ6NWtYeUdPbUllK1hsOWhSNDk3K0hNK1JWL3dnMTdJdTRvdjlyNWcrckplTzI5YnY0ejR3VzZlQjcyK2M1eDZuMmpmOWgvMDlPOEp0ODJ2NEE3L3VMem9RYjN2OHREL0JYLzNURy9HazErSUM4OGIzUE85aHVmNG5mVzlqdDIxK0R3d1A2TUg5U2wydW16NTVwekp4enpQZSt3NzNMUFArTGtMZnZqcmtyMW52UGVwL0p6bWZhQlBWL2lWSzN4M0xPL2k3NjBWZVBNOGMyaHg2ZjRrUHRqUHNjT3UyY25QTDQ1M253ZC83TVo0OEFGK3NRL25CdnpGbGZnaVgrQkg3eWV4THV1QVUrNnhxOGNyOStnSi9yQWplZDM3TWI2LzgveU02Y3M4Nmludk0zaS9oajZ1K3dtN0lSODRBRC9VVTE3M1lCL3lMSGtkT3lKLzVkKytvenQvTzArMmZPbDZnRGZ3NE9kNzd4TVI1K3dQNEpUbjdNUG8xYzU5MUFQMWlyNjNxcjdzNCt3SDhPWDh3djY1WTM2Q0wvYjJjK2RadzV2M0UvQTdmdWFlUEFXdWVlLzlBSzlUdkcreGEzZ0NieDZIMktQMWxhb2RXY2R4NlBYNW9jbUYvem1YM2JSOEJUL3M5Nkg1amJyYXY1TTRMcENET2dhN2toLytBSTZ0N24vZjdPcjVtYnFTY3duWDIxWVBlZCtLL1F0NXlaT2ZtNTFaSDMzOTNFYzhlMzcwT0VGL3g0WFhyZUNKK3JYVmJkVXU0TmZyTXBjTDNIZ2VJMDdnaDk3a0Uvek1QZk1QTFM5K1lQTHczUEhIYzlZalRwRFArL0xJZy8rSUIreEUvZWQ1elBWamZlek5PUGhnTi9KSXUxbzhzUjd4aXQySkE3Y1g0NGdyOHRWbmxuOTJ6VTllMzRBZjhBaitzS2VmbitHRDNQaUhmSHZaK1B1K1JwenluSG5FaVovSE9BZWRzampmTWx4eWY4cm1leitKZGIxZjQvc1JldnM2am5QdWZaOWl2OFQrMkFOL1ZmKzJ2aXArMjNyNmZkdS9ITi9ZZ2Z5QjNjazdlNVlIMnZtNHJrZWU5bjdjS2RPRGVlUUo0bzcxc1J0OVd1Ump2TmM1ekdNYzlRcjI1dnpoZVIzOUhELzQ3N1RwaVI3RUdmVUtWK3hHbmJlcTc0ai84SnZuSyt5SlBNUS9jcEkvV0FlK0I3YnZNSjU5aTNxRzc0VGUzeVdlaU5kV045bSt4WGowbzkrQS8rSHJ2K2VCTC9waTEwMno3MWw3VDczTFBPb3A5Q1l2M3JUNXlJbjg0QmE3b2EvM1k4NFpMbTRadi92MVBYNUFMczZSZnA1R0h1VHpmaXc0OWQ4ZHNMK0FYNjlqZWU1MTNJN2xJZlJqUDlnMnV4eWFYYkNUNzhjdUwvbUllZHpERDN0NDN3ODdJeWYxUC9xMy9SeTVyUjRFMzhoSHZtQy9KWDR1bXQrUkc1emV0ZlhZTDFwL3RZNzMvZEh6KzU3RkNUalpzdmoyNzhEZ2crdXUyWmZuNE0vUC84UVhmdlh2T0x1R04rUnQ1enErUzFqK1lCeCs5M3dHTHMvYmUrclhHNVlQcUUrUTIrc0luc01IUDdoZFBmOGhIN2p3L2l0eHluN0E5emZ5bGU4YnZrOWNzRHpvOWlXZjhoNDkyempMQytqaDMxZjgvSG5GK0Y0eWU0Rno3T1BuZFBJbGVmZWE1UWZzalZ6RW9kZWR5TDlwK29FVDZ0MERpK2NkbXc5K3NTdDYrRHJVWmV3ZjhFRVBycTMrcmZmMEJkMytuT3ZjbjM2dUpKK1FON0FuOHo0ei95QW4vdlI0SVM2OFR0a3pmZkFYOWlNUEVwZm8wZXEwK3A2cnh3ZHhqaDQzTFAvalg2Nm5MRStCTi9BSFB1NWIvUHYzcmRZZnEzRkxmUEgra3ZrZi9mMjdsSjhqVnYxZWdmeEovR0FmM2grWVhxekRmbkxGL0VPK3hZK3NUMTdBRG81ei9BTisvTnlNdmZBWCtRWDgrYmtNdjdPZmt5ZlFnM1VjRDlnSC9KMHhlekRQNC9PY3JVTmNJai95WWhmZWN3OGZ4bk91dUZ2bHdsNnNVK2UxOHdtNGF2YXovRWg4dERxdzRvdHhCNVpQTGxvY1hEUDk2UGRnRit5TlBjbEQ0QVJjaytmQkVmejluTWc4OGpsNURQd3czdjFQdkZ5dy9MOWpjWWs4NEo4NHg3NSszdHEwZFE0c0Q0SVQ1SFgvWHpQL09HN3VXcDcwY3d6NUYzbXhHM1k4WSs4OXpyRVg2MkxIY3l2azh2TUs5K1NKczVhSHZCK0NIL0hQcnVXM1ZvOVdISEh1STA3WUQ4RUw4OXYzbFJvWHJBc3U0ZWY5Ty9CNjBYRDF2dUhTL2VtL2cwQWY1bU0vNVBIZmwyTW40czMzRGZSbXY4V2ZmcjY2WVhsbDMvSmRpNXNWL2FCRHc1dWZyOGkvck8vMVB2UFFuM1g5KzVlZkY3RWJmRmpualBrVnY0RjcrRkl2WUZmc0RvN0lDNTVIRGswKzdPTDdPZlVuOW16MVVmVUhlWk0rcE5jL24xZ2VwUitBM0w1L1lFZms1M2RIcS9aZnJxenI1MEdQdzlPR2h5M0RBZlpuZnlXUHdZZHh2bit6RDkyeHZNcDc5TWUreERGNnRYTnJmZTk5SVBoaEYrcGN6c0g0bVR6c2ZSbnlJL1lsdjNBUEx2QVg5dnF5OHZXK2p1TjR6L0laOHZnNUdyeUIxMVcvSy9OK0Ivcmk3OHNtTjM3Qi91MTNZdncrQlJ4Vyt6WjdXdDVobnVNYTNIbSthUFZJSFllZTROelBZMjMvdDNNejlpTE8yem15NHBGNEl4OFNqK2lEWE9TZEwrdDQ3RXI5UTU0L05Od1QxMzV1QnAvSWg5ellIeHh3dmxtMXYzbCtCTmZ0UEVUL3pzN3Y2TGRsZVhWN1JWdzJmdFhlM20veE92R2U1UTNpYThmeU5lTjlQOGV2NEo5MXNCUDZFcGNlMzFjTVY5aUY3M1dlTjd4ZmhMNzRCenN4RG42dFRxcHlFUS9vZThmOGhUKzQ5KzlteUFjT2VGOXhzTVozZE96UE9SWTlpV2ZzNXZIajlSLytRVTV3NXVkem5qT09lSEw4ZVY3Q2J6ZXRmc0NlNU5FZHc0blhRZGluZlkrdjlpYitpQ2YyUC9LcDV4ZmlIcjh5My92ZDRKUDZDanM3WHRHUGVvazg0dkhJUHNIK2hQN3d4NzZiRmgvb2kxM1FoM3ZmZjFtdjlTRnF2dlB6cGVkbmp6L3NBUTZKRi9aaitsUGdHYjZiRnEvSWVjM3dCbjdnanoxY24yM0w0ejdPNzczT3ZXUjV4cys5MkJrNXNCOTRKWTk0M3dLL2NQK0orZDNQaDEvVjkxNjMrRDVMdkpHWHZPNWpQUEt5N3NHS2ZJdmU0STk5MGV0dDZpVHdCZDU1ajcrd1c5dHZxejUrYnZWK1JNdVRGcGVPVXovZjdwbC9yNXNlMkxQbG0zcTlhbjd4dXBUNnlPc2QvMjZEUHF6SGVIRE9QUGNUZVFSNWtRLy9naHYyTTg5VDZJMWYyTDg5VHBuUE9SNC8rZm1wL2Y2WmM2UDEvNGd2NU5veVA0Tm44cGZ2My9nTmUvUCtrdUhqeTZvLzQrazNrYzhaangyOGorSGZWWWpuNjRZbjVPQWVPZUh6ME96dGZRcnNSWjNYdm92WDUreXI0QWg1UHJhNkZyN0VBM0plTUw3WWk3aHIrNXpWTC9nTi9Ea095RmZZaVR6c2VXbmI0cEo2bXZmKy9lZVU1VnZpbFR6aStkSDNjZFlEUHp4SFAvaDZ2a0IrNGdjOWlVL3dUcDJFZlZxOXd2ZkxpZ1AwTzdUMXNMdnZyOXg3WFg3TzlndS9zZzc3Q1BrZS9CSFBYTUVUNC8zM3FmaVYrRUNlWm8rYWIveThpTjdVVWUzN1FwV0grVisvOS9Udm8zZ1BQNjc0RGI3Z2tMb1QvdkNsdnNKZSs0YVRHK1lmNzVjd0gvOHlEM3Npcjh2cDhkTHFTenVIcnpwbkUvZGVEMjNadmZmL3ZWL0dlb3ozZW5mVmVjM3pwdmN4d0Q5MjhYck84eGx5NGlmdnYzZy9IYm53TjNIR3ZvcDg2T054QVQveUErY2kxc1ZPNEo5K0MvZStIL285T1BKLy80cjEzTjcrWFlmN1ZvOVovOUtmbzk4SEZpZlltZWZNWTd6THRXZDV4L010ZWRXL2Y1QTM5ODF1ck1QZnErTCtRZFViZjU2ek9QTDZDZm53KzdhOTU3bmpuYndDUHYxY0NDN3J2UGJkaVBjM3JGN3kvSS85ZUk2ZHNQK081Vm5zUlA3aGVzcjg3djBUOThmbTA5Y21OODgvc0hqMGZZNzFHRS9lUUI3MnB3T3pNMWZmejhFNStHSC84dm5JUVg3QWZ0N1hSVTl3Z3Y3MHQ1aFg2NXVtdjlzWHYzOVJ4N2Z2SG5ZZXBuL0l1Wkc0UWI4dHk5L1VnK0NjY2VTUlZYL2ZCL3l6ZjdUNnVkclB6L0hJZ2Yzd2w5ZmhmbzQ0WlhrTVBMQXZnMGZtODV6ODN2SjVYUjhjb1IvNnd0ZlBSYnduVHNqTDdLUFloK2ZiWmg4Lzk1K3l1SE04ZTV6c1dKejRkejNHNFFlZU00KzgxdmFYZXUrL3F5Tk8yam02MmdjOXVQcDNxY3EzNGRiM2NkWmR0VjlpOTRjV2w5akgveDRuKzVmSEdWZm0rVGtUUEYweVA0RUh4alBmejNYb2oxN1ltK2ZZRTc3MHYrQkhmSkYvUFI3WVA4a0hyQXVld1NIcmM3NWpmZnhIUG5MYytiblQ4eXJQcWVQeEczWmpmVCtmZS8zaWRyNWVuM3Y5eFg0R0RyeFBoSDJJVy9UWXN2SG9UZndoSDNrYnZIcWZCSGtkTjl3anA1L2p2UDZETC9rR0hGQmZvQS81aG53TERqaDM3Szd3RTNVRWZrVytUWlBybk9FRStmMmNnN3llRnhtUDM3dy80OTluNFhmZTR2K1U1V1A4Z2gydm01MlJsMzBEUDRCRDlEeHQrUUs5MGNmUFRaNjN5TmRmMi9rRy9IQU9PMlB5SU9lOUZiakg3cjVmRWY5ZTMrRm42amV2ODVDTGVkamQ2dzNpZ3p6a2VMOXM4US8vUTRzZjd2RS9WL1pocjRONWp0emc0ZEQ4elQ2RnZNaURmemZOWHRpYmZFUmRReDJFWGRpL3NLZi9lM0VmR283cS9UUDQ5RHFFK01UT3ZBZWZUYjhhdjlSWnhEUG5QY2NuY3NQMzlBcDdVeTk1SGdjdm5tZnhoKzlQMTAxKy9PRDFEWFltL3VGREhPRmZQMmMxZjlYMUdZKzgyTWY3QU13anJ1OVkzNU1yY25yZjVpT0xUK3ptL1RuNi8zNE85UHJEOHRjejhYVEY3TWdWUFp1ZmFoNi9ZM1hncnZtZnZIN040dURRN0VCZUlGL2hCOWJ6UGgxNkk2L25KZVJnUC9KNkF6d2dIK3Q2SG9JUDQ5bVhXQWMvOFJ6K0Y4MXZ2Q2RmSXBlZlA5SEwxMGN1emxOK1R2VGZNWHM5c0cxeGhQeSt6NU92OERmcmNnNWhQbjV6ZnA2SFcxL2Q3TUU5L2tNUHp4UHM0OXpEbjd3Qi90Q1Q4NkRuVzQ5UDlQSnpQbjVpUHV0N0g5MzdzNWFQMXZpT2pwN3dKVTdCQ2ZwN0g1ODhoUjNKMC9BQno5aU4rT1BxNTByVzhYNlY5NVc4anRnemZ5Q243Kyt0bjJUOUdlVDMzeld3UHZJaWwzOVB4Zy9iRm9kZS94NllQUHc5d0xzMXJ0bXZ3T01WMDVOejM2SGhnSE9ZNzJ2a1AvekxkeHZ5U091YlZydXQrbmZKUEMrM2ZhbytKNDdCQlhZa0gySTM0cDk2Z3Vma0d6OW40UWY4ejc2MForUGg1MysvenZWREQvWmR6Z1h3NVFvZTRPL241Vlg2b3gvbllMNUhlaDdCei9nTmYvTjdPT29xOGh2NUZQOWlsMjNMTzhUZmc2b1hjdUlINmxYa0lwOTRIOHJ6S2V0UWgzdWQ1dnVJNHcyKzZFdjh3SS83Wi9TdThZY2Z2Ri9ZL0xuaTNNVDZ4Qi81R0R3UkI5Z05uR0JQNGhxOHNTN3llUHo3NzJiY0hvekhidVFoOUNBUE9PNlJwK1hGT3A3MzlDdDVqMzhmV2IveHV1R1hmWWkrT2ZvMysxamU5dmhDYi9CUDNGQi9jODdDN3ZpVC9FWDhnVmZmTjF2L3N0b0ZmSGlmbmp6ZDVLM3hoVDI4bjR2OG5BdDREaDZKUjNCQlBpUnU5KzE4Z2IrOUgwUStvRjRnejRFTDY5TTk4ejM2b3ZuWjk1OVZ1TnMxUHVBWFBydUdML0kxK1IzN2ZHTDI4djBhdTdNKy9pSk95Yit0WDFUZlArTDhXZWY1ZVJYN3NxN1hBZGpyMFBRQzEvUVJ3QzM3QS9Zbm43TXVlTUV1THZldTVUdmtkRDk2WG1ZL0llN0FFWEVQUGx4Tytoazh4eS93Ung3L3ZyaHQrWWw0SUg2d28rZDFjTU1WL0dJWDhNczg0b1gzalBkK0MrdWlGM1VxY1VXZTUzemllZDN4N1Axa1AwZXRxdWQ4bjl5eE9zTHpNbGZzei80SnZ0dTF5b2tlTnkwUHdzZmxKdStBVS9SbjNobXpwK1Bzdk4xakovTGovWW92eG5FdUFOZmJoblA4d3hWOWtkZnRUM3p2bXI5Y2I1NGpCL2h0ZmRscU4rSUNYTEF2SEJydXNEdjVwLzE3VWhXSDdCdXMzLzdlcCsyempHTWUrYjdpby9XajBKZDgyL1lsK3NQVnJyd25EdHhlNk1HKzNmS08yV25mbnZ0NW03aDJ2SE9PSU4rZ3Y1Lzc4QU54UU43Z1NuMkdmSTdIWGNzL1p3M2Y4R2U4OTR1b044aG4xS1BrUVkrL1MyWUgzNGM4NytOUDlOMHgvQkNmK0xIRmRiWC9hZk9UNzBQWUV6M3hCL3lvbzhBVDUyajh5WGMwMXYyNjVrZjBRRzYzMjhPNkh0K1Y4RE4yWjMzcU0rekIxZnVBN0gvb2NjN3lKZS85SEkyL3lCdjRCVHQ0UFhISzNtTmYvMzdwK3dmNFlmOWdIT3VDUy94RGY0Vjc3SS9kUFYreUwzcitiUDkrV1gxKzEvU0VqOXZ0c2wzeDF5TTdONEJ6ci9kWW43amsvWUh0SjR6YnRyemcvTEF6ZWJyRmQ4VVRmbVU5L09yN09uaWhUdU9lZkVNOFkwL3c5OHpWOWhQa1JBNzZYNDhzM3JBdjlSTjZrRy9aQnhqSCtabjg4a3k5VXZuNmVaNzltSFdveitDRC9QNzM3OGdickx0djcvM2Y1MWlWUjRsUHpoUG9qMTNQMnp6eUIvcDd2NXdyK3BOSEdJOTgrQlc3OHQ3N3pjUWRkZ1NQSHErbnJaNENyNTdIc1VmckUxVys0QlY5VnZXWm1RZC84algzM2k5ODMvUkNmL1JpM3A3bEIvVG5mTXIraVAzQkRmYnd2aHh4eEwrdjJPcCsrbjUxbk9jdjVuTnU5VHlGWHFkdFBleEFQWFcveXNWNjdFL29RVjFFM1FpKzhROTZvKzlaMDlmeGpKeFhEY2RYelEvZVY5dXgvYnoxT1dvZFJoNzAzd05mTlh1d0R6cSs4Q3R5YjlyK2lWNmNXNURUK3d2dDkrRFZmc1MvMXduTVIyN3lybi9YQTRmVUJlQU5IT0cvRnE5Vkwrend3UGdnQjNxMGVzYnFadVNCbitlWjFxOHhQTkcvSWI3SjM5aWIrZng5QU9wazlyMkhoaGZHc3o3eGdSMWIvN3F1NitkQzlDRnZjMzVoSE9kbzhncDVCTDlpSDk5SHZTN3hjeEI4OEJmN2tmZC9rQVA4NGZjRHkyUG82K2ZVYXhabjZOSE80eFdQektjKzVJcisxSm40NmFISngzejA1cHpGdW51V1o3Yk5mK3lQRHlwLzZpZnM0K2VGNXZmNm5ucUMzOTh6SHp4akw2OEx5RTl1ZCt6RCtkTHpDZnNpT0FkMzRJRDlFdnNRaitSSjVwMDMrL2gzUmZUR2p0dm1kNjZuTGIvdm1QemdBUHY2T1BEQyt0NG5hSGl0OW0zN2s4VVI5dlY4NXQ4YjRYL1Q5TU5lNUhYeUFQdEIrdzVmNzlFZnY4RGYrd3pFRGZma0pYQk1QSlAzdGsydUR3eFg0SVM2N2REODN1S3Y0b0k0OHI4ZmhKelV3ZUNZZWhyKzJQMXV4VHYyUmkvMlplb0o5Z1B3NDNVay9uQjkyem00enZlK0F2N0ViK1JONzJ1Qjg2L3FsZnlHSE14akhmWWo5R2x5ODEyTi9rMTlmN2ZhQS80ZlcvMjg2dnpwZlFid2dQLytVT1hFam80bitIbS9Dbi83ZW43dXd2NjNuNjViN0grYy91THhZejk3NDUyWmVHbHQrVDlQZi9Md0cyK2ZQUEhqVnhkUDNYVlAzYVduN3JJeC9lYkpFNzk0VlJpLzhLQytlTDViYjlRQWxWS2pKcWpjTmFwdlZJSHEyOXkrelJnYTU2SE5HTnFNc21oVVc2M2tSclVaWllRYTI5dXhyVFl4N29YRitucVFYU01YOFhTeENISUlzZ1E1TmJMTGpVekJMQWV6SEF2MzhiU1BzWDF3NkdPMVBwWVl4a2FXR0RBMnZ0MjZrR09RalVNWHVuV0xGR1JNVzhTMHJpbmZwUnhrQ1RMR3BsZ2lkT3RDb2E2UEpmcmdNQVJaWWxxSmhVc3NNY2FBTVFhTVhaQXhkb3F4VTZnNWhUZ0JnbTVxb3FmMVJaQTV5QkprV3lLRkpWUGdJUzFrUVBBTmFLUXVucVloeU9BUWxrdzVwdVVZMEllUWZkTXREVUxHdENHV0dJSnZtRHFGelZMWUxJWE5VbGduaDNYeWVncXlEekxHaHZLNWl3RXRYY3hrQ1RMR3B2VWd1eURISUdOc0RobHlEQmhDeUNFV2pzREpKWjRHZHZJVTB3SWxmZmk0ajJEb1E0cys0cUlQY2Zwd1N4OXU2VXVRWTB3YlM1Qk55RDdFR1NKSERSR21ReGRQdzFCRGlEUGtJY2kyeEJDeE9VU0dHWVo0R2lnWlF0NmhCSWNJeUNHaWNCaGoyaFRpQkl5R01Hb0pHSlVBVE9sU2tERTJzRk1DSlNYRjJOQzRwQmdiS0NuaGxoSW9LV0dIRXM0cWtXbEx3S2hFT0pWSVZ5VU1WY0pRSlhKVUNYQ1ZNUWFFWTR1WVpHcDh4d2luY2IxNWFGeTBzV09ZWkF3Z2poRk9ZOXQxWnpLbUJSN0d3TU1ZdVdRTVBJeTlER2cyR3lPWGpLSDhHTzRlSTZkT0Fkb3AzRDNGSGp0RkFwbldaVm9Kc2kwOFJlaE5BWTBweGROdzl4VGVuR0x6bVVLM0tYdzhSV3hPb2RzVXVrM2gyS2tFczBnZ1U4VEZGSjZmSWk2bVNjaVkxbkxxa2JrQ3lVTDNRYmZVczZRWFFnOUNqMEYzTXFZVFBwMk03MlI4eXhwTE9nV2RaWHd2WXhvbVpucVE1NFBJWDRRZWhjOG9jeWVSY3hJK2s4Z3d4ZmpGZWhJNkM5MExIWG90RnAzUU1uNFJObDkwNjBMTEdMSFBJc21ZdEJCYTVFbkNNOHY0TEdPeWpPbUZUeS95OTdKdWtlZEZubzhpNXlnOEoxbFg3VGJGM0c2OUV6cm1kb0t4YmhGek83RkpKL2pwT3BtYjVIa1cvbG5XRmQyN1hzYUx2bDB2WXdhaFN4SmF4byt5bHRnaENyb2xMV3VKSFpMWUlhM0htQ1IyU0JKclNiQ1VrdkFSZlpQRVM5UmdTenJzbVVTWFZHUzg2SkpFbHl5WXo0THRMUEdlSmQ2ellES0xuRmx3bUhQNE5Jc3Zzc1IxSHVTNStDS1hkYUhEemxsaVBLdjg0b3RlNHJRWGUvYVN4M3JCWGk4NjloS252ZUF3cXE0bExmd2xCbnZSdmMvQ1gvRFpTMzdyZStFcG1PekZkNzNrdDE3czBBcysrekg4MGt0TzY5VW1nc2xCTURrSUpnZkpYWVBZWkJCOUI0bkJRWExSSVBnY3hPOVIvUzFwZVM3NWZCajB1YXdsZUJqS1FtZ1pVOExPdzVpRUZ2bW40QitWNEV4M1JlaFlxMGp1TGFKdlNUSmVNRit5akJjTUZQRjdFVHNVeWNsUkNpNXBrVlBpcFVpOEZNRi9rWDJ0U0I0ZUJmK2orSGRjMXpFaDJ5ajVaNVI0R1NWZVJ0bkx4azdHeUo0MUpuMHVjeVZHUnNIR0tQWVp4U2FqNkRzT01sY3dNMHJlRzR2U3dsTXdNRW9zakJJTGs4VEN0RkE2NUp3a0oweUNqVW53TUltT2srZzFDWjRud2Vja3VXc0szM1hyNGErWkhvTU9lV1pheHJjenpFd25HWk9FVDVMeEViUGRlcGJ4V2NaRWp1cld3LzR6TGZJVVdYZVV0VVlaRTdtb2t6cXFpMmJYa3RibnZkQWhqOVJPbmRST015M2pJeDY3YUhvdGFYbmV5M1BSYXpISVdrWG9VY1pFanUya0Z1bzY4VmNYY1RUVFdlaFlOeHBjUzdvVHVnZ2RNbmVCcTA1cW5wa08yMGJMYTBuTFdyM0lPWWljZ3p3WFAzYWlleWY2cHZXUVFXcVZMaHBQTXgxN1JDZDFTNWV5akJlOEpmRkxpbmp2cEo3cGtzZ1RuYUlsTFR3anJydG9GaTNwMERjTDNyTElMN1ZORjEyZ0pSMzJrWHBtcG1VdHdVLzBmNVowK0NnWHBXV3UyRFk2UDBlNlBuTEw3RktsUTRaZS9Odkx1djBnZkdSZHFSL21FbmdRT216YlR6SmVZcllYMllaRlBCOEVxNE5nZFJBYkRwS0xCc2t0dzZCMDJGbjI5MDcyOTI2UUdCekVia1BVc1RPdDQ0Vy94T2tnK0JrazM1Ykkvek1kTXBlRjByM1FnOURDUi9KU0Vac1VpUXVwRXpxcERUcXBEVHFwQWJvaS9pMkNxMUpFQm9tUklyYUsvczlNeDlta2s5cGdwc00rc3RmUGRQQ01mczlNUy80WnhhZXlkOCswOEJlWlI5azdSdkg3S0w0ZWl6d1gvNDZpeXlnK0hRV2ZZL1EzdWttd0tudjNUQWZQS1NrdGN5WGZUaEozMGVSWjByM1FZWWRKY3NJa09rNmkxeVI2U1QwdzA4MCtTZm96U2VxQnRCNDRuT2twNk5BclNUMHcwMFZvNFo5a2ZCS2VnY09aWGdndDh2UWl6eUJ6QitGZlpFd1JHVVpaZDVReDRic2tkVUtTZnN0TUQwSUhUK205SktrZmt0UUpNeTNqSTBmTnRJeEpJa09XdFhyaEh4aElVa3VrUmEvamhjOGdhdzB5ZDVEeFJjWVU0U2wyVzR4S2g1MmxEelBUb2t2VXQ2a1RlM1lSNDBscWo1bnVoUTc1TzdGUEp6aVIybU9taGFkZ3BoTzdkV0tIYmhDZWc2d2x1bmVDSCtuRHBHNlVkVVgzTHZKNWt0NUxpaTluTXkxK1Q5RzdTTkpYbVdsOW5vVVdQaElMU2ZSS2thdG5PblJKNHJza1BrcVJrMmM2eGtoOWtxVGZNdE5aNkppYkpaWnpGajZEMEJKMzBqTkpXZGJ0bzNaSzBqK1o2ZEJkK2llcGw1anFSUWJwazh5bWt2RmlRK2wxcEg2UXVZS05YdkpKTHpqcEpWNzZJaklYNFJNMWJaS2FKMG5mWTZhVDBGbm9XR3NRZmFYT21VdlVoZEJocTBId00yVGhJOWlXZmtXU0dpWko3eUlOazR3WEh3MlNKNlZ1U1VVd1V4YjZQSGdXaVhmcGV5VHBkYVFpY1NIOWpWUmlUNXpwTFBRZ3RJeVhQQ0M5aTFRa0J4YXhpZFF6TXkwNmluMksyRVRxbVNSOWp6UUtocVcva2FUT1NWTGJKS2x0WmxyR1M2NlFQa1lhcy9BWDNhWCttZW5RY1pROE1Fbzhqb0xoTWVyem1SYWVvL0NVbUowa0owd2k4eVMrbXdTVDBxT1kwNC9NRlh4T2t0TW04Y3NrT1hhS3ZKcmxPOUZNZDBKUFFYY3lKbXFTTE4rSnNud0R5dEt2bUdsWkt3blB3RjZXSGtXVzcwVHpzVS9HRjFtckNQOGkvRWZoT1FxZlNXU0x1SnVQa2d1aFF3YXBRN0wwSzJaYTVvcStVbnZNZE1ncDMydnlZbEJheG91T2l5TDBLUHhGbDRYbzBxMHJIVGFSM3NWTUI4OU9kSkU2SVV1L0lrdWRrS1ZPeVBHem5abU9mWGFtNVhrdk1nd2ltK2pZRlprN3luUFZheElaQktzcFlqL0xONWVaVGtMTCtNaVRNeDI2Sk5GRnZyL010UENQT01wU0c4eTA4Qzh5cGdoUDBVWHFoQ3kxd1V3SG54dzltWm5PUWc5Q3gxenBkZVFzdnN2aXV5dzZTaTJSczhSYUZoMXpMenhGWDZrOVpqWWlqMkJWdnVQTXRJeVAralpMVFRMVFNlaEJhQmt2TVNpOWxDejFTWmI2WktabHJ1UVorYWFUcFliSjh1MW1wa05tNmNOazZiZk10TXdkaGFmbzIwL3lYUHd1MzJWbXVnZ2RjZzZpN3hEN3drekw4NlMwOE14S0MwK0p4MkVRV21KUWFwZ3N2WmNzMzF5eTlGNnk5RjV5L0ZSblNZZSs4bDBtU3cyVHBRK1RwZmVTcFplU3BXZVNpK1NORXZ0c2poL01MR2taTHptenFEeVM4NlYreUtQa0VQbFdrcVd1bU9td2czd3J5YVBrL3pGNmlWbStnK1JSWW1xVXZXa1VPMHM5a09XN1JwWnZGak1kWXliWmZ5ZkphWlBzVFZJL1pLa2ZaanJra2Q1SWx0NUlubVRQaXQrNnpQUWdmQVEvay9oQzZvMDhTWXhJYnlSTGJ5VExkNU4rUGZiZlhub21jOXBJUXN2ejJNdG1XdmdrZVo2RVoxWmErUFJLOTBJWG9ZWC9JR01HR1JQN1FyOWVSSVlpNnhhWk84cnpVWFFjWlV6Z29aZmZzZlRTUCttbGY5Skx6NlNYMzZ2MDhydVVYbXFWWHI2ejlBdXh5U0lMVDdHSi9NNmtsMzVJTC8yUW1aYnhjODZ4MzVtLy90YXhkL2lkT1ErUC91VFl1OGRlL2VuSll6OTdZNzc3czAzNWl4TnZ2L3ZtaWVQenBPZS92ZVJnazU4N2FROWUrdm54NVcvT2YvTEs2Ly8wOCtQLy9NcWNHdWNWSHI5Lzh0OWZQcm0rZEMzb2VlYmpOWi8vdjhycnhjcnJHMjhjLzhjM2o3K0I4RzhkKy9FYmI5V2JiNTA4OFl2SHYyMS85ZTJUYng1L0YxWG1wKys4K3U2SmQ0OHg3dWpySjk3aXlXUGwxdjc4L3hDeDBRVFJyQUFBIn0= -->

<div data-pagedtable="false">
  <script data-pagedtable-source type="application/json">
{"columns":[{"label":[""],"name":["_rn_"],"type":[""],"align":["left"]},{"label":["prob.1"],"name":[1],"type":["dbl"],"align":["right"]},{"label":["prob.2"],"name":[2],"type":["dbl"],"align":["right"]},{"label":["prob.3"],"name":[3],"type":["dbl"],"align":["right"]},{"label":["prob.4"],"name":[4],"type":["dbl"],"align":["right"]}],"data":[{"1":"0.33","2":"0.27","3":"0.14","4":"0.26","_rn_":"20"},{"1":"0.16","2":"0.15","3":"0.46","4":"0.23","_rn_":"26"},{"1":"0.09","2":"0.01","3":"0.05","4":"0.85","_rn_":"33"},{"1":"0.08","2":"0.17","3":"0.56","4":"0.19","_rn_":"39"},{"1":"0.43","2":"0.21","3":"0.16","4":"0.20","_rn_":"42"},{"1":"0.20","2":"0.09","3":"0.58","4":"0.13","_rn_":"45"},{"1":"0.34","2":"0.29","3":"0.15","4":"0.22","_rn_":"47"},{"1":"0.35","2":"0.32","3":"0.08","4":"0.25","_rn_":"56"},{"1":"0.48","2":"0.25","3":"0.09","4":"0.18","_rn_":"59"},{"1":"0.00","2":"0.06","3":"0.32","4":"0.62","_rn_":"60"},{"1":"0.07","2":"0.06","3":"0.64","4":"0.23","_rn_":"65"},{"1":"0.46","2":"0.19","3":"0.21","4":"0.14","_rn_":"67"},{"1":"0.26","2":"0.44","3":"0.13","4":"0.17","_rn_":"71"},{"1":"0.06","2":"0.24","3":"0.42","4":"0.28","_rn_":"73"},{"1":"0.11","2":"0.40","3":"0.21","4":"0.28","_rn_":"74"},{"1":"0.07","2":"0.13","3":"0.64","4":"0.16","_rn_":"75"},{"1":"0.09","2":"0.24","3":"0.37","4":"0.30","_rn_":"78"},{"1":"0.12","2":"0.59","3":"0.09","4":"0.20","_rn_":"84"},{"1":"0.55","2":"0.22","3":"0.04","4":"0.19","_rn_":"86"},{"1":"0.59","2":"0.12","3":"0.04","4":"0.25","_rn_":"95"},{"1":"0.15","2":"0.53","3":"0.15","4":"0.17","_rn_":"100"},{"1":"0.16","2":"0.39","3":"0.31","4":"0.14","_rn_":"102"},{"1":"0.12","2":"0.35","3":"0.48","4":"0.05","_rn_":"110"},{"1":"0.15","2":"0.56","3":"0.22","4":"0.07","_rn_":"111"},{"1":"0.15","2":"0.67","3":"0.02","4":"0.16","_rn_":"116"},{"1":"0.25","2":"0.41","3":"0.19","4":"0.15","_rn_":"117"},{"1":"0.66","2":"0.13","3":"0.08","4":"0.13","_rn_":"119"},{"1":"0.12","2":"0.52","3":"0.30","4":"0.06","_rn_":"124"},{"1":"0.19","2":"0.62","3":"0.05","4":"0.14","_rn_":"132"},{"1":"0.08","2":"0.08","3":"0.49","4":"0.35","_rn_":"140"},{"1":"0.51","2":"0.06","3":"0.06","4":"0.37","_rn_":"145"},{"1":"0.03","2":"0.53","3":"0.35","4":"0.09","_rn_":"150"},{"1":"0.00","2":"0.28","3":"0.54","4":"0.18","_rn_":"152"},{"1":"0.08","2":"0.09","3":"0.72","4":"0.11","_rn_":"154"},{"1":"0.04","2":"0.54","3":"0.39","4":"0.03","_rn_":"156"},{"1":"0.30","2":"0.07","3":"0.38","4":"0.25","_rn_":"159"},{"1":"0.20","2":"0.12","3":"0.40","4":"0.28","_rn_":"168"},{"1":"0.56","2":"0.15","3":"0.10","4":"0.19","_rn_":"176"},{"1":"0.35","2":"0.13","3":"0.05","4":"0.47","_rn_":"185"},{"1":"0.20","2":"0.04","3":"0.67","4":"0.09","_rn_":"205"},{"1":"0.38","2":"0.23","3":"0.29","4":"0.10","_rn_":"208"},{"1":"0.18","2":"0.04","3":"0.72","4":"0.06","_rn_":"209"},{"1":"0.29","2":"0.18","3":"0.29","4":"0.24","_rn_":"210"},{"1":"0.10","2":"0.30","3":"0.39","4":"0.21","_rn_":"213"},{"1":"0.42","2":"0.23","3":"0.07","4":"0.28","_rn_":"215"},{"1":"0.27","2":"0.16","3":"0.33","4":"0.24","_rn_":"218"},{"1":"0.33","2":"0.07","3":"0.46","4":"0.14","_rn_":"221"},{"1":"0.09","2":"0.03","3":"0.09","4":"0.79","_rn_":"234"},{"1":"0.08","2":"0.20","3":"0.06","4":"0.66","_rn_":"237"},{"1":"0.04","2":"0.11","3":"0.31","4":"0.54","_rn_":"238"},{"1":"0.13","2":"0.15","3":"0.36","4":"0.36","_rn_":"239"},{"1":"0.08","2":"0.12","3":"0.60","4":"0.20","_rn_":"250"},{"1":"0.35","2":"0.54","3":"0.04","4":"0.07","_rn_":"254"},{"1":"0.05","2":"0.05","3":"0.03","4":"0.87","_rn_":"255"},{"1":"0.08","2":"0.05","3":"0.55","4":"0.32","_rn_":"257"},{"1":"0.09","2":"0.25","3":"0.43","4":"0.23","_rn_":"267"},{"1":"0.04","2":"0.35","3":"0.02","4":"0.59","_rn_":"270"},{"1":"0.23","2":"0.38","3":"0.33","4":"0.06","_rn_":"271"},{"1":"0.30","2":"0.42","3":"0.21","4":"0.07","_rn_":"278"},{"1":"0.53","2":"0.07","3":"0.01","4":"0.39","_rn_":"280"},{"1":"0.44","2":"0.23","3":"0.07","4":"0.26","_rn_":"281"},{"1":"0.05","2":"0.09","3":"0.35","4":"0.51","_rn_":"282"},{"1":"0.33","2":"0.04","3":"0.30","4":"0.33","_rn_":"288"},{"1":"0.10","2":"0.57","3":"0.10","4":"0.23","_rn_":"290"},{"1":"0.08","2":"0.32","3":"0.30","4":"0.30","_rn_":"293"},{"1":"0.26","2":"0.07","3":"0.06","4":"0.61","_rn_":"294"},{"1":"0.37","2":"0.01","3":"0.04","4":"0.58","_rn_":"295"},{"1":"0.04","2":"0.88","3":"0.03","4":"0.05","_rn_":"299"},{"1":"0.06","2":"0.48","3":"0.33","4":"0.13","_rn_":"301"},{"1":"0.15","2":"0.03","3":"0.47","4":"0.35","_rn_":"304"},{"1":"0.44","2":"0.23","3":"0.09","4":"0.24","_rn_":"307"},{"1":"0.16","2":"0.53","3":"0.05","4":"0.26","_rn_":"308"},{"1":"0.36","2":"0.21","3":"0.31","4":"0.12","_rn_":"310"},{"1":"0.47","2":"0.22","3":"0.09","4":"0.22","_rn_":"317"},{"1":"0.09","2":"0.74","3":"0.06","4":"0.11","_rn_":"318"},{"1":"0.06","2":"0.55","3":"0.08","4":"0.31","_rn_":"319"},{"1":"0.13","2":"0.39","3":"0.29","4":"0.19","_rn_":"324"},{"1":"0.21","2":"0.06","3":"0.18","4":"0.55","_rn_":"329"},{"1":"0.13","2":"0.19","3":"0.26","4":"0.42","_rn_":"336"},{"1":"0.11","2":"0.04","3":"0.66","4":"0.19","_rn_":"337"},{"1":"0.11","2":"0.26","3":"0.61","4":"0.02","_rn_":"339"},{"1":"0.15","2":"0.13","3":"0.53","4":"0.19","_rn_":"344"},{"1":"0.03","2":"0.77","3":"0.05","4":"0.15","_rn_":"347"},{"1":"0.02","2":"0.94","3":"0.01","4":"0.03","_rn_":"351"},{"1":"0.18","2":"0.34","3":"0.31","4":"0.17","_rn_":"353"},{"1":"0.03","2":"0.03","3":"0.30","4":"0.64","_rn_":"363"},{"1":"0.33","2":"0.15","3":"0.36","4":"0.16","_rn_":"364"},{"1":"0.26","2":"0.23","3":"0.32","4":"0.19","_rn_":"366"},{"1":"0.08","2":"0.13","3":"0.37","4":"0.42","_rn_":"369"},{"1":"0.14","2":"0.07","3":"0.18","4":"0.61","_rn_":"370"},{"1":"0.33","2":"0.51","3":"0.02","4":"0.14","_rn_":"388"},{"1":"0.08","2":"0.63","3":"0.06","4":"0.23","_rn_":"390"},{"1":"0.03","2":"0.63","3":"0.04","4":"0.30","_rn_":"393"},{"1":"0.23","2":"0.58","3":"0.06","4":"0.13","_rn_":"399"},{"1":"0.17","2":"0.09","3":"0.29","4":"0.45","_rn_":"401"},{"1":"0.03","2":"0.42","3":"0.02","4":"0.53","_rn_":"403"},{"1":"0.01","2":"0.29","3":"0.06","4":"0.64","_rn_":"405"},{"1":"0.03","2":"0.31","3":"0.02","4":"0.64","_rn_":"409"},{"1":"0.37","2":"0.39","3":"0.06","4":"0.18","_rn_":"419"},{"1":"0.23","2":"0.49","3":"0.09","4":"0.19","_rn_":"425"},{"1":"0.12","2":"0.50","3":"0.09","4":"0.29","_rn_":"426"},{"1":"0.46","2":"0.35","3":"0.14","4":"0.05","_rn_":"427"},{"1":"0.52","2":"0.15","3":"0.09","4":"0.24","_rn_":"429"},{"1":"0.31","2":"0.22","3":"0.21","4":"0.26","_rn_":"430"},{"1":"0.57","2":"0.23","3":"0.08","4":"0.12","_rn_":"432"},{"1":"0.32","2":"0.06","3":"0.07","4":"0.55","_rn_":"438"},{"1":"0.11","2":"0.02","3":"0.18","4":"0.69","_rn_":"439"},{"1":"0.17","2":"0.12","3":"0.03","4":"0.68","_rn_":"443"},{"1":"0.19","2":"0.07","3":"0.48","4":"0.26","_rn_":"448"},{"1":"0.35","2":"0.25","3":"0.20","4":"0.20","_rn_":"461"},{"1":"0.29","2":"0.06","3":"0.11","4":"0.54","_rn_":"465"},{"1":"0.30","2":"0.04","3":"0.07","4":"0.59","_rn_":"468"},{"1":"0.13","2":"0.16","3":"0.35","4":"0.36","_rn_":"475"},{"1":"0.03","2":"0.07","3":"0.07","4":"0.83","_rn_":"482"},{"1":"0.07","2":"0.20","3":"0.30","4":"0.43","_rn_":"491"},{"1":"0.51","2":"0.07","3":"0.04","4":"0.38","_rn_":"494"},{"1":"0.02","2":"0.18","3":"0.04","4":"0.76","_rn_":"510"},{"1":"0.04","2":"0.79","3":"0.11","4":"0.06","_rn_":"521"},{"1":"0.08","2":"0.04","3":"0.85","4":"0.03","_rn_":"526"},{"1":"0.46","2":"0.34","3":"0.14","4":"0.06","_rn_":"534"},{"1":"0.31","2":"0.28","3":"0.26","4":"0.15","_rn_":"543"},{"1":"0.17","2":"0.08","3":"0.13","4":"0.62","_rn_":"553"},{"1":"0.04","2":"0.13","3":"0.05","4":"0.78","_rn_":"563"},{"1":"0.11","2":"0.23","3":"0.42","4":"0.24","_rn_":"573"},{"1":"0.13","2":"0.06","3":"0.50","4":"0.31","_rn_":"584"},{"1":"0.40","2":"0.05","3":"0.06","4":"0.49","_rn_":"587"},{"1":"0.11","2":"0.10","3":"0.11","4":"0.68","_rn_":"588"},{"1":"0.72","2":"0.07","3":"0.13","4":"0.08","_rn_":"591"},{"1":"0.17","2":"0.52","3":"0.18","4":"0.13","_rn_":"600"},{"1":"0.16","2":"0.34","3":"0.30","4":"0.20","_rn_":"615"},{"1":"0.20","2":"0.49","3":"0.24","4":"0.07","_rn_":"620"},{"1":"0.10","2":"0.14","3":"0.32","4":"0.44","_rn_":"632"},{"1":"0.11","2":"0.53","3":"0.30","4":"0.06","_rn_":"634"},{"1":"0.25","2":"0.14","3":"0.09","4":"0.52","_rn_":"646"},{"1":"0.32","2":"0.10","3":"0.55","4":"0.03","_rn_":"647"},{"1":"0.42","2":"0.32","3":"0.12","4":"0.14","_rn_":"650"},{"1":"0.60","2":"0.10","3":"0.17","4":"0.13","_rn_":"656"},{"1":"0.65","2":"0.17","3":"0.13","4":"0.05","_rn_":"660"},{"1":"0.12","2":"0.28","3":"0.34","4":"0.26","_rn_":"669"},{"1":"0.29","2":"0.10","3":"0.10","4":"0.51","_rn_":"673"},{"1":"0.11","2":"0.23","3":"0.46","4":"0.20","_rn_":"677"},{"1":"0.19","2":"0.09","3":"0.13","4":"0.59","_rn_":"678"},{"1":"0.06","2":"0.10","3":"0.37","4":"0.47","_rn_":"681"},{"1":"0.06","2":"0.34","3":"0.08","4":"0.52","_rn_":"689"},{"1":"0.13","2":"0.30","3":"0.10","4":"0.47","_rn_":"692"},{"1":"0.24","2":"0.05","3":"0.28","4":"0.43","_rn_":"693"},{"1":"0.16","2":"0.38","3":"0.08","4":"0.38","_rn_":"694"},{"1":"0.04","2":"0.18","3":"0.03","4":"0.75","_rn_":"701"},{"1":"0.10","2":"0.49","3":"0.22","4":"0.19","_rn_":"709"},{"1":"0.09","2":"0.58","3":"0.12","4":"0.21","_rn_":"723"},{"1":"0.12","2":"0.54","3":"0.10","4":"0.24","_rn_":"724"},{"1":"0.12","2":"0.20","3":"0.55","4":"0.13","_rn_":"725"},{"1":"0.09","2":"0.37","3":"0.10","4":"0.44","_rn_":"729"},{"1":"0.06","2":"0.27","3":"0.33","4":"0.34","_rn_":"733"},{"1":"0.12","2":"0.15","3":"0.47","4":"0.26","_rn_":"734"},{"1":"0.18","2":"0.07","3":"0.24","4":"0.51","_rn_":"735"},{"1":"0.09","2":"0.52","3":"0.11","4":"0.28","_rn_":"739"},{"1":"0.29","2":"0.09","3":"0.16","4":"0.46","_rn_":"743"},{"1":"0.61","2":"0.16","3":"0.06","4":"0.17","_rn_":"748"},{"1":"0.12","2":"0.15","3":"0.40","4":"0.33","_rn_":"750"},{"1":"0.06","2":"0.20","3":"0.12","4":"0.62","_rn_":"753"},{"1":"0.43","2":"0.13","3":"0.14","4":"0.30","_rn_":"759"},{"1":"0.29","2":"0.41","3":"0.15","4":"0.15","_rn_":"761"},{"1":"0.19","2":"0.41","3":"0.30","4":"0.10","_rn_":"764"},{"1":"0.14","2":"0.42","3":"0.28","4":"0.16","_rn_":"767"},{"1":"0.39","2":"0.07","3":"0.30","4":"0.24","_rn_":"769"},{"1":"0.42","2":"0.37","3":"0.02","4":"0.19","_rn_":"773"},{"1":"0.78","2":"0.11","3":"0.02","4":"0.09","_rn_":"780"},{"1":"0.12","2":"0.08","3":"0.70","4":"0.10","_rn_":"782"},{"1":"0.22","2":"0.21","3":"0.18","4":"0.39","_rn_":"783"},{"1":"0.27","2":"0.53","3":"0.13","4":"0.07","_rn_":"791"},{"1":"0.70","2":"0.14","3":"0.11","4":"0.05","_rn_":"794"},{"1":"0.25","2":"0.11","3":"0.38","4":"0.26","_rn_":"797"},{"1":"0.87","2":"0.01","3":"0.03","4":"0.09","_rn_":"805"},{"1":"0.07","2":"0.67","3":"0.12","4":"0.14","_rn_":"806"},{"1":"0.06","2":"0.39","3":"0.18","4":"0.37","_rn_":"814"},{"1":"0.58","2":"0.28","3":"0.01","4":"0.13","_rn_":"825"},{"1":"0.09","2":"0.12","3":"0.13","4":"0.66","_rn_":"826"},{"1":"0.86","2":"0.05","3":"0.06","4":"0.03","_rn_":"827"},{"1":"0.63","2":"0.09","3":"0.11","4":"0.17","_rn_":"842"},{"1":"0.32","2":"0.14","3":"0.39","4":"0.15","_rn_":"844"},{"1":"0.16","2":"0.25","3":"0.40","4":"0.19","_rn_":"846"},{"1":"0.09","2":"0.50","3":"0.15","4":"0.26","_rn_":"847"},{"1":"0.19","2":"0.22","3":"0.42","4":"0.17","_rn_":"851"},{"1":"0.01","2":"0.10","3":"0.07","4":"0.82","_rn_":"856"},{"1":"0.29","2":"0.25","3":"0.27","4":"0.19","_rn_":"857"},{"1":"0.06","2":"0.16","3":"0.10","4":"0.68","_rn_":"858"},{"1":"0.02","2":"0.05","3":"0.05","4":"0.88","_rn_":"866"},{"1":"0.16","2":"0.07","3":"0.16","4":"0.61","_rn_":"869"},{"1":"0.09","2":"0.17","3":"0.34","4":"0.40","_rn_":"892"},{"1":"0.11","2":"0.24","3":"0.41","4":"0.24","_rn_":"895"},{"1":"0.16","2":"0.18","3":"0.06","4":"0.60","_rn_":"900"},{"1":"0.05","2":"0.14","3":"0.08","4":"0.73","_rn_":"901"},{"1":"0.36","2":"0.07","3":"0.27","4":"0.30","_rn_":"902"},{"1":"0.10","2":"0.05","3":"0.68","4":"0.17","_rn_":"903"},{"1":"0.10","2":"0.19","3":"0.37","4":"0.34","_rn_":"905"},{"1":"0.49","2":"0.07","3":"0.16","4":"0.28","_rn_":"907"},{"1":"0.06","2":"0.24","3":"0.09","4":"0.61","_rn_":"908"},{"1":"0.04","2":"0.31","3":"0.51","4":"0.14","_rn_":"921"},{"1":"0.48","2":"0.06","3":"0.36","4":"0.10","_rn_":"924"},{"1":"0.43","2":"0.11","3":"0.34","4":"0.12","_rn_":"931"},{"1":"0.33","2":"0.46","3":"0.03","4":"0.18","_rn_":"939"},{"1":"0.05","2":"0.03","3":"0.29","4":"0.63","_rn_":"953"},{"1":"0.07","2":"0.19","3":"0.07","4":"0.67","_rn_":"954"},{"1":"0.50","2":"0.02","3":"0.02","4":"0.46","_rn_":"956"},{"1":"0.02","2":"0.13","3":"0.10","4":"0.75","_rn_":"959"},{"1":"0.62","2":"0.02","3":"0.04","4":"0.32","_rn_":"963"},{"1":"0.01","2":"0.03","3":"0.43","4":"0.53","_rn_":"966"},{"1":"0.04","2":"0.11","3":"0.07","4":"0.78","_rn_":"969"},{"1":"0.56","2":"0.13","3":"0.09","4":"0.22","_rn_":"973"},{"1":"0.19","2":"0.11","3":"0.24","4":"0.46","_rn_":"974"},{"1":"0.13","2":"0.07","3":"0.30","4":"0.50","_rn_":"975"},{"1":"0.58","2":"0.02","3":"0.07","4":"0.33","_rn_":"977"},{"1":"0.32","2":"0.11","3":"0.14","4":"0.43","_rn_":"980"},{"1":"0.54","2":"0.21","3":"0.08","4":"0.17","_rn_":"981"},{"1":"0.04","2":"0.09","3":"0.84","4":"0.03","_rn_":"991"},{"1":"0.26","2":"0.25","3":"0.24","4":"0.25","_rn_":"995"},{"1":"0.05","2":"0.02","3":"0.59","4":"0.34","_rn_":"999"},{"1":"0.06","2":"0.13","3":"0.54","4":"0.27","_rn_":"1004"},{"1":"0.57","2":"0.10","3":"0.14","4":"0.19","_rn_":"1005"},{"1":"0.06","2":"0.07","3":"0.05","4":"0.82","_rn_":"1010"},{"1":"0.69","2":"0.10","3":"0.06","4":"0.15","_rn_":"1011"},{"1":"0.44","2":"0.06","3":"0.36","4":"0.14","_rn_":"1016"},{"1":"0.13","2":"0.06","3":"0.25","4":"0.56","_rn_":"1018"},{"1":"0.06","2":"0.69","3":"0.10","4":"0.15","_rn_":"1021"},{"1":"0.15","2":"0.12","3":"0.21","4":"0.52","_rn_":"1025"},{"1":"0.10","2":"0.14","3":"0.40","4":"0.36","_rn_":"1026"},{"1":"0.04","2":"0.01","3":"0.33","4":"0.62","_rn_":"1028"},{"1":"0.04","2":"0.08","3":"0.30","4":"0.58","_rn_":"1032"},{"1":"0.13","2":"0.00","3":"0.33","4":"0.54","_rn_":"1033"},{"1":"0.30","2":"0.39","3":"0.09","4":"0.22","_rn_":"1046"},{"1":"0.08","2":"0.02","3":"0.36","4":"0.54","_rn_":"1052"},{"1":"0.05","2":"0.08","3":"0.80","4":"0.07","_rn_":"1057"},{"1":"0.18","2":"0.59","3":"0.05","4":"0.18","_rn_":"1062"},{"1":"0.57","2":"0.10","3":"0.13","4":"0.20","_rn_":"1064"},{"1":"0.03","2":"0.34","3":"0.08","4":"0.55","_rn_":"1074"},{"1":"0.26","2":"0.19","3":"0.51","4":"0.04","_rn_":"1086"},{"1":"0.19","2":"0.35","3":"0.44","4":"0.02","_rn_":"1087"},{"1":"0.20","2":"0.18","3":"0.57","4":"0.05","_rn_":"1091"},{"1":"0.12","2":"0.29","3":"0.51","4":"0.08","_rn_":"1092"},{"1":"0.21","2":"0.16","3":"0.58","4":"0.05","_rn_":"1093"},{"1":"0.08","2":"0.16","3":"0.70","4":"0.06","_rn_":"1097"},{"1":"0.76","2":"0.04","3":"0.10","4":"0.10","_rn_":"1103"},{"1":"0.08","2":"0.50","3":"0.29","4":"0.13","_rn_":"1104"},{"1":"0.16","2":"0.57","3":"0.10","4":"0.17","_rn_":"1105"},{"1":"0.16","2":"0.52","3":"0.13","4":"0.19","_rn_":"1108"},{"1":"0.23","2":"0.08","3":"0.22","4":"0.47","_rn_":"1112"},{"1":"0.47","2":"0.03","3":"0.40","4":"0.10","_rn_":"1114"},{"1":"0.44","2":"0.06","3":"0.11","4":"0.39","_rn_":"1119"},{"1":"0.32","2":"0.04","3":"0.57","4":"0.07","_rn_":"1120"},{"1":"0.02","2":"0.79","3":"0.07","4":"0.12","_rn_":"1124"},{"1":"0.21","2":"0.23","3":"0.46","4":"0.10","_rn_":"1128"},{"1":"0.30","2":"0.19","3":"0.35","4":"0.16","_rn_":"1130"},{"1":"0.07","2":"0.21","3":"0.55","4":"0.17","_rn_":"1131"},{"1":"0.10","2":"0.75","3":"0.03","4":"0.12","_rn_":"1133"},{"1":"0.15","2":"0.34","3":"0.34","4":"0.17","_rn_":"1139"},{"1":"0.09","2":"0.79","3":"0.01","4":"0.11","_rn_":"1140"},{"1":"0.49","2":"0.08","3":"0.08","4":"0.35","_rn_":"1143"},{"1":"0.49","2":"0.10","3":"0.27","4":"0.14","_rn_":"1149"},{"1":"0.31","2":"0.19","3":"0.20","4":"0.30","_rn_":"1151"},{"1":"0.27","2":"0.34","3":"0.25","4":"0.14","_rn_":"1155"},{"1":"0.06","2":"0.47","3":"0.31","4":"0.16","_rn_":"1158"},{"1":"0.41","2":"0.25","3":"0.04","4":"0.30","_rn_":"1175"},{"1":"0.28","2":"0.43","3":"0.04","4":"0.25","_rn_":"1178"},{"1":"0.49","2":"0.06","3":"0.06","4":"0.39","_rn_":"1184"},{"1":"0.21","2":"0.19","3":"0.10","4":"0.50","_rn_":"1189"},{"1":"0.16","2":"0.18","3":"0.45","4":"0.21","_rn_":"1190"},{"1":"0.54","2":"0.19","3":"0.04","4":"0.23","_rn_":"1197"},{"1":"0.35","2":"0.07","3":"0.45","4":"0.13","_rn_":"1198"},{"1":"0.07","2":"0.11","3":"0.57","4":"0.25","_rn_":"1202"},{"1":"0.03","2":"0.04","3":"0.03","4":"0.90","_rn_":"1209"},{"1":"0.55","2":"0.04","3":"0.02","4":"0.39","_rn_":"1210"},{"1":"0.39","2":"0.03","3":"0.47","4":"0.11","_rn_":"1217"},{"1":"0.22","2":"0.42","3":"0.17","4":"0.19","_rn_":"1224"},{"1":"0.07","2":"0.60","3":"0.11","4":"0.22","_rn_":"1226"},{"1":"0.19","2":"0.60","3":"0.07","4":"0.14","_rn_":"1229"},{"1":"0.10","2":"0.35","3":"0.30","4":"0.25","_rn_":"1236"},{"1":"0.42","2":"0.36","3":"0.05","4":"0.17","_rn_":"1242"},{"1":"0.76","2":"0.13","3":"0.06","4":"0.05","_rn_":"1248"},{"1":"0.14","2":"0.11","3":"0.43","4":"0.32","_rn_":"1249"},{"1":"0.69","2":"0.09","3":"0.05","4":"0.17","_rn_":"1256"},{"1":"0.09","2":"0.12","3":"0.55","4":"0.24","_rn_":"1258"},{"1":"0.05","2":"0.01","3":"0.02","4":"0.92","_rn_":"1259"},{"1":"0.46","2":"0.14","3":"0.12","4":"0.28","_rn_":"1269"},{"1":"0.38","2":"0.10","3":"0.10","4":"0.42","_rn_":"1273"},{"1":"0.17","2":"0.50","3":"0.31","4":"0.02","_rn_":"1278"},{"1":"0.49","2":"0.07","3":"0.15","4":"0.29","_rn_":"1282"},{"1":"0.47","2":"0.15","3":"0.27","4":"0.11","_rn_":"1289"},{"1":"0.16","2":"0.19","3":"0.38","4":"0.27","_rn_":"1295"},{"1":"0.24","2":"0.27","3":"0.37","4":"0.12","_rn_":"1296"},{"1":"0.18","2":"0.36","3":"0.23","4":"0.23","_rn_":"1298"},{"1":"0.44","2":"0.09","3":"0.34","4":"0.13","_rn_":"1302"},{"1":"0.43","2":"0.24","3":"0.21","4":"0.12","_rn_":"1306"},{"1":"0.06","2":"0.15","3":"0.65","4":"0.14","_rn_":"1310"},{"1":"0.29","2":"0.19","3":"0.33","4":"0.19","_rn_":"1311"},{"1":"0.65","2":"0.26","3":"0.04","4":"0.05","_rn_":"1312"},{"1":"0.20","2":"0.40","3":"0.32","4":"0.08","_rn_":"1338"},{"1":"0.04","2":"0.81","3":"0.12","4":"0.03","_rn_":"1342"},{"1":"0.19","2":"0.29","3":"0.46","4":"0.06","_rn_":"1346"},{"1":"0.14","2":"0.12","3":"0.10","4":"0.64","_rn_":"1363"},{"1":"0.25","2":"0.05","3":"0.30","4":"0.40","_rn_":"1367"},{"1":"0.10","2":"0.22","3":"0.10","4":"0.58","_rn_":"1373"},{"1":"0.13","2":"0.32","3":"0.18","4":"0.37","_rn_":"1376"},{"1":"0.12","2":"0.26","3":"0.07","4":"0.55","_rn_":"1382"},{"1":"0.56","2":"0.07","3":"0.09","4":"0.28","_rn_":"1389"},{"1":"0.43","2":"0.12","3":"0.03","4":"0.42","_rn_":"1403"},{"1":"0.13","2":"0.13","3":"0.03","4":"0.71","_rn_":"1408"},{"1":"0.04","2":"0.65","3":"0.03","4":"0.28","_rn_":"1418"},{"1":"0.04","2":"0.57","3":"0.04","4":"0.35","_rn_":"1421"},{"1":"0.20","2":"0.29","3":"0.48","4":"0.03","_rn_":"1433"},{"1":"0.23","2":"0.39","3":"0.26","4":"0.12","_rn_":"1438"},{"1":"0.11","2":"0.42","3":"0.36","4":"0.11","_rn_":"1443"},{"1":"0.20","2":"0.22","3":"0.41","4":"0.17","_rn_":"1444"},{"1":"0.39","2":"0.14","3":"0.41","4":"0.06","_rn_":"1456"},{"1":"0.19","2":"0.04","3":"0.58","4":"0.19","_rn_":"1457"},{"1":"0.17","2":"0.01","3":"0.48","4":"0.34","_rn_":"1466"},{"1":"0.29","2":"0.03","3":"0.35","4":"0.33","_rn_":"1469"},{"1":"0.10","2":"0.07","3":"0.58","4":"0.25","_rn_":"1470"},{"1":"0.32","2":"0.04","3":"0.20","4":"0.44","_rn_":"1475"},{"1":"0.19","2":"0.61","3":"0.14","4":"0.06","_rn_":"1486"},{"1":"0.61","2":"0.14","3":"0.18","4":"0.07","_rn_":"1489"},{"1":"0.10","2":"0.56","3":"0.27","4":"0.07","_rn_":"1495"},{"1":"0.32","2":"0.09","3":"0.39","4":"0.20","_rn_":"1505"},{"1":"0.37","2":"0.20","3":"0.16","4":"0.27","_rn_":"1512"},{"1":"0.44","2":"0.48","3":"0.04","4":"0.04","_rn_":"1516"},{"1":"0.16","2":"0.07","3":"0.33","4":"0.44","_rn_":"1517"},{"1":"0.24","2":"0.12","3":"0.49","4":"0.15","_rn_":"1518"},{"1":"0.12","2":"0.26","3":"0.35","4":"0.27","_rn_":"1520"},{"1":"0.12","2":"0.16","3":"0.52","4":"0.20","_rn_":"1526"},{"1":"0.28","2":"0.36","3":"0.27","4":"0.09","_rn_":"1534"},{"1":"0.00","2":"0.05","3":"0.93","4":"0.02","_rn_":"1535"},{"1":"0.24","2":"0.13","3":"0.30","4":"0.33","_rn_":"1540"},{"1":"0.03","2":"0.03","3":"0.52","4":"0.42","_rn_":"1543"},{"1":"0.27","2":"0.09","3":"0.16","4":"0.48","_rn_":"1547"},{"1":"0.03","2":"0.19","3":"0.16","4":"0.62","_rn_":"1548"},{"1":"0.01","2":"0.56","3":"0.05","4":"0.38","_rn_":"1552"},{"1":"0.14","2":"0.12","3":"0.33","4":"0.41","_rn_":"1554"},{"1":"0.03","2":"0.72","3":"0.01","4":"0.24","_rn_":"1559"},{"1":"0.04","2":"0.27","3":"0.56","4":"0.13","_rn_":"1573"},{"1":"0.44","2":"0.14","3":"0.35","4":"0.07","_rn_":"1574"},{"1":"0.16","2":"0.45","3":"0.27","4":"0.12","_rn_":"1575"},{"1":"0.06","2":"0.11","3":"0.72","4":"0.11","_rn_":"1578"},{"1":"0.70","2":"0.10","3":"0.09","4":"0.11","_rn_":"1581"},{"1":"0.62","2":"0.07","3":"0.22","4":"0.09","_rn_":"1593"},{"1":"0.33","2":"0.48","3":"0.07","4":"0.12","_rn_":"1595"},{"1":"0.11","2":"0.35","3":"0.45","4":"0.09","_rn_":"1598"},{"1":"0.25","2":"0.24","3":"0.33","4":"0.18","_rn_":"1602"},{"1":"0.14","2":"0.25","3":"0.50","4":"0.11","_rn_":"1606"},{"1":"0.21","2":"0.23","3":"0.50","4":"0.06","_rn_":"1614"},{"1":"0.02","2":"0.11","3":"0.72","4":"0.15","_rn_":"1618"},{"1":"0.31","2":"0.38","3":"0.23","4":"0.08","_rn_":"1635"},{"1":"0.34","2":"0.43","3":"0.07","4":"0.16","_rn_":"1636"},{"1":"0.33","2":"0.32","3":"0.26","4":"0.09","_rn_":"1639"},{"1":"0.31","2":"0.25","3":"0.30","4":"0.14","_rn_":"1642"},{"1":"0.27","2":"0.43","3":"0.16","4":"0.14","_rn_":"1644"},{"1":"0.38","2":"0.18","3":"0.16","4":"0.28","_rn_":"1650"},{"1":"0.36","2":"0.30","3":"0.03","4":"0.31","_rn_":"1654"},{"1":"0.08","2":"0.08","3":"0.34","4":"0.50","_rn_":"1662"},{"1":"0.31","2":"0.11","3":"0.08","4":"0.50","_rn_":"1664"},{"1":"0.04","2":"0.13","3":"0.31","4":"0.52","_rn_":"1665"},{"1":"0.19","2":"0.37","3":"0.06","4":"0.38","_rn_":"1669"},{"1":"0.34","2":"0.06","3":"0.12","4":"0.48","_rn_":"1671"},{"1":"0.40","2":"0.26","3":"0.16","4":"0.18","_rn_":"1675"},{"1":"0.50","2":"0.32","3":"0.04","4":"0.14","_rn_":"1677"},{"1":"0.09","2":"0.36","3":"0.44","4":"0.11","_rn_":"1683"},{"1":"0.65","2":"0.16","3":"0.08","4":"0.11","_rn_":"1688"},{"1":"0.09","2":"0.06","3":"0.29","4":"0.56","_rn_":"1694"},{"1":"0.01","2":"0.05","3":"0.02","4":"0.92","_rn_":"1701"},{"1":"0.06","2":"0.24","3":"0.29","4":"0.41","_rn_":"1727"},{"1":"0.46","2":"0.25","3":"0.11","4":"0.18","_rn_":"1729"},{"1":"0.18","2":"0.69","3":"0.08","4":"0.05","_rn_":"1731"},{"1":"0.11","2":"0.36","3":"0.44","4":"0.09","_rn_":"1736"},{"1":"0.14","2":"0.69","3":"0.12","4":"0.05","_rn_":"1737"},{"1":"0.29","2":"0.32","3":"0.31","4":"0.08","_rn_":"1738"},{"1":"0.16","2":"0.45","3":"0.29","4":"0.10","_rn_":"1741"},{"1":"0.39","2":"0.34","3":"0.19","4":"0.08","_rn_":"1743"},{"1":"0.59","2":"0.26","3":"0.08","4":"0.07","_rn_":"1748"},{"1":"0.51","2":"0.03","3":"0.03","4":"0.43","_rn_":"1750"},{"1":"0.03","2":"0.08","3":"0.15","4":"0.74","_rn_":"1751"},{"1":"0.04","2":"0.33","3":"0.53","4":"0.10","_rn_":"1753"},{"1":"0.06","2":"0.23","3":"0.14","4":"0.57","_rn_":"1754"},{"1":"0.55","2":"0.14","3":"0.08","4":"0.23","_rn_":"1756"},{"1":"0.08","2":"0.52","3":"0.34","4":"0.06","_rn_":"1757"},{"1":"0.16","2":"0.36","3":"0.25","4":"0.23","_rn_":"1786"},{"1":"0.16","2":"0.04","3":"0.38","4":"0.42","_rn_":"1791"},{"1":"0.20","2":"0.15","3":"0.23","4":"0.42","_rn_":"1797"},{"1":"0.32","2":"0.06","3":"0.31","4":"0.31","_rn_":"1805"},{"1":"0.59","2":"0.21","3":"0.09","4":"0.11","_rn_":"1806"},{"1":"0.21","2":"0.54","3":"0.13","4":"0.12","_rn_":"1807"},{"1":"0.31","2":"0.50","3":"0.08","4":"0.11","_rn_":"1808"},{"1":"0.55","2":"0.13","3":"0.11","4":"0.21","_rn_":"1811"},{"1":"0.16","2":"0.16","3":"0.56","4":"0.12","_rn_":"1812"},{"1":"0.38","2":"0.25","3":"0.24","4":"0.13","_rn_":"1816"},{"1":"0.18","2":"0.57","3":"0.07","4":"0.18","_rn_":"1819"},{"1":"0.08","2":"0.67","3":"0.11","4":"0.14","_rn_":"1822"},{"1":"0.25","2":"0.14","3":"0.48","4":"0.13","_rn_":"1830"},{"1":"0.56","2":"0.31","3":"0.03","4":"0.10","_rn_":"1832"},{"1":"0.81","2":"0.11","3":"0.05","4":"0.03","_rn_":"1836"},{"1":"0.16","2":"0.25","3":"0.37","4":"0.22","_rn_":"1840"},{"1":"0.07","2":"0.14","3":"0.28","4":"0.51","_rn_":"1844"},{"1":"0.17","2":"0.11","3":"0.14","4":"0.58","_rn_":"1848"},{"1":"0.08","2":"0.11","3":"0.19","4":"0.62","_rn_":"1851"},{"1":"0.19","2":"0.21","3":"0.07","4":"0.53","_rn_":"1857"},{"1":"0.43","2":"0.16","3":"0.09","4":"0.32","_rn_":"1860"},{"1":"0.19","2":"0.26","3":"0.11","4":"0.44","_rn_":"1862"},{"1":"0.21","2":"0.12","3":"0.17","4":"0.50","_rn_":"1863"},{"1":"0.06","2":"0.02","3":"0.62","4":"0.30","_rn_":"1873"},{"1":"0.09","2":"0.14","3":"0.24","4":"0.53","_rn_":"1878"},{"1":"0.06","2":"0.23","3":"0.44","4":"0.27","_rn_":"1888"},{"1":"0.06","2":"0.07","3":"0.41","4":"0.46","_rn_":"1895"},{"1":"0.07","2":"0.10","3":"0.69","4":"0.14","_rn_":"1898"},{"1":"0.04","2":"0.10","3":"0.41","4":"0.45","_rn_":"1902"},{"1":"0.04","2":"0.12","3":"0.10","4":"0.74","_rn_":"1912"},{"1":"0.57","2":"0.25","3":"0.05","4":"0.13","_rn_":"1914"},{"1":"0.34","2":"0.10","3":"0.10","4":"0.46","_rn_":"1920"},{"1":"0.12","2":"0.56","3":"0.05","4":"0.27","_rn_":"1931"},{"1":"0.29","2":"0.13","3":"0.22","4":"0.36","_rn_":"1937"},{"1":"0.17","2":"0.09","3":"0.59","4":"0.15","_rn_":"1940"},{"1":"0.46","2":"0.39","3":"0.15","4":"0.00","_rn_":"1948"},{"1":"0.35","2":"0.37","3":"0.17","4":"0.11","_rn_":"1965"},{"1":"0.01","2":"0.23","3":"0.29","4":"0.47","_rn_":"1983"},{"1":"0.08","2":"0.27","3":"0.19","4":"0.46","_rn_":"1989"},{"1":"0.10","2":"0.12","3":"0.25","4":"0.53","_rn_":"1997"},{"1":"0.02","2":"0.35","3":"0.06","4":"0.57","_rn_":"2006"},{"1":"0.07","2":"0.21","3":"0.25","4":"0.47","_rn_":"2008"},{"1":"0.49","2":"0.17","3":"0.14","4":"0.20","_rn_":"2014"},{"1":"0.15","2":"0.48","3":"0.25","4":"0.12","_rn_":"2017"},{"1":"0.37","2":"0.36","3":"0.16","4":"0.11","_rn_":"2023"},{"1":"0.30","2":"0.24","3":"0.14","4":"0.32","_rn_":"2034"},{"1":"0.28","2":"0.21","3":"0.25","4":"0.26","_rn_":"2036"},{"1":"0.77","2":"0.05","3":"0.07","4":"0.11","_rn_":"2037"},{"1":"0.19","2":"0.36","3":"0.17","4":"0.28","_rn_":"2039"},{"1":"0.44","2":"0.13","3":"0.30","4":"0.13","_rn_":"2044"},{"1":"0.27","2":"0.38","3":"0.13","4":"0.22","_rn_":"2047"},{"1":"0.45","2":"0.10","3":"0.17","4":"0.28","_rn_":"2052"},{"1":"0.38","2":"0.36","3":"0.14","4":"0.12","_rn_":"2060"},{"1":"0.22","2":"0.63","3":"0.07","4":"0.08","_rn_":"2068"},{"1":"0.21","2":"0.45","3":"0.15","4":"0.19","_rn_":"2073"},{"1":"0.14","2":"0.57","3":"0.17","4":"0.12","_rn_":"2086"},{"1":"0.34","2":"0.41","3":"0.16","4":"0.09","_rn_":"2088"},{"1":"0.00","2":"0.09","3":"0.12","4":"0.79","_rn_":"2093"},{"1":"0.04","2":"0.28","3":"0.04","4":"0.64","_rn_":"2104"},{"1":"0.29","2":"0.12","3":"0.45","4":"0.14","_rn_":"2110"},{"1":"0.41","2":"0.09","3":"0.47","4":"0.03","_rn_":"2114"},{"1":"0.20","2":"0.36","3":"0.35","4":"0.09","_rn_":"2115"},{"1":"0.23","2":"0.31","3":"0.36","4":"0.10","_rn_":"2117"},{"1":"0.26","2":"0.16","3":"0.43","4":"0.15","_rn_":"2120"},{"1":"0.35","2":"0.26","3":"0.22","4":"0.17","_rn_":"2124"},{"1":"0.23","2":"0.19","3":"0.41","4":"0.17","_rn_":"2125"},{"1":"0.07","2":"0.27","3":"0.07","4":"0.59","_rn_":"2138"},{"1":"0.11","2":"0.05","3":"0.73","4":"0.11","_rn_":"2145"},{"1":"0.71","2":"0.04","3":"0.07","4":"0.18","_rn_":"2148"},{"1":"0.06","2":"0.13","3":"0.76","4":"0.05","_rn_":"2155"},{"1":"0.34","2":"0.06","3":"0.48","4":"0.12","_rn_":"2160"},{"1":"0.07","2":"0.72","3":"0.09","4":"0.12","_rn_":"2167"},{"1":"0.23","2":"0.47","3":"0.23","4":"0.07","_rn_":"2177"},{"1":"0.09","2":"0.68","3":"0.12","4":"0.11","_rn_":"2180"},{"1":"0.11","2":"0.65","3":"0.14","4":"0.10","_rn_":"2181"},{"1":"0.59","2":"0.19","3":"0.17","4":"0.05","_rn_":"2189"},{"1":"0.46","2":"0.34","3":"0.03","4":"0.17","_rn_":"2206"},{"1":"0.10","2":"0.43","3":"0.06","4":"0.41","_rn_":"2211"},{"1":"0.20","2":"0.06","3":"0.14","4":"0.60","_rn_":"2214"},{"1":"0.51","2":"0.11","3":"0.03","4":"0.35","_rn_":"2215"},{"1":"0.08","2":"0.32","3":"0.11","4":"0.49","_rn_":"2221"},{"1":"0.14","2":"0.23","3":"0.08","4":"0.55","_rn_":"2222"},{"1":"0.10","2":"0.27","3":"0.49","4":"0.14","_rn_":"2227"},{"1":"0.12","2":"0.40","3":"0.34","4":"0.14","_rn_":"2228"},{"1":"0.17","2":"0.32","3":"0.37","4":"0.14","_rn_":"2240"},{"1":"0.38","2":"0.41","3":"0.09","4":"0.12","_rn_":"2242"},{"1":"0.32","2":"0.23","3":"0.31","4":"0.14","_rn_":"2243"},{"1":"0.45","2":"0.28","3":"0.11","4":"0.16","_rn_":"2254"},{"1":"0.38","2":"0.45","3":"0.00","4":"0.17","_rn_":"2255"},{"1":"0.41","2":"0.26","3":"0.09","4":"0.24","_rn_":"2259"},{"1":"0.07","2":"0.22","3":"0.44","4":"0.27","_rn_":"2266"},{"1":"0.13","2":"0.39","3":"0.39","4":"0.09","_rn_":"2269"},{"1":"0.61","2":"0.19","3":"0.12","4":"0.08","_rn_":"2273"},{"1":"0.02","2":"0.44","3":"0.10","4":"0.44","_rn_":"2277"},{"1":"0.17","2":"0.20","3":"0.15","4":"0.48","_rn_":"2281"},{"1":"0.10","2":"0.32","3":"0.12","4":"0.46","_rn_":"2300"},{"1":"0.10","2":"0.14","3":"0.45","4":"0.31","_rn_":"2311"},{"1":"0.06","2":"0.12","3":"0.30","4":"0.52","_rn_":"2318"},{"1":"0.34","2":"0.54","3":"0.08","4":"0.04","_rn_":"2335"},{"1":"0.10","2":"0.16","3":"0.48","4":"0.26","_rn_":"2338"},{"1":"0.18","2":"0.17","3":"0.43","4":"0.22","_rn_":"2341"},{"1":"0.12","2":"0.47","3":"0.27","4":"0.14","_rn_":"2344"},{"1":"0.09","2":"0.61","3":"0.27","4":"0.03","_rn_":"2348"},{"1":"0.45","2":"0.42","3":"0.03","4":"0.10","_rn_":"2357"},{"1":"0.20","2":"0.36","3":"0.25","4":"0.19","_rn_":"2363"},{"1":"0.20","2":"0.30","3":"0.04","4":"0.46","_rn_":"2377"},{"1":"0.09","2":"0.09","3":"0.09","4":"0.73","_rn_":"2390"},{"1":"0.07","2":"0.45","3":"0.12","4":"0.36","_rn_":"2391"},{"1":"0.03","2":"0.08","3":"0.04","4":"0.85","_rn_":"2395"},{"1":"0.62","2":"0.09","3":"0.01","4":"0.28","_rn_":"2401"},{"1":"0.02","2":"0.16","3":"0.01","4":"0.81","_rn_":"2406"},{"1":"0.41","2":"0.28","3":"0.00","4":"0.31","_rn_":"2410"},{"1":"0.07","2":"0.01","3":"0.01","4":"0.91","_rn_":"2411"},{"1":"0.25","2":"0.41","3":"0.17","4":"0.17","_rn_":"2418"},{"1":"0.42","2":"0.24","3":"0.25","4":"0.09","_rn_":"2430"},{"1":"0.29","2":"0.49","3":"0.05","4":"0.17","_rn_":"2439"},{"1":"0.29","2":"0.24","3":"0.06","4":"0.41","_rn_":"2443"},{"1":"0.09","2":"0.08","3":"0.03","4":"0.80","_rn_":"2445"},{"1":"0.07","2":"0.08","3":"0.01","4":"0.84","_rn_":"2460"},{"1":"0.06","2":"0.28","3":"0.02","4":"0.64","_rn_":"2461"},{"1":"0.06","2":"0.56","3":"0.27","4":"0.11","_rn_":"2462"},{"1":"0.27","2":"0.06","3":"0.21","4":"0.46","_rn_":"2472"},{"1":"0.03","2":"0.12","3":"0.24","4":"0.61","_rn_":"2475"},{"1":"0.46","2":"0.16","3":"0.14","4":"0.24","_rn_":"2481"},{"1":"0.34","2":"0.28","3":"0.14","4":"0.24","_rn_":"2494"},{"1":"0.11","2":"0.46","3":"0.27","4":"0.16","_rn_":"2531"},{"1":"0.13","2":"0.29","3":"0.20","4":"0.38","_rn_":"2541"},{"1":"0.04","2":"0.08","3":"0.59","4":"0.29","_rn_":"2543"},{"1":"0.32","2":"0.42","3":"0.16","4":"0.10","_rn_":"2555"},{"1":"0.17","2":"0.42","3":"0.32","4":"0.09","_rn_":"2561"},{"1":"0.52","2":"0.13","3":"0.24","4":"0.11","_rn_":"2564"},{"1":"0.15","2":"0.42","3":"0.14","4":"0.29","_rn_":"2572"},{"1":"0.11","2":"0.71","3":"0.05","4":"0.13","_rn_":"2574"},{"1":"0.15","2":"0.41","3":"0.38","4":"0.06","_rn_":"2586"},{"1":"0.13","2":"0.23","3":"0.15","4":"0.49","_rn_":"2587"},{"1":"0.11","2":"0.55","3":"0.11","4":"0.23","_rn_":"2592"},{"1":"0.71","2":"0.06","3":"0.11","4":"0.12","_rn_":"2593"},{"1":"0.66","2":"0.11","3":"0.04","4":"0.19","_rn_":"2594"},{"1":"0.03","2":"0.04","3":"0.76","4":"0.17","_rn_":"2613"},{"1":"0.20","2":"0.57","3":"0.12","4":"0.11","_rn_":"2627"},{"1":"0.00","2":"0.15","3":"0.82","4":"0.03","_rn_":"2628"},{"1":"0.16","2":"0.50","3":"0.15","4":"0.19","_rn_":"2630"},{"1":"0.09","2":"0.17","3":"0.70","4":"0.04","_rn_":"2634"},{"1":"0.07","2":"0.06","3":"0.00","4":"0.87","_rn_":"2647"},{"1":"0.12","2":"0.38","3":"0.43","4":"0.07","_rn_":"2667"},{"1":"0.09","2":"0.44","3":"0.27","4":"0.20","_rn_":"2669"},{"1":"0.11","2":"0.46","3":"0.34","4":"0.09","_rn_":"2671"},{"1":"0.09","2":"0.26","3":"0.60","4":"0.05","_rn_":"2675"},{"1":"0.11","2":"0.21","3":"0.35","4":"0.33","_rn_":"2680"},{"1":"0.16","2":"0.05","3":"0.72","4":"0.07","_rn_":"2681"},{"1":"0.07","2":"0.10","3":"0.60","4":"0.23","_rn_":"2682"},{"1":"0.11","2":"0.08","3":"0.63","4":"0.18","_rn_":"2685"},{"1":"0.07","2":"0.10","3":"0.61","4":"0.22","_rn_":"2687"},{"1":"0.06","2":"0.16","3":"0.13","4":"0.65","_rn_":"2689"},{"1":"0.16","2":"0.42","3":"0.20","4":"0.22","_rn_":"2690"},{"1":"0.15","2":"0.47","3":"0.02","4":"0.36","_rn_":"2697"},{"1":"0.32","2":"0.24","3":"0.16","4":"0.28","_rn_":"2702"},{"1":"0.03","2":"0.33","3":"0.54","4":"0.10","_rn_":"2704"},{"1":"0.03","2":"0.48","3":"0.41","4":"0.08","_rn_":"2714"},{"1":"0.23","2":"0.23","3":"0.20","4":"0.34","_rn_":"2715"},{"1":"0.20","2":"0.10","3":"0.62","4":"0.08","_rn_":"2716"},{"1":"0.15","2":"0.33","3":"0.45","4":"0.07","_rn_":"2717"},{"1":"0.01","2":"0.33","3":"0.03","4":"0.63","_rn_":"2724"},{"1":"0.17","2":"0.16","3":"0.03","4":"0.64","_rn_":"2728"},{"1":"0.00","2":"0.26","3":"0.07","4":"0.67","_rn_":"2735"},{"1":"0.03","2":"0.35","3":"0.50","4":"0.12","_rn_":"2741"},{"1":"0.06","2":"0.50","3":"0.34","4":"0.10","_rn_":"2748"},{"1":"0.04","2":"0.44","3":"0.42","4":"0.10","_rn_":"2750"},{"1":"0.32","2":"0.13","3":"0.46","4":"0.09","_rn_":"2754"},{"1":"0.64","2":"0.09","3":"0.09","4":"0.18","_rn_":"2761"},{"1":"0.60","2":"0.09","3":"0.13","4":"0.18","_rn_":"2762"},{"1":"0.03","2":"0.03","3":"0.14","4":"0.80","_rn_":"2776"},{"1":"0.08","2":"0.02","3":"0.01","4":"0.89","_rn_":"2777"},{"1":"0.39","2":"0.32","3":"0.05","4":"0.24","_rn_":"2780"},{"1":"0.09","2":"0.20","3":"0.16","4":"0.55","_rn_":"2783"},{"1":"0.20","2":"0.45","3":"0.30","4":"0.05","_rn_":"2796"},{"1":"0.20","2":"0.72","3":"0.02","4":"0.06","_rn_":"2805"},{"1":"0.77","2":"0.13","3":"0.03","4":"0.07","_rn_":"2809"},{"1":"0.24","2":"0.17","3":"0.44","4":"0.15","_rn_":"2816"},{"1":"0.75","2":"0.03","3":"0.17","4":"0.05","_rn_":"2817"},{"1":"0.28","2":"0.30","3":"0.18","4":"0.24","_rn_":"2826"},{"1":"0.03","2":"0.31","3":"0.48","4":"0.18","_rn_":"2842"},{"1":"0.12","2":"0.65","3":"0.16","4":"0.07","_rn_":"2847"},{"1":"0.73","2":"0.03","3":"0.18","4":"0.06","_rn_":"2851"},{"1":"0.25","2":"0.14","3":"0.53","4":"0.08","_rn_":"2855"},{"1":"0.55","2":"0.06","3":"0.34","4":"0.05","_rn_":"2862"},{"1":"0.26","2":"0.42","3":"0.26","4":"0.06","_rn_":"2868"},{"1":"0.54","2":"0.17","3":"0.25","4":"0.04","_rn_":"2869"},{"1":"0.09","2":"0.08","3":"0.62","4":"0.21","_rn_":"2875"},{"1":"0.54","2":"0.09","3":"0.20","4":"0.17","_rn_":"2879"},{"1":"0.12","2":"0.14","3":"0.17","4":"0.57","_rn_":"2882"},{"1":"0.52","2":"0.11","3":"0.24","4":"0.13","_rn_":"2883"},{"1":"0.65","2":"0.09","3":"0.11","4":"0.15","_rn_":"2889"},{"1":"0.20","2":"0.11","3":"0.20","4":"0.49","_rn_":"2894"},{"1":"0.68","2":"0.07","3":"0.10","4":"0.15","_rn_":"2899"},{"1":"0.06","2":"0.39","3":"0.28","4":"0.27","_rn_":"2913"},{"1":"0.19","2":"0.10","3":"0.33","4":"0.38","_rn_":"2920"},{"1":"0.05","2":"0.11","3":"0.45","4":"0.39","_rn_":"2922"},{"1":"0.10","2":"0.24","3":"0.28","4":"0.38","_rn_":"2932"},{"1":"0.13","2":"0.02","3":"0.06","4":"0.79","_rn_":"2939"},{"1":"0.16","2":"0.17","3":"0.07","4":"0.60","_rn_":"2940"},{"1":"0.05","2":"0.52","3":"0.09","4":"0.34","_rn_":"2943"},{"1":"0.13","2":"0.20","3":"0.06","4":"0.61","_rn_":"2953"},{"1":"0.28","2":"0.08","3":"0.03","4":"0.61","_rn_":"2955"},{"1":"0.04","2":"0.25","3":"0.10","4":"0.61","_rn_":"2957"},{"1":"0.12","2":"0.21","3":"0.05","4":"0.62","_rn_":"2960"},{"1":"0.69","2":"0.14","3":"0.06","4":"0.11","_rn_":"2969"},{"1":"0.51","2":"0.34","3":"0.09","4":"0.06","_rn_":"2979"},{"1":"0.30","2":"0.45","3":"0.12","4":"0.13","_rn_":"2982"},{"1":"0.67","2":"0.24","3":"0.08","4":"0.01","_rn_":"2983"},{"1":"0.03","2":"0.08","3":"0.01","4":"0.88","_rn_":"2988"},{"1":"0.18","2":"0.23","3":"0.45","4":"0.14","_rn_":"3004"},{"1":"0.67","2":"0.15","3":"0.09","4":"0.09","_rn_":"3006"},{"1":"0.23","2":"0.25","3":"0.12","4":"0.40","_rn_":"3015"},{"1":"0.27","2":"0.06","3":"0.50","4":"0.17","_rn_":"3019"},{"1":"0.31","2":"0.13","3":"0.09","4":"0.47","_rn_":"3022"},{"1":"0.29","2":"0.33","3":"0.08","4":"0.30","_rn_":"3023"},{"1":"0.12","2":"0.43","3":"0.36","4":"0.09","_rn_":"3027"},{"1":"0.08","2":"0.04","3":"0.05","4":"0.83","_rn_":"3028"},{"1":"0.18","2":"0.11","3":"0.36","4":"0.35","_rn_":"3032"},{"1":"0.13","2":"0.08","3":"0.35","4":"0.44","_rn_":"3035"},{"1":"0.13","2":"0.48","3":"0.24","4":"0.15","_rn_":"3050"},{"1":"0.33","2":"0.18","3":"0.21","4":"0.28","_rn_":"3051"},{"1":"0.38","2":"0.22","3":"0.11","4":"0.29","_rn_":"3054"},{"1":"0.06","2":"0.02","3":"0.87","4":"0.05","_rn_":"3056"},{"1":"0.22","2":"0.15","3":"0.07","4":"0.56","_rn_":"3065"},{"1":"0.43","2":"0.10","3":"0.07","4":"0.40","_rn_":"3068"},{"1":"0.08","2":"0.33","3":"0.22","4":"0.37","_rn_":"3076"},{"1":"0.48","2":"0.09","3":"0.08","4":"0.35","_rn_":"3077"},{"1":"0.28","2":"0.05","3":"0.05","4":"0.62","_rn_":"3082"},{"1":"0.32","2":"0.05","3":"0.00","4":"0.63","_rn_":"3086"},{"1":"0.61","2":"0.11","3":"0.08","4":"0.20","_rn_":"3099"},{"1":"0.04","2":"0.15","3":"0.29","4":"0.52","_rn_":"3104"},{"1":"0.04","2":"0.08","3":"0.05","4":"0.83","_rn_":"3105"},{"1":"0.61","2":"0.08","3":"0.09","4":"0.22","_rn_":"3106"},{"1":"0.58","2":"0.09","3":"0.13","4":"0.20","_rn_":"3107"},{"1":"0.05","2":"0.22","3":"0.08","4":"0.65","_rn_":"3112"},{"1":"0.52","2":"0.12","3":"0.17","4":"0.19","_rn_":"3115"},{"1":"0.08","2":"0.06","3":"0.81","4":"0.05","_rn_":"3124"},{"1":"0.31","2":"0.43","3":"0.14","4":"0.12","_rn_":"3127"},{"1":"0.57","2":"0.21","3":"0.10","4":"0.12","_rn_":"3130"},{"1":"0.13","2":"0.12","3":"0.59","4":"0.16","_rn_":"3134"},{"1":"0.11","2":"0.21","3":"0.14","4":"0.54","_rn_":"3139"},{"1":"0.09","2":"0.08","3":"0.69","4":"0.14","_rn_":"3146"},{"1":"0.49","2":"0.06","3":"0.11","4":"0.34","_rn_":"3152"},{"1":"0.20","2":"0.59","3":"0.11","4":"0.10","_rn_":"3153"},{"1":"0.11","2":"0.66","3":"0.11","4":"0.12","_rn_":"3155"},{"1":"0.35","2":"0.24","3":"0.10","4":"0.31","_rn_":"3156"},{"1":"0.48","2":"0.09","3":"0.13","4":"0.30","_rn_":"3159"},{"1":"0.33","2":"0.21","3":"0.11","4":"0.35","_rn_":"3160"},{"1":"0.88","2":"0.00","3":"0.02","4":"0.10","_rn_":"3163"},{"1":"0.04","2":"0.72","3":"0.14","4":"0.10","_rn_":"3166"},{"1":"0.35","2":"0.15","3":"0.06","4":"0.44","_rn_":"3170"},{"1":"0.08","2":"0.26","3":"0.25","4":"0.41","_rn_":"3175"},{"1":"0.35","2":"0.22","3":"0.13","4":"0.30","_rn_":"3177"},{"1":"0.23","2":"0.47","3":"0.10","4":"0.20","_rn_":"3187"},{"1":"0.41","2":"0.04","3":"0.16","4":"0.39","_rn_":"3188"},{"1":"0.12","2":"0.62","3":"0.09","4":"0.17","_rn_":"3190"},{"1":"0.37","2":"0.26","3":"0.07","4":"0.30","_rn_":"3194"},{"1":"0.18","2":"0.24","3":"0.29","4":"0.29","_rn_":"3198"},{"1":"0.05","2":"0.12","3":"0.50","4":"0.33","_rn_":"3204"},{"1":"0.57","2":"0.13","3":"0.07","4":"0.23","_rn_":"3216"},{"1":"0.15","2":"0.35","3":"0.35","4":"0.15","_rn_":"3222"},{"1":"0.40","2":"0.36","3":"0.06","4":"0.18","_rn_":"3225"},{"1":"0.02","2":"0.22","3":"0.67","4":"0.09","_rn_":"3229"},{"1":"0.11","2":"0.74","3":"0.10","4":"0.05","_rn_":"3234"},{"1":"0.44","2":"0.11","3":"0.41","4":"0.04","_rn_":"3235"},{"1":"0.04","2":"0.48","3":"0.39","4":"0.09","_rn_":"3243"},{"1":"0.70","2":"0.10","3":"0.13","4":"0.07","_rn_":"3246"},{"1":"0.44","2":"0.28","3":"0.14","4":"0.14","_rn_":"3251"},{"1":"0.17","2":"0.52","3":"0.19","4":"0.12","_rn_":"3252"},{"1":"0.36","2":"0.30","3":"0.15","4":"0.19","_rn_":"3260"},{"1":"0.40","2":"0.40","3":"0.09","4":"0.11","_rn_":"3264"},{"1":"0.35","2":"0.17","3":"0.18","4":"0.30","_rn_":"3269"},{"1":"0.24","2":"0.35","3":"0.20","4":"0.21","_rn_":"3275"},{"1":"0.23","2":"0.13","3":"0.29","4":"0.35","_rn_":"3276"},{"1":"0.06","2":"0.02","3":"0.84","4":"0.08","_rn_":"3278"},{"1":"0.35","2":"0.52","3":"0.07","4":"0.06","_rn_":"3283"},{"1":"0.48","2":"0.19","3":"0.21","4":"0.12","_rn_":"3290"},{"1":"0.22","2":"0.29","3":"0.38","4":"0.11","_rn_":"3297"},{"1":"0.39","2":"0.45","3":"0.09","4":"0.07","_rn_":"3298"},{"1":"0.55","2":"0.13","3":"0.06","4":"0.26","_rn_":"3308"},{"1":"0.60","2":"0.04","3":"0.29","4":"0.07","_rn_":"3330"},{"1":"0.65","2":"0.11","3":"0.07","4":"0.17","_rn_":"3333"},{"1":"0.04","2":"0.65","3":"0.17","4":"0.14","_rn_":"3342"},{"1":"0.49","2":"0.11","3":"0.25","4":"0.15","_rn_":"3343"},{"1":"0.44","2":"0.20","3":"0.07","4":"0.29","_rn_":"3344"},{"1":"0.07","2":"0.08","3":"0.65","4":"0.20","_rn_":"3348"},{"1":"0.40","2":"0.32","3":"0.05","4":"0.23","_rn_":"3354"},{"1":"0.61","2":"0.30","3":"0.02","4":"0.07","_rn_":"3360"},{"1":"0.57","2":"0.07","3":"0.20","4":"0.16","_rn_":"3361"},{"1":"0.04","2":"0.50","3":"0.12","4":"0.34","_rn_":"3365"},{"1":"0.40","2":"0.09","3":"0.18","4":"0.33","_rn_":"3387"},{"1":"0.11","2":"0.08","3":"0.03","4":"0.78","_rn_":"3394"},{"1":"0.33","2":"0.03","3":"0.05","4":"0.59","_rn_":"3396"},{"1":"0.66","2":"0.01","3":"0.02","4":"0.31","_rn_":"3397"},{"1":"0.19","2":"0.11","3":"0.14","4":"0.56","_rn_":"3411"},{"1":"0.18","2":"0.24","3":"0.40","4":"0.18","_rn_":"3421"},{"1":"0.54","2":"0.11","3":"0.03","4":"0.32","_rn_":"3424"},{"1":"0.57","2":"0.05","3":"0.15","4":"0.23","_rn_":"3426"},{"1":"0.05","2":"0.19","3":"0.49","4":"0.27","_rn_":"3432"},{"1":"0.03","2":"0.12","3":"0.08","4":"0.77","_rn_":"3447"},{"1":"0.49","2":"0.06","3":"0.08","4":"0.37","_rn_":"3467"},{"1":"0.03","2":"0.36","3":"0.07","4":"0.54","_rn_":"3482"},{"1":"0.05","2":"0.15","3":"0.13","4":"0.67","_rn_":"3489"},{"1":"0.01","2":"0.08","3":"0.26","4":"0.65","_rn_":"3497"},{"1":"0.63","2":"0.09","3":"0.06","4":"0.22","_rn_":"3501"},{"1":"0.07","2":"0.68","3":"0.02","4":"0.23","_rn_":"3505"},{"1":"0.04","2":"0.06","3":"0.08","4":"0.82","_rn_":"3508"},{"1":"0.57","2":"0.14","3":"0.21","4":"0.08","_rn_":"3516"},{"1":"0.55","2":"0.10","3":"0.28","4":"0.07","_rn_":"3527"},{"1":"0.19","2":"0.55","3":"0.12","4":"0.14","_rn_":"3532"},{"1":"0.57","2":"0.07","3":"0.32","4":"0.04","_rn_":"3534"},{"1":"0.20","2":"0.35","3":"0.22","4":"0.23","_rn_":"3546"},{"1":"0.35","2":"0.17","3":"0.22","4":"0.26","_rn_":"3548"},{"1":"0.25","2":"0.05","3":"0.12","4":"0.58","_rn_":"3559"},{"1":"0.16","2":"0.07","3":"0.05","4":"0.72","_rn_":"3562"},{"1":"0.26","2":"0.09","3":"0.11","4":"0.54","_rn_":"3564"},{"1":"0.06","2":"0.11","3":"0.10","4":"0.73","_rn_":"3568"},{"1":"0.17","2":"0.13","3":"0.11","4":"0.59","_rn_":"3569"},{"1":"0.03","2":"0.08","3":"0.15","4":"0.74","_rn_":"3570"},{"1":"0.08","2":"0.45","3":"0.04","4":"0.43","_rn_":"3571"},{"1":"0.22","2":"0.09","3":"0.06","4":"0.63","_rn_":"3572"},{"1":"0.35","2":"0.15","3":"0.31","4":"0.19","_rn_":"3573"},{"1":"0.05","2":"0.08","3":"0.09","4":"0.78","_rn_":"3586"},{"1":"0.15","2":"0.32","3":"0.22","4":"0.31","_rn_":"3602"},{"1":"0.14","2":"0.26","3":"0.53","4":"0.07","_rn_":"3603"},{"1":"0.15","2":"0.32","3":"0.35","4":"0.18","_rn_":"3604"},{"1":"0.09","2":"0.39","3":"0.47","4":"0.05","_rn_":"3609"},{"1":"0.74","2":"0.07","3":"0.07","4":"0.12","_rn_":"3616"},{"1":"0.24","2":"0.06","3":"0.62","4":"0.08","_rn_":"3628"},{"1":"0.11","2":"0.08","3":"0.54","4":"0.27","_rn_":"3631"},{"1":"0.26","2":"0.21","3":"0.39","4":"0.14","_rn_":"3635"},{"1":"0.12","2":"0.02","3":"0.58","4":"0.28","_rn_":"3643"},{"1":"0.69","2":"0.04","3":"0.12","4":"0.15","_rn_":"3649"},{"1":"0.40","2":"0.12","3":"0.29","4":"0.19","_rn_":"3675"},{"1":"0.34","2":"0.30","3":"0.07","4":"0.29","_rn_":"3677"},{"1":"0.17","2":"0.52","3":"0.23","4":"0.08","_rn_":"3682"},{"1":"0.08","2":"0.37","3":"0.38","4":"0.17","_rn_":"3683"},{"1":"0.10","2":"0.60","3":"0.11","4":"0.19","_rn_":"3695"},{"1":"0.31","2":"0.43","3":"0.01","4":"0.25","_rn_":"3697"},{"1":"0.21","2":"0.49","3":"0.06","4":"0.24","_rn_":"3699"},{"1":"0.16","2":"0.40","3":"0.09","4":"0.35","_rn_":"3702"},{"1":"0.10","2":"0.12","3":"0.20","4":"0.58","_rn_":"3711"},{"1":"0.13","2":"0.44","3":"0.15","4":"0.28","_rn_":"3712"},{"1":"0.05","2":"0.33","3":"0.48","4":"0.14","_rn_":"3713"},{"1":"0.11","2":"0.08","3":"0.35","4":"0.46","_rn_":"3722"},{"1":"0.37","2":"0.22","3":"0.22","4":"0.19","_rn_":"3727"},{"1":"0.18","2":"0.37","3":"0.34","4":"0.11","_rn_":"3731"},{"1":"0.08","2":"0.50","3":"0.26","4":"0.16","_rn_":"3733"},{"1":"0.32","2":"0.35","3":"0.15","4":"0.18","_rn_":"3738"},{"1":"0.11","2":"0.74","3":"0.14","4":"0.01","_rn_":"3740"},{"1":"0.65","2":"0.13","3":"0.11","4":"0.11","_rn_":"3744"},{"1":"0.36","2":"0.19","3":"0.38","4":"0.07","_rn_":"3746"},{"1":"0.34","2":"0.15","3":"0.49","4":"0.02","_rn_":"3748"},{"1":"0.26","2":"0.40","3":"0.19","4":"0.15","_rn_":"3751"},{"1":"0.51","2":"0.11","3":"0.35","4":"0.03","_rn_":"3757"},{"1":"0.60","2":"0.05","3":"0.10","4":"0.25","_rn_":"3763"},{"1":"0.06","2":"0.49","3":"0.00","4":"0.45","_rn_":"3775"},{"1":"0.35","2":"0.03","3":"0.03","4":"0.59","_rn_":"3776"},{"1":"0.28","2":"0.04","3":"0.16","4":"0.52","_rn_":"3779"},{"1":"0.31","2":"0.19","3":"0.28","4":"0.22","_rn_":"3782"},{"1":"0.73","2":"0.09","3":"0.04","4":"0.14","_rn_":"3795"},{"1":"0.12","2":"0.33","3":"0.40","4":"0.15","_rn_":"3796"},{"1":"0.16","2":"0.49","3":"0.11","4":"0.24","_rn_":"3797"},{"1":"0.02","2":"0.21","3":"0.20","4":"0.57","_rn_":"3801"},{"1":"0.26","2":"0.07","3":"0.42","4":"0.25","_rn_":"3808"},{"1":"0.02","2":"0.62","3":"0.10","4":"0.26","_rn_":"3809"},{"1":"0.06","2":"0.09","3":"0.65","4":"0.20","_rn_":"3817"},{"1":"0.09","2":"0.09","3":"0.01","4":"0.81","_rn_":"3826"},{"1":"0.03","2":"0.60","3":"0.04","4":"0.33","_rn_":"3828"},{"1":"0.10","2":"0.41","3":"0.17","4":"0.32","_rn_":"3830"},{"1":"0.05","2":"0.17","3":"0.06","4":"0.72","_rn_":"3836"},{"1":"0.42","2":"0.16","3":"0.22","4":"0.20","_rn_":"3841"},{"1":"0.10","2":"0.36","3":"0.16","4":"0.38","_rn_":"3846"},{"1":"0.16","2":"0.44","3":"0.05","4":"0.35","_rn_":"3851"},{"1":"0.04","2":"0.12","3":"0.07","4":"0.77","_rn_":"3853"},{"1":"0.51","2":"0.22","3":"0.07","4":"0.20","_rn_":"3865"},{"1":"0.37","2":"0.08","3":"0.33","4":"0.22","_rn_":"3867"},{"1":"0.04","2":"0.02","3":"0.92","4":"0.02","_rn_":"3870"},{"1":"0.39","2":"0.50","3":"0.07","4":"0.04","_rn_":"3874"},{"1":"0.38","2":"0.16","3":"0.21","4":"0.25","_rn_":"3876"},{"1":"0.13","2":"0.05","3":"0.52","4":"0.30","_rn_":"3881"},{"1":"0.06","2":"0.32","3":"0.05","4":"0.57","_rn_":"3889"},{"1":"0.11","2":"0.09","3":"0.42","4":"0.38","_rn_":"3921"},{"1":"0.10","2":"0.45","3":"0.11","4":"0.34","_rn_":"3928"},{"1":"0.08","2":"0.11","3":"0.71","4":"0.10","_rn_":"3933"},{"1":"0.13","2":"0.23","3":"0.18","4":"0.46","_rn_":"3935"},{"1":"0.03","2":"0.17","3":"0.03","4":"0.77","_rn_":"3937"},{"1":"0.20","2":"0.13","3":"0.09","4":"0.58","_rn_":"3941"},{"1":"0.07","2":"0.16","3":"0.02","4":"0.75","_rn_":"3949"},{"1":"0.79","2":"0.09","3":"0.03","4":"0.09","_rn_":"3954"},{"1":"0.16","2":"0.04","3":"0.67","4":"0.13","_rn_":"3975"},{"1":"0.67","2":"0.08","3":"0.06","4":"0.19","_rn_":"3990"},{"1":"0.59","2":"0.17","3":"0.14","4":"0.10","_rn_":"3998"},{"1":"0.14","2":"0.04","3":"0.12","4":"0.70","_rn_":"4010"},{"1":"0.03","2":"0.07","3":"0.37","4":"0.53","_rn_":"4012"},{"1":"0.22","2":"0.12","3":"0.11","4":"0.55","_rn_":"4019"},{"1":"0.09","2":"0.03","3":"0.08","4":"0.80","_rn_":"4020"},{"1":"0.65","2":"0.16","3":"0.07","4":"0.12","_rn_":"4023"},{"1":"0.61","2":"0.23","3":"0.06","4":"0.10","_rn_":"4025"},{"1":"0.12","2":"0.05","3":"0.49","4":"0.34","_rn_":"4033"},{"1":"0.46","2":"0.12","3":"0.05","4":"0.37","_rn_":"4036"},{"1":"0.11","2":"0.37","3":"0.38","4":"0.14","_rn_":"4038"},{"1":"0.34","2":"0.14","3":"0.13","4":"0.39","_rn_":"4039"},{"1":"0.08","2":"0.17","3":"0.50","4":"0.25","_rn_":"4044"},{"1":"0.17","2":"0.38","3":"0.32","4":"0.13","_rn_":"4052"},{"1":"0.22","2":"0.49","3":"0.22","4":"0.07","_rn_":"4057"},{"1":"0.33","2":"0.19","3":"0.18","4":"0.30","_rn_":"4069"},{"1":"0.24","2":"0.30","3":"0.06","4":"0.40","_rn_":"4075"},{"1":"0.21","2":"0.29","3":"0.08","4":"0.42","_rn_":"4076"},{"1":"0.20","2":"0.35","3":"0.32","4":"0.13","_rn_":"4078"},{"1":"0.08","2":"0.13","3":"0.08","4":"0.71","_rn_":"4084"},{"1":"0.15","2":"0.32","3":"0.32","4":"0.21","_rn_":"4087"},{"1":"0.62","2":"0.17","3":"0.15","4":"0.06","_rn_":"4093"},{"1":"0.32","2":"0.50","3":"0.15","4":"0.03","_rn_":"4099"},{"1":"0.16","2":"0.50","3":"0.30","4":"0.04","_rn_":"4101"},{"1":"0.26","2":"0.44","3":"0.22","4":"0.08","_rn_":"4102"},{"1":"0.27","2":"0.27","3":"0.41","4":"0.05","_rn_":"4112"},{"1":"0.41","2":"0.22","3":"0.30","4":"0.07","_rn_":"4120"},{"1":"0.27","2":"0.42","3":"0.11","4":"0.20","_rn_":"4129"},{"1":"0.29","2":"0.41","3":"0.19","4":"0.11","_rn_":"4133"},{"1":"0.09","2":"0.47","3":"0.32","4":"0.12","_rn_":"4134"},{"1":"0.08","2":"0.58","3":"0.28","4":"0.06","_rn_":"4135"},{"1":"0.16","2":"0.06","3":"0.45","4":"0.33","_rn_":"4151"},{"1":"0.04","2":"0.11","3":"0.38","4":"0.47","_rn_":"4161"},{"1":"0.33","2":"0.31","3":"0.18","4":"0.18","_rn_":"4164"},{"1":"0.44","2":"0.26","3":"0.10","4":"0.20","_rn_":"4169"},{"1":"0.05","2":"0.81","3":"0.01","4":"0.13","_rn_":"4179"},{"1":"0.43","2":"0.38","3":"0.03","4":"0.16","_rn_":"4183"},{"1":"0.27","2":"0.05","3":"0.42","4":"0.26","_rn_":"4187"},{"1":"0.40","2":"0.11","3":"0.25","4":"0.24","_rn_":"4193"},{"1":"0.08","2":"0.09","3":"0.72","4":"0.11","_rn_":"4203"},{"1":"0.13","2":"0.30","3":"0.43","4":"0.14","_rn_":"4206"},{"1":"0.17","2":"0.70","3":"0.07","4":"0.06","_rn_":"4211"},{"1":"0.10","2":"0.39","3":"0.36","4":"0.15","_rn_":"4219"},{"1":"0.40","2":"0.10","3":"0.14","4":"0.36","_rn_":"4220"},{"1":"0.23","2":"0.11","3":"0.61","4":"0.05","_rn_":"4225"},{"1":"0.58","2":"0.11","3":"0.10","4":"0.21","_rn_":"4227"},{"1":"0.69","2":"0.06","3":"0.21","4":"0.04","_rn_":"4234"},{"1":"0.13","2":"0.32","3":"0.10","4":"0.45","_rn_":"4235"},{"1":"0.20","2":"0.45","3":"0.21","4":"0.14","_rn_":"4237"},{"1":"0.09","2":"0.07","3":"0.00","4":"0.84","_rn_":"4242"},{"1":"0.07","2":"0.06","3":"0.03","4":"0.84","_rn_":"4247"},{"1":"0.11","2":"0.04","3":"0.14","4":"0.71","_rn_":"4256"},{"1":"0.06","2":"0.02","3":"0.01","4":"0.91","_rn_":"4263"},{"1":"0.17","2":"0.02","3":"0.62","4":"0.19","_rn_":"4269"},{"1":"0.10","2":"0.05","3":"0.05","4":"0.80","_rn_":"4272"},{"1":"0.50","2":"0.23","3":"0.14","4":"0.13","_rn_":"4289"},{"1":"0.41","2":"0.25","3":"0.20","4":"0.14","_rn_":"4293"},{"1":"0.37","2":"0.24","3":"0.32","4":"0.07","_rn_":"4294"},{"1":"0.16","2":"0.07","3":"0.27","4":"0.50","_rn_":"4298"},{"1":"0.27","2":"0.05","3":"0.63","4":"0.05","_rn_":"4309"},{"1":"0.09","2":"0.21","3":"0.54","4":"0.16","_rn_":"4310"},{"1":"0.07","2":"0.14","3":"0.56","4":"0.23","_rn_":"4313"},{"1":"0.11","2":"0.66","3":"0.17","4":"0.06","_rn_":"4318"},{"1":"0.21","2":"0.63","3":"0.06","4":"0.10","_rn_":"4322"},{"1":"0.11","2":"0.20","3":"0.07","4":"0.62","_rn_":"4325"},{"1":"0.05","2":"0.14","3":"0.05","4":"0.76","_rn_":"4337"},{"1":"0.06","2":"0.26","3":"0.04","4":"0.64","_rn_":"4346"},{"1":"0.02","2":"0.50","3":"0.15","4":"0.33","_rn_":"4349"},{"1":"0.30","2":"0.43","3":"0.15","4":"0.12","_rn_":"4354"},{"1":"0.08","2":"0.39","3":"0.46","4":"0.07","_rn_":"4360"},{"1":"0.35","2":"0.34","3":"0.21","4":"0.10","_rn_":"4368"},{"1":"0.04","2":"0.25","3":"0.38","4":"0.33","_rn_":"4374"},{"1":"0.05","2":"0.04","3":"0.12","4":"0.79","_rn_":"4377"},{"1":"0.01","2":"0.03","3":"0.03","4":"0.93","_rn_":"4393"},{"1":"0.11","2":"0.20","3":"0.05","4":"0.64","_rn_":"4394"},{"1":"0.07","2":"0.30","3":"0.41","4":"0.22","_rn_":"4397"},{"1":"0.11","2":"0.56","3":"0.09","4":"0.24","_rn_":"4398"},{"1":"0.10","2":"0.69","3":"0.07","4":"0.14","_rn_":"4400"},{"1":"0.22","2":"0.29","3":"0.40","4":"0.09","_rn_":"4404"},{"1":"0.07","2":"0.01","3":"0.14","4":"0.78","_rn_":"4406"},{"1":"0.22","2":"0.15","3":"0.45","4":"0.18","_rn_":"4407"},{"1":"0.44","2":"0.19","3":"0.17","4":"0.20","_rn_":"4418"},{"1":"0.17","2":"0.55","3":"0.24","4":"0.04","_rn_":"4427"},{"1":"0.15","2":"0.09","3":"0.25","4":"0.51","_rn_":"4434"},{"1":"0.03","2":"0.52","3":"0.26","4":"0.19","_rn_":"4437"},{"1":"0.57","2":"0.09","3":"0.09","4":"0.25","_rn_":"4447"},{"1":"0.27","2":"0.13","3":"0.16","4":"0.44","_rn_":"4452"},{"1":"0.12","2":"0.31","3":"0.11","4":"0.46","_rn_":"4454"},{"1":"0.43","2":"0.06","3":"0.14","4":"0.37","_rn_":"4458"},{"1":"0.02","2":"0.83","3":"0.07","4":"0.08","_rn_":"4460"},{"1":"0.30","2":"0.26","3":"0.03","4":"0.41","_rn_":"4467"},{"1":"0.06","2":"0.40","3":"0.41","4":"0.13","_rn_":"4476"},{"1":"0.45","2":"0.07","3":"0.16","4":"0.32","_rn_":"4479"},{"1":"0.07","2":"0.12","3":"0.38","4":"0.43","_rn_":"4486"},{"1":"0.11","2":"0.16","3":"0.27","4":"0.46","_rn_":"4487"},{"1":"0.78","2":"0.04","3":"0.08","4":"0.10","_rn_":"4488"},{"1":"0.24","2":"0.29","3":"0.31","4":"0.16","_rn_":"4501"},{"1":"0.71","2":"0.04","3":"0.03","4":"0.22","_rn_":"4503"},{"1":"0.02","2":"0.15","3":"0.65","4":"0.18","_rn_":"4506"},{"1":"0.07","2":"0.17","3":"0.64","4":"0.12","_rn_":"4508"},{"1":"0.36","2":"0.49","3":"0.10","4":"0.05","_rn_":"4512"},{"1":"0.14","2":"0.70","3":"0.10","4":"0.06","_rn_":"4531"},{"1":"0.28","2":"0.44","3":"0.17","4":"0.11","_rn_":"4532"},{"1":"0.31","2":"0.17","3":"0.41","4":"0.11","_rn_":"4534"},{"1":"0.28","2":"0.27","3":"0.41","4":"0.04","_rn_":"4538"},{"1":"0.53","2":"0.14","3":"0.26","4":"0.07","_rn_":"4539"},{"1":"0.31","2":"0.56","3":"0.05","4":"0.08","_rn_":"4540"},{"1":"0.23","2":"0.11","3":"0.15","4":"0.51","_rn_":"4546"},{"1":"0.45","2":"0.18","3":"0.01","4":"0.36","_rn_":"4552"},{"1":"0.03","2":"0.13","3":"0.08","4":"0.76","_rn_":"4557"},{"1":"0.03","2":"0.05","3":"0.85","4":"0.07","_rn_":"4561"},{"1":"0.12","2":"0.63","3":"0.17","4":"0.08","_rn_":"4574"},{"1":"0.20","2":"0.49","3":"0.16","4":"0.15","_rn_":"4576"},{"1":"0.24","2":"0.45","3":"0.09","4":"0.22","_rn_":"4580"},{"1":"0.17","2":"0.62","3":"0.13","4":"0.08","_rn_":"4587"},{"1":"0.30","2":"0.25","3":"0.35","4":"0.10","_rn_":"4590"},{"1":"0.64","2":"0.20","3":"0.10","4":"0.06","_rn_":"4594"},{"1":"0.22","2":"0.31","3":"0.13","4":"0.34","_rn_":"4606"},{"1":"0.18","2":"0.09","3":"0.50","4":"0.23","_rn_":"4607"},{"1":"0.03","2":"0.49","3":"0.20","4":"0.28","_rn_":"4609"},{"1":"0.09","2":"0.47","3":"0.14","4":"0.30","_rn_":"4612"},{"1":"0.14","2":"0.58","3":"0.14","4":"0.14","_rn_":"4621"},{"1":"0.30","2":"0.10","3":"0.38","4":"0.22","_rn_":"4622"},{"1":"0.60","2":"0.05","3":"0.21","4":"0.14","_rn_":"4632"},{"1":"0.06","2":"0.32","3":"0.39","4":"0.23","_rn_":"4636"},{"1":"0.19","2":"0.06","3":"0.65","4":"0.10","_rn_":"4646"},{"1":"0.06","2":"0.58","3":"0.28","4":"0.08","_rn_":"4649"},{"1":"0.25","2":"0.15","3":"0.12","4":"0.48","_rn_":"4656"},{"1":"0.07","2":"0.81","3":"0.07","4":"0.05","_rn_":"4666"},{"1":"0.13","2":"0.45","3":"0.29","4":"0.13","_rn_":"4672"},{"1":"0.09","2":"0.26","3":"0.36","4":"0.29","_rn_":"4677"},{"1":"0.10","2":"0.10","3":"0.10","4":"0.70","_rn_":"4685"},{"1":"0.14","2":"0.27","3":"0.32","4":"0.27","_rn_":"4688"},{"1":"0.13","2":"0.12","3":"0.10","4":"0.65","_rn_":"4689"},{"1":"0.07","2":"0.29","3":"0.37","4":"0.27","_rn_":"4690"},{"1":"0.19","2":"0.37","3":"0.33","4":"0.11","_rn_":"4692"},{"1":"0.17","2":"0.47","3":"0.11","4":"0.25","_rn_":"4694"},{"1":"0.02","2":"0.67","3":"0.21","4":"0.10","_rn_":"4701"},{"1":"0.17","2":"0.41","3":"0.31","4":"0.11","_rn_":"4712"},{"1":"0.04","2":"0.60","3":"0.15","4":"0.21","_rn_":"4714"},{"1":"0.44","2":"0.22","3":"0.08","4":"0.26","_rn_":"4716"},{"1":"0.34","2":"0.25","3":"0.20","4":"0.21","_rn_":"4741"},{"1":"0.15","2":"0.05","3":"0.03","4":"0.77","_rn_":"4754"},{"1":"0.03","2":"0.08","3":"0.09","4":"0.80","_rn_":"4769"},{"1":"0.03","2":"0.82","3":"0.12","4":"0.03","_rn_":"4770"},{"1":"0.07","2":"0.44","3":"0.47","4":"0.02","_rn_":"4773"},{"1":"0.08","2":"0.72","3":"0.14","4":"0.06","_rn_":"4774"},{"1":"0.02","2":"0.74","3":"0.19","4":"0.05","_rn_":"4783"},{"1":"0.27","2":"0.32","3":"0.24","4":"0.17","_rn_":"4794"},{"1":"0.02","2":"0.05","3":"0.29","4":"0.64","_rn_":"4799"},{"1":"0.05","2":"0.42","3":"0.33","4":"0.20","_rn_":"4809"},{"1":"0.01","2":"0.09","3":"0.07","4":"0.83","_rn_":"4810"},{"1":"0.34","2":"0.28","3":"0.06","4":"0.32","_rn_":"4812"},{"1":"0.08","2":"0.39","3":"0.12","4":"0.41","_rn_":"4817"},{"1":"0.32","2":"0.29","3":"0.24","4":"0.15","_rn_":"4818"},{"1":"0.13","2":"0.27","3":"0.23","4":"0.37","_rn_":"4822"},{"1":"0.02","2":"0.18","3":"0.09","4":"0.71","_rn_":"4833"},{"1":"0.07","2":"0.07","3":"0.41","4":"0.45","_rn_":"4845"},{"1":"0.16","2":"0.57","3":"0.23","4":"0.04","_rn_":"4848"},{"1":"0.49","2":"0.21","3":"0.19","4":"0.11","_rn_":"4858"},{"1":"0.25","2":"0.02","3":"0.13","4":"0.60","_rn_":"4884"},{"1":"0.12","2":"0.04","3":"0.05","4":"0.79","_rn_":"4885"},{"1":"0.19","2":"0.09","3":"0.23","4":"0.49","_rn_":"4889"},{"1":"0.11","2":"0.04","3":"0.02","4":"0.83","_rn_":"4895"},{"1":"0.16","2":"0.03","3":"0.47","4":"0.34","_rn_":"4914"},{"1":"0.04","2":"0.15","3":"0.15","4":"0.66","_rn_":"4919"},{"1":"0.05","2":"0.53","3":"0.12","4":"0.30","_rn_":"4923"},{"1":"0.25","2":"0.41","3":"0.08","4":"0.26","_rn_":"4925"},{"1":"0.12","2":"0.09","3":"0.06","4":"0.73","_rn_":"4929"},{"1":"0.07","2":"0.25","3":"0.12","4":"0.56","_rn_":"4933"},{"1":"0.10","2":"0.02","3":"0.07","4":"0.81","_rn_":"4935"},{"1":"0.04","2":"0.08","3":"0.76","4":"0.12","_rn_":"4938"},{"1":"0.28","2":"0.06","3":"0.01","4":"0.65","_rn_":"4940"},{"1":"0.57","2":"0.07","3":"0.17","4":"0.19","_rn_":"4943"},{"1":"0.17","2":"0.05","3":"0.55","4":"0.23","_rn_":"4951"},{"1":"0.77","2":"0.15","3":"0.06","4":"0.02","_rn_":"4959"},{"1":"0.30","2":"0.52","3":"0.12","4":"0.06","_rn_":"4965"},{"1":"0.30","2":"0.14","3":"0.32","4":"0.24","_rn_":"4972"},{"1":"0.35","2":"0.31","3":"0.06","4":"0.28","_rn_":"4973"},{"1":"0.58","2":"0.24","3":"0.05","4":"0.13","_rn_":"4975"},{"1":"0.60","2":"0.09","3":"0.09","4":"0.22","_rn_":"4980"},{"1":"0.11","2":"0.71","3":"0.04","4":"0.14","_rn_":"4983"},{"1":"0.14","2":"0.32","3":"0.06","4":"0.48","_rn_":"4988"},{"1":"0.12","2":"0.30","3":"0.18","4":"0.40","_rn_":"4989"},{"1":"0.04","2":"0.08","3":"0.38","4":"0.50","_rn_":"5002"},{"1":"0.36","2":"0.07","3":"0.07","4":"0.50","_rn_":"5004"},{"1":"0.04","2":"0.09","3":"0.44","4":"0.43","_rn_":"5013"},{"1":"0.03","2":"0.06","3":"0.05","4":"0.86","_rn_":"5014"},{"1":"0.35","2":"0.08","3":"0.10","4":"0.47","_rn_":"5020"},{"1":"0.04","2":"0.26","3":"0.05","4":"0.65","_rn_":"5029"},{"1":"0.35","2":"0.16","3":"0.09","4":"0.40","_rn_":"5030"},{"1":"0.19","2":"0.31","3":"0.22","4":"0.28","_rn_":"5032"},{"1":"0.20","2":"0.27","3":"0.18","4":"0.35","_rn_":"5042"},{"1":"0.34","2":"0.04","3":"0.36","4":"0.26","_rn_":"5044"},{"1":"0.61","2":"0.01","3":"0.26","4":"0.12","_rn_":"5054"},{"1":"0.06","2":"0.25","3":"0.05","4":"0.64","_rn_":"5055"},{"1":"0.07","2":"0.04","3":"0.26","4":"0.63","_rn_":"5057"},{"1":"0.08","2":"0.20","3":"0.20","4":"0.52","_rn_":"5059"},{"1":"0.10","2":"0.05","3":"0.17","4":"0.68","_rn_":"5065"},{"1":"0.07","2":"0.03","3":"0.04","4":"0.86","_rn_":"5067"},{"1":"0.05","2":"0.51","3":"0.15","4":"0.29","_rn_":"5068"},{"1":"0.04","2":"0.26","3":"0.48","4":"0.22","_rn_":"5070"},{"1":"0.06","2":"0.50","3":"0.15","4":"0.29","_rn_":"5072"},{"1":"0.54","2":"0.11","3":"0.22","4":"0.13","_rn_":"5075"},{"1":"0.23","2":"0.25","3":"0.46","4":"0.06","_rn_":"5082"},{"1":"0.09","2":"0.15","3":"0.65","4":"0.11","_rn_":"5083"},{"1":"0.29","2":"0.29","3":"0.32","4":"0.10","_rn_":"5085"},{"1":"0.17","2":"0.02","3":"0.60","4":"0.21","_rn_":"5095"},{"1":"0.47","2":"0.11","3":"0.10","4":"0.32","_rn_":"5103"},{"1":"0.02","2":"0.03","3":"0.01","4":"0.94","_rn_":"5106"},{"1":"0.07","2":"0.10","3":"0.27","4":"0.56","_rn_":"5107"},{"1":"0.28","2":"0.54","3":"0.08","4":"0.10","_rn_":"5115"},{"1":"0.20","2":"0.62","3":"0.07","4":"0.11","_rn_":"5119"},{"1":"0.27","2":"0.30","3":"0.30","4":"0.13","_rn_":"5128"},{"1":"0.15","2":"0.64","3":"0.08","4":"0.13","_rn_":"5134"},{"1":"0.55","2":"0.04","3":"0.23","4":"0.18","_rn_":"5138"},{"1":"0.22","2":"0.53","3":"0.10","4":"0.15","_rn_":"5144"},{"1":"0.35","2":"0.04","3":"0.48","4":"0.13","_rn_":"5147"},{"1":"0.14","2":"0.23","3":"0.57","4":"0.06","_rn_":"5155"},{"1":"0.69","2":"0.06","3":"0.07","4":"0.18","_rn_":"5158"},{"1":"0.33","2":"0.51","3":"0.09","4":"0.07","_rn_":"5160"},{"1":"0.60","2":"0.18","3":"0.03","4":"0.19","_rn_":"5163"},{"1":"0.10","2":"0.23","3":"0.09","4":"0.58","_rn_":"5165"},{"1":"0.28","2":"0.57","3":"0.07","4":"0.08","_rn_":"5166"}],"options":{"columns":{"min":{},"max":[10],"total":[4]},"rows":{"min":[10],"max":[10],"total":[4311]},"pages":{}}}
  </script>
</div>

<!-- rnb-frame-end -->

<!-- rnb-chunk-end -->


<!-- rnb-text-begin -->



<!-- rnb-text-end -->


<!-- rnb-chunk-begin -->


<!-- rnb-source-begin eyJkYXRhIjoiYGBgclxuY29sbmFtZXMoQ1ZSRikgPC0gYyhcIkNoMVwiLCBcIkNoMlwiLCBcIkNoM1wiLCBcIkNoNFwiKVxuQ1Zsb3NzIDwtIGxvZ2xvc3ModGVzdFNldCwgQ1ZSRilcbkNWbG9zc1xuYGBgIn0= -->

```r
colnames(CVRF) <- c("Ch1", "Ch2", "Ch3", "Ch4")
CVloss <- logloss(testSet, CVRF)
CVloss
```

<!-- rnb-source-end -->

<!-- rnb-output-begin eyJkYXRhIjoiWzFdIDEuMDY5Njc5XG4ifQ== -->

```
[1] 1.069679
```



<!-- rnb-output-end -->

<!-- rnb-chunk-end -->


<!-- rnb-text-begin -->



<!-- rnb-text-end -->


<!-- rnb-chunk-begin -->


<!-- rnb-source-begin eyJkYXRhIjoiYGBgclxuZ2V0TnVtIDwtIHJlYWQuY3N2KFwiLi90ZXN0MV90cnlpbmcuY3N2XCIpXG5cbnRlc3QgPC0gc3Vic2V0KGdldE51bSwgc2VsZWN0ID0gLWMoTm8sIENhc2UsIENDNCxHTjQsTlM0LEJVNCxGQTQsTEQ0LEJaNCxGQzQsRlA0LFJQNCxQUDQsS0E0LFNDNCxUUzQsTlY0LE1BNCxMQjQsQUY0LEhVNCxQcmljZTQpKVxuXG4jIENvbnZlcnQgdGhlIGluY29tZWEgY29sdW1uIHRvIGJpbnMgYW5kIGxhYmVsIHRoZW1cbiN0ZXN0JGluY29tZWJpbnMgPC0gY3V0KHRlc3QkaW5jb21lYSwgYnJlYWtzID0gYnJlYWtzLCBsYWJlbHMgPSBsYWJlbHMsIGluY2x1ZGUubG93ZXN0ID0gVFJVRSlcblxuIyBDb252ZXJ0IHRoZSBpbmNvbWVfYmlucyBjb2x1bW4gdG8gYSBmYWN0b3JcbiN0ZXN0JGluY29tZWJpbnMgPC0gYXMuZmFjdG9yKHRlc3QkaW5jb21lYmlucylcblxuIyBEZWxldGUgb3RoZXIgaW5jb21lIGNvbHVtbnNcbiN0ZXN0IDwtIHN1YnNldCh0ZXN0LCBzZWxlY3QgPSAtYyhpbmNvbWVpbmQsaW5jb21lKSlcblxuIyBGaW5kIHRoZSBwb3NpdGlvbiBvZiAnaW5jb21lYSdcbiNwb3MgPC0gd2hpY2gobmFtZXModGVzdCkgPT0gXCJpbmNvbWVhXCIpXG5cbiMgQ3JlYXRlIGEgbmV3IGNvbHVtbiBvcmRlclxuI25ld19vcmRlciA8LSBjKG5hbWVzKHRlc3QpWzE6KHBvcy0xKV0sIFwiaW5jb21lYmluc1wiLCBuYW1lcyh0ZXN0KVtwb3M6KGxlbmd0aCh0ZXN0KS0xKV0pXG5cbiMgUmVhcnJhbmdlIHRoZSBjb2x1bW5zXG4jdGVzdCA8LSB0ZXN0WywgbmV3X29yZGVyXVxuXG4jaGVhZChzYWZldHkpXG5cbmBgYCJ9 -->

```r
getNum <- read.csv("./test1_trying.csv")

test <- subset(getNum, select = -c(No, Case, CC4,GN4,NS4,BU4,FA4,LD4,BZ4,FC4,FP4,RP4,PP4,KA4,SC4,TS4,NV4,MA4,LB4,AF4,HU4,Price4))

# Convert the incomea column to bins and label them
#test$incomebins <- cut(test$incomea, breaks = breaks, labels = labels, include.lowest = TRUE)

# Convert the income_bins column to a factor
#test$incomebins <- as.factor(test$incomebins)

# Delete other income columns
#test <- subset(test, select = -c(incomeind,income))

# Find the position of 'incomea'
#pos <- which(names(test) == "incomea")

# Create a new column order
#new_order <- c(names(test)[1:(pos-1)], "incomebins", names(test)[pos:(length(test)-1)])

# Rearrange the columns
#test <- test[, new_order]

#head(safety)

```

<!-- rnb-source-end -->

<!-- rnb-chunk-end -->


<!-- rnb-text-begin -->



<!-- rnb-text-end -->


<!-- rnb-chunk-begin -->


<!-- rnb-source-begin eyJkYXRhIjoiYGBgclxuI3Rlc3QgPC0gc3Vic2V0KHRlc3QsIHNlbGVjdCA9IGMoQ2hvaWNlLCBQcmljZTMsIFByaWNlMiwgUHJpY2UxLCBhZ2VhLCBpbmNvbWVhLCBtaWxlc2EsIG5pZ2h0YSwgeWVhciwgeWVhcmluZCwgaW5jb21lLCBpbmNvbWVpbmQsIG1pbGVzLCBtaWxlc2luZCAsbmlnaHRpbmQsIG5pZ2h0LCBwcGFya2luZCwgcHBhcmssIHNlZ21lbnQsIHNlZ21lbnRpbmQsIHJlZ2lvbiwgcmVnaW9uaW5kLCBCVTMpKVxuI2ZhY3RfY29sIDwtIGNvbG5hbWVzKHRlc3QpW3NhcHBseSh0ZXN0LGlzLmNoYXJhY3RlcildXG5cbiNmb3IoaSBpbiBmYWN0X2NvbClcbiMgICAgICAgc2V0KHRlc3Qsaj1pLHZhbHVlID0gZmFjdG9yKHRlc3RbW2ldXSkpXG5cblxuI3Rlc3RDbGFzcyA8LSBtYWtlQ2xhc3NpZlRhc2soZGF0YSA9IHRlc3QsdGFyZ2V0ID0gXCJDaG9pY2VcIilcbiNzZXQuc2VlZChzZWVkKVxuI210cnkgPC0gdHVuZVJGKHNhZmV0eVsxOm5jb2woc2FmZXR5KS0xXSwgYXMuZmFjdG9yKHNhZmV0eSRDaG9pY2UpLFxuIyAgICAgICAgICAgICAgIHN0ZXBGYWN0b3I9MS41LGltcHJvdmU9MC4wMSwgdHJhY2U9VFJVRSwgcGxvdD1GQUxTRSlcbiNiZXN0Lm0gPC0gbXRyeVttdHJ5WywgMl0gPT0gbWluKG10cnlbLCAyXSksIDFdXG5cbiNzZXQuc2VlZChzZWVkKVxuI3JmIDwtcmFuZG9tRm9yZXN0KGFzLmZhY3RvcihDaG9pY2Upfi4sIGRhdGEgPSBzYWZldHksIG10cnk9YmVzdC5tLCBpbXBvcnRhbmNlPVRSVUUpXG5cbmZpbmFsX3ByZWRpY3QgPC0gcHJlZGljdChtb2RlbCwgdGVzdCwgdHlwZT1cInByb2JcIilcbiNmaW5hbF9wcmVkaWN0IDwtIGZpbmFsX3ByZWRpY3QkZGF0YVssMzo2XVxuY29sbmFtZXMoZmluYWxfcHJlZGljdCkgPC0gYyhcIkNoMVwiLFwiQ2gyXCIsXCJDaDNcIixcIkNoNFwiKVxuZmluYWxfcHJlZGljdF9kZiA8LSBhcy5kYXRhLmZyYW1lKGZpbmFsX3ByZWRpY3QpXG5maW5hbF9wcmVkaWN0X2RmJE5vIDwtIGdldE51bSROb1xuXG5maW5hbF9wcmVkaWN0X2RmIDwtIGZpbmFsX3ByZWRpY3RfZGZbYyhcIk5vXCIsXCJDaDFcIixcIkNoMlwiLFwiQ2gzXCIsXCJDaDRcIildXG5gYGAifQ== -->

```r
#test <- subset(test, select = c(Choice, Price3, Price2, Price1, agea, incomea, milesa, nighta, year, yearind, income, incomeind, miles, milesind ,nightind, night, pparkind, ppark, segment, segmentind, region, regionind, BU3))
#fact_col <- colnames(test)[sapply(test,is.character)]

#for(i in fact_col)
#       set(test,j=i,value = factor(test[[i]]))


#testClass <- makeClassifTask(data = test,target = "Choice")
#set.seed(seed)
#mtry <- tuneRF(safety[1:ncol(safety)-1], as.factor(safety$Choice),
#               stepFactor=1.5,improve=0.01, trace=TRUE, plot=FALSE)
#best.m <- mtry[mtry[, 2] == min(mtry[, 2]), 1]

#set.seed(seed)
#rf <-randomForest(as.factor(Choice)~., data = safety, mtry=best.m, importance=TRUE)

final_predict <- predict(model, test, type="prob")
#final_predict <- final_predict$data[,3:6]
colnames(final_predict) <- c("Ch1","Ch2","Ch3","Ch4")
final_predict_df <- as.data.frame(final_predict)
final_predict_df$No <- getNum$No

final_predict_df <- final_predict_df[c("No","Ch1","Ch2","Ch3","Ch4")]
```

<!-- rnb-source-end -->

<!-- rnb-chunk-end -->


<!-- rnb-text-begin -->



<!-- rnb-text-end -->


<!-- rnb-chunk-begin -->


<!-- rnb-source-begin eyJkYXRhIjoiYGBgclxud3JpdGUuY3N2KGZpbmFsX3ByZWRpY3RfZGYsIGZpbGUgPSBcIi4vSm9hc2hfcmFuZEZvcmVzdF9ub2JpbnMuY3N2XCIsIHJvdy5uYW1lcyA9IEZBTFNFKVxuYGBgIn0= -->

```r
write.csv(final_predict_df, file = "./Joash_randForest_nobins.csv", row.names = FALSE)
```

<!-- rnb-source-end -->

<!-- rnb-chunk-end -->


<!-- rnb-text-begin -->



<!-- rnb-text-end -->


<!-- rnb-chunk-begin -->


<!-- rnb-source-begin eyJkYXRhIjoiYGBgclxudHJhaW50YXNrXG5gYGAifQ== -->

```r
traintask
```

<!-- rnb-source-end -->

<!-- rnb-chunk-end -->


<!-- rnb-chunk-begin -->


<!-- rnb-source-begin eyJkYXRhIjoiYGBgclxudGVzdENsYXNzXG5gYGAifQ== -->

```r
testClass
```

<!-- rnb-source-end -->

<!-- rnb-chunk-end -->


<!-- rnb-text-begin -->



<!-- rnb-text-end -->

