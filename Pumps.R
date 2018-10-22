# Load Packages

# Data Manipulation
library(dplyr)
library(lubridate)
library(dummies)

# Data Visualization 
library(ggplot2)

# Prediction
library(nnet)
library(neuralnet)
library(class)
library(rpart)
library(randomForest)
library(party)
library(e1071)
library(gbm)
library(caret)
library(DMwR)
library(xgboost)

library(tictoc)

# Load Data
setwd("Documents/DataScience/Competitions/Pumps/")
train_x <- read.csv("train_values.csv")
train_y <- read.csv("train_labels.csv")

test <- read.csv("test_values.csv")

# Merge train & test sets for data manipulation
data <- rbind(train_x, test)


# Data structure and summary
str(data)
summary(data)

# Note there are no NA's but many 0's so most likely missing values are coded as 0's.
# In many cases it seems that Unknown also designates a missing value.

na_str <- c("Unknown", "unknown", "", " ", "none", "other", "_", "-")

# Data cleaning and feature engineering

# recorded_by -- only 1 level, will be removed

data$date_recorded <- ymd(data$date_recorded)


# Since there are many character features I will convert all such variables to lower case 
# in order to avoid having the same information coded differently (e.g. Government vs GOVERMENT)

chr_cols <- c("funder", "installer", "wpt_name", "basin", "subvillage", "region", "lga",
              "ward", "recorded_by", "scheme_management", "scheme_name", "extraction_type",
              "extraction_type_group", "extraction_type_class", "management",
              "management_group", "payment", "payment_type", "water_quality",
              "quality_group", "quantity", "quantity_group", "source", "source_type",
              "source_class", "waterpoint_type", "waterpoint_type_group")

data <- data %>% mutate_at(vars(one_of(chr_cols)), funs(tolower))

for(col in colnames(data)){
  data[[col]][data[[col]] %in% na_str] <- NA
}

data <- data %>% mutate_at(vars(one_of(chr_cols)), funs(factor))
data <- droplevels(data)

# amount_tsh
length(data$amount_tsh[data$amount_tsh == 0])/length(data$amount_tsh)
# 70% of the entries in amount_tsh are missing, therefore will not be used.


# funder & installer  
# Both variables funder and installer have too many levels (2138 and 2160 reps.). 
# Therefore will extract only the first 4 letter from the name and then keep only the 
# top 20 entries

data$funder <- substr(data$funder, 1, 4)
data$funder[data$funder == "0"] <- NA
data$funder <- factor(data$funder)

fund_4_top <- factor(names(summary(data$funder))[1:20])
levels(data$funder) <- c(levels(data$funder), "other")
data$funder[!data$funder %in% fund_4_top] <- "other"


data$installer <- substr(data$installer, 1, 4)
data$installer[data$installer == "0"] <- NA
data$installer <- factor(data$installer)

inst_4_top <- factor(names(summary(data$installer))[1:20])
levels(data$installer) <- c(levels(data$installer), "other")
data$installer[!data$installer %in% fund_4_top] <- "other"

# num_private
length(data$num_private[data$num_private == 0])/length(data$num_private)
# 99% of the entries in num_private are missing, therefore will not be used.

# population
length(data$population[data$population == 0])/length(data$population)
# 36% of the entries in population are missing, therefore will not be used. 
data$population[data$population == 0] <- NA


# Extraction type
data %>% 
  group_by(extraction_type_class, extraction_type_group, extraction_type) %>% 
  tally()
# Same information on different levels of granularity, will use the variable 
# extraction_type_group.

# Construction year
length(data$construction_year[data$construction_year == 0])/length(data$construction_year)
# 35% of the entries in construction_year are missing
data$construction_year[data$construction_year == 0] <- NA

# Check if all entries are valid
summary(year(data$date_recorded) - data$construction_year)

# There are some negative results, will set them to NA
data$construction_year[year(data$date_recorded) - data$construction_year < 0] <- NA

# Payment
table(data$payment, data$payment_type)
# The exact same information, will use payment.


# Water quality
table(data$water_quality, data$quality_group)
# Same information on different levels of granularity, will use the finer variable 
# water_quality

# Water quantity
table(data$quantity, data$quantity_group)
# These two variables are exactly the same, will use quantity

# Waterpoint type
table(data$waterpoint_type, data$waterpoint_type_group)
# These two variables are almost the same, the only difference is in communal standpipe pumps.
# Will use the finer variable waterpoint_type

# Source
data %>% group_by(source_class, source_type, source) %>% tally()
# Same information on different levels of granularity, will use the middle level variable 
# source_type


# Management & Scheme
table(data$management, data$management_group)
# Same information on different levels of granularity, will use the finer variable management.

table(data$scheme_name, data$scheme_management)
# The variable scheme_name has too many levels, therefore will use the coarser variable
# scheme_management

# Location
# There are several variables that seem to describe the location of the pumps: 
# gps_height, latitude, longitude, basin, subvillage, region, region_code, district_code, 
# lga?, ward

# According to Wikipedia, the administrative subdivisions of Tanzania are 
# Region > District > Division > Ward (Urban) > Street
#                                Ward (Rural) > Village > Hamlet
# Can be confirmed from the given data as well: there are multiple districts within a region

table(data$region, data$region_code)
# A unique region_code corresponds to most of the regions. Exceptions are Arusha (codes 2 & 24),
# Lindi (codes 8 & 80), Mtwara (codes 9, 90 & 99), Mwanza (codes 17 & 19), Pwani (codes 6 & 60),
# Shinyanga (codes 14 & 17), Tanga (codes 4 & 5). Will use the feature region.

length(data$gps_height[data$gps_height == 0])/length(data$gps_height)
# 34% of the entries in gps_height are missing 
data$gps_height[data$gps_height == 0] <- NA
# Will use ward to "impute" some (~2%) of the missing values
data <- data %>% 
  group_by(region, district_code, ward, subvillage) %>%
  mutate(subGpsH = mean(gps_height, na.rm = TRUE))%>%
  ungroup()

data <- data %>% 
  group_by(region, district_code, ward) %>%
  mutate(wardGpsH = mean(gps_height, na.rm = TRUE))%>%
  ungroup()

data <- data %>% 
  group_by(region, district_code) %>%
  mutate(disGpsH = mean(gps_height, na.rm = TRUE))%>%
  ungroup()

data <- data %>% mutate(gps_height = ifelse(!is.na(gps_height), gps_height, 
                                            ifelse(!is.na(subGpsH), subGpsH, 
                                                   ifelse(!is.na(warGpsH), warGpsH, disGpsH))))

# 0's in latitude and longitude make no sense since there are no such locations in Tanzania, 
# therefore these designate missing values.

data$latitude <- round(data$latitude, digits = 6)
length(data$latitude[data$latitude == 0])/length(data$latitude)
# 3% (1812 out of 59400 in train, resp 2269 out of 74250 in the combined data) 
# missing entries in latitude

length(data$longitude[data$longitude == 0])/length(data$longitude)
# 3% (1812 out of 59400 in train, resp 2269 out of 74250 in the combined data) 
# missing entries in longitude

# There are 1812, resp 2269, observation missing both latitude & latitude.

summary(data[data$latitude == 0 & data$longitude == 0, ])

# Note: For all such observations (with missing latitude & longitude), the features gps_height, 
# population, and construction_year are missing as well. Moreover, all such observations are
# in the basins Lake Victoria and Lake Tanganyika, in the regions Shinyaga an Mwanza, and 
# in lga Bariadi, Geita and Magu. According to Wikipedia, both Geita and Magu are in the 
# region Mwanza, and Bariadi is in the region Simiyu which is not among the levels of data$region.

# Will impute missing values with the mean within region & district_code.

data$latitude[data$latitude == 0] <- NA
data$longitude[data$longitude == 0] <- NA


data <- data %>% 
  group_by(region, district_code, ward, subvillage) %>%
  mutate(subLat = mean(latitude, na.rm = TRUE),
         subLong = mean(longitude, na.rm = TRUE)) %>%
  ungroup()

data <- data %>% 
  group_by(region, district_code, ward) %>%
  mutate(warLat = mean(latitude, na.rm = TRUE),
         warLong = mean(longitude, na.rm = TRUE)) %>%
  ungroup()


data <- data %>% 
  group_by(region, district_code) %>%
  mutate(disLat = mean(latitude, na.rm = TRUE),
         disLong = mean(longitude, na.rm = TRUE)) %>%
  ungroup()


data <- data %>% mutate(latitude = ifelse(!is.na(latitude), latitude,  
                                          ifelse(!is.na(subLat), subLat, 
                                                 ifelse(!is.na(warLat), warLat, disLat))),
                        longitude = ifelse(!is.na(longitude), longitude,
                                           ifelse(!is.na(subLong), subLong, 
                                                  ifelse(!is.na(warLong), warLong, disLong))))

# Define unique lat_long feature by combining latitude and longitude and impute the
# missing entries by the median
data$lat_long <- data$latitude + max(data$latitude, na.rm = TRUE) * data$longitude
data$lat_long[is.na(data$lat_long)] <- median(data$lat_long, na.rm = TRUE)

# Seasonality
# Will extract month from date_recorded to try to capture seasonal effects
data$date_recorded <- ymd(data$date_recorded)
data$month <- factor(month(data$date_recorded))


# wpt_name seems to be the name of the waterpoint which brings no information about the 
# functionality of the pump, therefore will not be used


# Assign new level "unknown" to all categorical variables with missing entries.
cat_predictors_NA <- c("scheme_management", "extraction_type_group", "management", 
                       "payment", "water_quality",
                      "quantity", "source_type", "waterpoint_type")

for (pred in cat_predictors_NA){
  levels(data[[pred]]) <- c(levels(data[[pred]]), "unknown")
  data[[pred]][is.na(data[pred])] <- "unknown" 
}


data <- droplevels(data)


# ========================================================================
# Select only the variables which will be used 
predictors <- c("funder", "installer", "extraction_type_group",
                "payment", "water_quality", "quantity",
                "waterpoint_type", "source_type", "management", "scheme_management",
                "region", "lat_long", "month")

data_X <- data.frame(data[, names(data) %in%  predictors])

# Split back into train and test sets
train <- data.frame(data_X[1:59400, ], status_group = train_y$status_group)

test <- data_X[59401:74250, ]


# Some algorithms work only with numerical predictors, other work better when the dimension
# of the data is lower. Therefore we first dummify the data and then reduce it via PCA.

# Dummify the categorical variables
cat_predictors <- c("funder", "installer", "region", "lga", "scheme_management",
                   "extraction_type_group", "management", "payment", "water_quality",
                   "quantity", "source_type", "waterpoint_type", "month")

data_X_dummy <- dummy.data.frame(data_X, 
                                  names = cat_predictors)

train_X_dummy <- data.frame(data_X_dummy[1:59400, ])
train_dummy <- data.frame(train_X_dummy, status_group = train_y$status_group)
test_X_dummy <- data.frame(data_X_dummy[59401:74250, ])

# PCA
pr_comp <- prcomp(na.exclude(train_X_dummy))
std_dev <- pr_comp$sdev
pr_var <- std_dev^2
prop_varex <- pr_var/sum(pr_var)
prop_varex[1:60]

plot(cumsum(prop_varex[1:70]), xlab = "Principal Component",
       ylab = "Cumulative Proportion of Variance Explained",
       type = "b")

# 70 components explain more than 98.6% variance in the data set.

train_pca <- data.frame(status_group = train_y$status_group, pr_comp$x)
train_pca70 <- train_pca[, 1:71]

test_pca <- predict(pr_comp, test_X_dummy)
test_pca70 <- test_pca[, 1:70]


# ===================
# Models
set.seed(42)

# CV 
n_folds <- 3


# Benchmark
accuracy <- 0

# Split into train and test sets for CV
cv_ind <- sample(seq_len(nrow(train)), size = floor(0.80 * nrow(train)))
cv_tr <- train[cv_ind, ]
cv_test <- train[-cv_ind, ]

# Train the model on the training set
pred <- rep("functional", length(cv_test$status_group))
  
conf <- table(pred, cv_test$status_group)
accuracy <- sum(diag(conf)) / sum(conf)


print(paste("CV accuracy of the benchmark is", accuracy))



# Logistic Regression
accuracy <- 0

for (i in 1:n_folds){
  # Split into train and test sets for CV
  cv_ind <- sample(seq_len(nrow(train_dummy)), size = floor(0.80 * nrow(train_dummy)))
  cv_tr <- train_dummy[cv_ind, ]
  cv_test <- train_dummy[-cv_ind, ]
  
  # Train the model on the training set
  model_logr <- multinom(status_group ~.,
                         data = cv_tr)
  
  pred <- predict(model_logr, 
                  newdata = cv_test,
                  prob = "probs")
  
  
  conf <- confusionMatrix(pred, cv_test$status_group)$table
  accuracy <- accuracy + sum(diag(conf)) / sum(conf)
}

print(paste("CV accuracy of Logistic Regression model is", accuracy / n_folds))



# Neural Network
accuracy <- 0

for (i in 1:n_folds){
  # Split into train and test set for CV
  cv_ind <- sample(seq_len(nrow(train_pca70)), size = floor(0.80 * nrow(train_pca70)))
  cv_tr <- train_pca70[cv_ind, ]
  cv_test <- train_pca70[-cv_ind, ]
  
  # Train the model on the training set
  model_nnet <- nnet(status_group ~. ,
                     data = cv_tr,
                     size = 50,
                     entropy = TRUE,
                     maxit = 200,
                     MaxNWts = 14000)
  
  pred <- as.factor(predict(model_nnet, 
                            newdata = cv_test,
                            type = "class"))
  
  
  conf <- conf <- table(pred, cv_test$status_group)
  accuracy <- accuracy + sum(diag(conf)) / sum(conf)
}

print(paste("CV accuracy of Neural Network model is", accuracy / n_folds))



# KNN
k_to_try = c(10, 20, 50, 100)
acc_k = rep(x = 0, times = length(k_to_try))

for(k in seq_along(k_to_try)) {
  accuracy <- 0
  for (i in 1:n_folds){
    # Split into train and test sets for CV
    cv_ind <- sample(seq_len(nrow(train_pca70)), size = floor(0.80 * nrow(train_pca70)))
    cv_tr <- train_pca70[cv_ind, ]
    cv_test <- train_pca70[-cv_ind, ]
    
    tr_x <- subset(cv_tr, select = -c(status_group))
    tr_y <- cv_tr$status_group
    test_x <- subset(cv_test, select = -c(status_group))
    
    # Train and predict
    pred <- knn(train = tr_x, test = test_x, cl = tr_y, k = k_to_try[k])
    conf <- table(pred, cv_test$status_group)
    accuracy <- accuracy + sum(diag(conf)) / sum(conf)
  }
  acc_k[k] = accuracy/n_folds
  print(paste("CV accuracy of KNN model on the reduced set with k =", k_to_try[k], 
              "is", acc_k[k]))
}



# Naive Bayes
accuracy <- 0

for (i in 1:n_folds){
  # Split into train and test sets for CV
  cv_ind <- sample(seq_len(nrow(train_dummy)), size = floor(0.80 * nrow(train_dummy)))
  cv_tr <- train_dummy[cv_ind, ]
  cv_test <- train_dummy[-cv_ind, ]
 
  # Train the model on the training set
  model_nbayes <- naiveBayes(status_group ~. ,
                              data = cv_tr)
  
  pred <- predict(model_nbayes,
                  newdata = cv_test,
                  type = "class")
  
  conf <- table(pred, cv_test$status_group)
  accuracy <- accuracy + sum(diag(conf)) / sum(conf)
  print(accuracy)
}

print(paste("CV accuracy of Naive Bayes model is", accuracy / n_folds))



# Decision Tree
accuracy <- 0

for (i in 1:n_folds){
  # Split into train and test sets for CV
  cv_ind <- sample(seq_len(nrow(train)), size = floor(0.80 * nrow(train)))
  cv_tr <- train[cv_ind, ]
  cv_test <- train[-cv_ind, ]
  
  # Train the model on the training set
  model_dectree <- ctree(status_group ~. ,
                         data = cv_tr)
  
  
  pred <- predict(model_dectree,
                  newdata = cv_test,
                  type = "response")
  
  conf <- table(pred, cv_test$status_group)
  accuracy <- accuracy + sum(diag(conf)) / sum(conf)
}

print(paste("CV accuracy of Decision Tree model is", accuracy / n_folds))



# Random Forest with RandomForest
accuracy <- 0

for (i in 1:n_folds){
  # Split into train and test sets for CV
  cv_ind <- sample(seq_len(nrow(train)), size = floor(0.80 * nrow(train)))
  cv_tr <- train[cv_ind, ]
  cv_test <- train[-cv_ind, ]
  # Train the model on the training set
  model_rforest <- randomForest(status_group ~.,
                          data = cv_tr, 
                          num.tree = 500)
  
  pred <- predict(model_rforest,
                  newdata = cv_test)
  
  
  conf <- table(pred, cv_test$status_group)
  accuracy <- accuracy + sum(diag(conf)) / sum(conf)
}

print(paste("CV accuracy of Random forest model is", accuracy / n_folds))



# xgboost
accuracy <- 0

for (i in 1:n_folds){
  # Split into train and test sets for CV
  cv_ind <- sample(seq_len(nrow(train)), size = floor(0.80 * nrow(train)))
  cv_tr <- train[cv_ind, ]
  cv_test <- train[-cv_ind, ]
  
  # Train the model on the training set
  tlabels <- as.numeric(as.factor(cv_tr$status_group)) - 1 
  model_xgboost <- xgboost(data = data.matrix(subset(cv_tr, select = -c(status_group))),
                           label = tlabels,
                           nrounds = 500,
                           max_depth = 12,
                           eta = 0.1,
                           gamma = 1,
                           colsample_bytree = 0.7,
                           min_child_weight = 2,
                           subsample = 1,
                           objecive = "multi:softprob",
                           num_class = length(unique(cv_tr$status_group)))
  
  pred <- predict(model_xgboost,
                  newdata = data.matrix(subset(cv_test, select = -c(status_group))))
  
  conf <- table(pred, cv_test$status_group)
  accuracy <- accuracy + sum(diag(conf)) / sum(conf)
}

print(paste("CV accuracy of xgboost model is", accuracy / n_folds))



# SVM with radial kernel
accuracy <- 0

for (i in 1:n_folds){
  # Split into train and test sets for CV
  cv_ind <- sample(seq_len(nrow(train_pca70)), size = floor(0.80 * nrow(train_pca70)))
  cv_tr <- train_pca70[cv_ind, ]
  cv_test <- train_pca70[-cv_ind, ]
  
  # Train the model on the training set
  model_svm <- svm(status_group ~.,
                   data = cv_tr, 
                   kernel = "radial")
  
  pred <- predict(model_svm,
                  newdata = cv_test)
  
  conf <- table(pred, cv_test$status_group)
  accuracy <- accuracy + sum(diag(conf)) / sum(conf)
}

print(paste("CV accuracy of SVM model with radial kernel is", accuracy / n_folds))



# SVM with linear kernel
accuracy <- 0

for (i in 1:n_folds){
  # Split into train and test sets for CV
  cv_ind <- sample(seq_len(nrow(train_pca70)), size = floor(0.80 * nrow(train_pca70)))
  cv_tr <- train_pca70[cv_ind, ]
  cv_test <- train_pca70[-cv_ind, ]
  
  # Train the model on the training set
  model_svm <- svm(status_group ~.,
                   data = cv_tr, 
                   kernel = "linear",
                   iter.max = 10000)
  
  pred <- predict(model_svm,
                  newdata = cv_test)
  
  conf <- table(pred, cv_test$status_group)
  accuracy <- accuracy + sum(diag(conf)) / sum(conf)
}

print(paste("CV accuracy of SVM model with linear kernel is", accuracy / n_folds))



# SVM with polynomial kernel
accuracy <- 0

for (i in 1:n_folds){
  # Split into train and test sets for CV
  cv_ind <- sample(seq_len(nrow(train_pca70)), size = floor(0.80 * nrow(train_pca70)))
  cv_tr <- train_pca70[cv_ind, ]
  cv_test <- train_pca70[-cv_ind, ]
  
  # Train the model on the training set
  model_svm <- svm(status_group ~.,
                   data = cv_tr, 
                   kernel = "polynomial")
  
  pred <- predict(model_svm,
                  newdata = cv_test)
  
  conf <- table(pred, cv_test$status_group)
  accuracy <- accuracy + sum(diag(conf)) / sum(conf)
}

print(paste("CV accuracy of SVM model with polynomial kernel is", accuracy / n_folds))



# SVM with sigmoid kernel
accuracy <- 0

for (i in 1:n_folds){
  # Split into train and test sets for CV
  cv_ind <- sample(seq_len(nrow(train_pca70)), size = floor(0.80 * nrow(train_pca70)))
  cv_tr <- train_pca70[cv_ind, ]
  cv_test <- train_pca70[-cv_ind, ]
  
  # Train the model on the training set
  model_svm <- svm(status_group ~.,
                   data = cv_tr, 
                   kernel = "sigmoid")
  
  pred <- predict(model_svm,
                  newdata = cv_test)
  
  conf <- table(pred, cv_test$status_group)
  accuracy <- accuracy + sum(diag(conf)) / sum(conf)
}

print(paste("CV accuracy of SVM model with sigmoid kernel is", accuracy / n_folds))



# Model submission
# Train the model on the training set
model_svm <- svm(status_group ~.,
                 data = train_pca70, 
                 kernel = "sigmoid")

pred <- predict(model_svm,
                newdata = test_pca70)

# Create submission data frame
test <- read.csv("test_values.csv")
submission <- data.frame(test$id)
submission$status_group <- pred
names(submission)[1] <- "id"

write.csv(submission,
          file = "svm_sigmoid.csv",
          row.names = FALSE)
# 


