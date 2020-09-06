
# Title: Predicting Chemical-induced Liver Toxicity Using High Content Imaging Phenotypes and Chemical Descriptors: Random Forest Approach
# Authors: Swapnil Chavan †*, Nikolai Scherbak †, Magnus Engwall †, Dirk Repsilber ‡*
# Affiliations:
# † School of Science and Technology, Örebro University, 70112 Örebro, Sweden.
# ‡ School of Medical Sciences, Örebro University, 70185 Örebro, Sweden.

# Description: Random forest model construction procedure
# Author: swapnil chavan
# Date: 17 July 2020

library(data.table)
library(dplyr)
library(splitstackshape)
library(caret)
library(randomForest)
library(pROC)
library(doMC)
library(Boruta)



# -----------------------------------------------------------------------------
#                              Model building
# -----------------------------------------------------------------------------


######################################################################################################
#             Approach - A:  No Hyperparameter Selection & No Feature Selection (NoHS_NoFS)
######################################################################################################

# 1. Model building with HCI phenotypes

# Step - 1: Remove zero variance variables

rm(list=ls())

avg_variables <- fread('avg_variables_346_inst_table.csv')

# keep only HCI phenotypes
hci_var_names <- names(avg_variables)[grep('Cells|Cytoplasm|Nuclei',names(avg_variables))]
x_data <- avg_variables[,hci_var_names, with=F]

# Remove descriptors with near zero variance
nzv <- nearZeroVar(x_data)
if (length(nzv) != 0){
	x_data <- x_data[,-nzv, with=F]
}
# no variable with zero or near zero variance
rm(nzv)

# add CLASS column
x_data[, Sample_id := avg_variables$Sample_id]
x_data[, CLASS := ifelse(avg_variables$TOXICITY == 'Hepatotoxic', 1, 0)]
x_data <- x_data[,c(1721:1722,1:1720)]

# replace '-' by underscore
colnames(x_data) <- gsub('\\-','_',colnames(x_data))

write.table(x_data, 'input_HCI.csv', row.names = F, col.names = T, sep = ',')

path <- getwd()

# 2. model construction

rm(list = setdiff(ls(),'path'))

x_data <- fread('input_HCI.csv')
# to save all results
result_table <- matrix(NA,nrow = 1, ncol = 53)
colnames(result_table) <- c('Model','TN','FP','FN','TP','ACC','BA','SN','SP','PPV','NPV','RC','F1','AUC_score',
	'TN_CV','FP_CV','FN_CV','TP_CV','ACC_CV','BA_CV','SN_CV','SP_CV','PPV_CV','NPV_CV','RC_CV','F1_CV','AUC_score_CV',
	'TN_TE','FP_TE','FN_TE','TP_TE','ACC_TE','BA_TE','SN_TE','SP_TE','PPV_TE','NPV_TE','RC_TE','F1_TE','AUC_score_TE',
	'TN_EXT','FP_EXT','FN_EXT','TP_EXT','ACC_EXT','BA_EXT','SN_EXT','SP_EXT','PPV_EXT','NPV_EXT','RC_EXT','F1_EXT','AUC_score_EXT')

set.seed(47)
# Data splitting in training and test set
x_train_orig <- stratified(x_data, c('CLASS'), 0.8)
x_test_orig <- setdiff(x_data,x_train_orig)

# save train and test set
new_folder <- paste(path,'/HCI_NoHS_NoFS',sep = '')
dir.create(new_folder)
setwd(new_folder)

write.table(x_train_orig,file = 'Orig_train_set.csv', row.names = F, col.names = T, sep = ',')
write.table(x_test_orig,file = 'Orig_test_set.csv', row.names = F, col.names = T, sep = ',')

# pre-processing train set 
x_train <- as.data.frame(x_train_orig[,!c('Sample_id', 'CLASS')])
# save the MEAN & STD. of x_train, later will be used for normalization of test set
mean_train <- apply(x_train, 2, mean, na.rm = TRUE)
std_train <- apply(x_train, 2, sd, na.rm = TRUE)
write.table(mean_train,file = 'Mean_train_set.csv', row.names = F, col.names = T, sep = ',')
write.table(std_train,file = 'STD_train_set.csv', row.names = F, col.names = T, sep = ',')

# center
x_train <- scale(x_train, center = T, scale = F)
# scale
x_train <- scale(x_train, center = FALSE, scale = apply(x_train, 2, sd, na.rm = TRUE))

# pre-processing test set
x_test <- as.data.frame(x_test_orig[,!c('Sample_id', 'CLASS')])
# scale and center using mean and std of train set
x_test <- x_test[,names(mean_train)]        # Arrange columns as same as 'mean_train' dataframe
x_test <- data.table(t(apply(x_test, 1, function(x) x - mean_train))) # centering
x_test <- data.table(t(apply(x_test, 1, function(x) x / std_train))) # scaling

y_train_binary <- x_train_orig$CLASS
y_test_binary <- x_test_orig$CLASS

# save normalized train and test set
write.table(cbind(x_train_orig[,1:2],x_train),file = 'train_set_norm.csv', row.names = F, col.names = T, sep = ',')
write.table(cbind(x_test_orig[,1:2],x_test),file = 'test_set_norm.csv', row.names = F, col.names = T, sep = ',')

x_train_1 <- as.data.frame(cbind(y_train_binary,x_train))
x_train_1$y_train_binary <- as.factor(x_train_1$y_train_binary)
levels(x_train_1$y_train_binary) <- make.names(levels(factor(x_train_1$y_train_binary)))

# RF model building
model_weights <- ifelse(y_train_binary=='0', (1/table(y_train_binary)[1]) * 0.5, (1/table(y_train_binary)[2]) * 0.5)
seeds <- as.vector(c(1:6), mode = "list")
seeds[[6]] <- 1
mtry_val = as.integer(sqrt(dim(x_train_1)[2]))

set.seed(47)

library(doMC)
registerDoMC(cores = 24)

weighted_fit <- caret::train(y_train_binary ~ .,
	data = x_train_1,
	method = "rf",
	importance = TRUE,
	verbose = FALSE,
	ntree=100,
	replace=FALSE, 
	nodesize = 5,
	metric = "Accuracy",
	tuneGrid = data.frame(mtry = mtry_val),
	tuneLength = 1,
	weights = model_weights,
	trControl = trainControl(method = "cv", number = 5, classProbs=T, savePredictions = "final", seeds = seeds))

# training pred
pred_tr = predict(weighted_fit, x_train_1[,-1])
tr_res <- confusionMatrix(data = pred_tr, reference = x_train_1$y_train_binary, positive = 'X1')

i <- 1
result_table[i,'Model'] <- 'HCI_NoHS_NoFS'
result_table[i,'TN'] <- as.numeric(tr_res$table[1])
result_table[i,'FP'] <- as.numeric(tr_res$table[2])
result_table[i,'FN'] <- as.numeric(tr_res$table[3])
result_table[i,'TP'] <- as.numeric(tr_res$table[4])
result_table[i,'ACC'] <- as.numeric(tr_res$overall['Accuracy'])
result_table[i,'BA'] <- as.numeric(tr_res$byClass['Balanced Accuracy'])
result_table[i,'SN'] <- as.numeric(tr_res$byClass['Sensitivity'])
result_table[i,'SP'] <- as.numeric(tr_res$byClass['Specificity'])
result_table[i,'PPV'] <- as.numeric(tr_res$byClass['Pos Pred Value'])
result_table[i,'NPV'] <- as.numeric(tr_res$byClass['Neg Pred Value'])
result_table[i,'RC'] <- as.numeric(sensitivity(pred_tr, x_train_1$y_train_binary, positive="X1"))
result_table[i,'F1'] <- as.numeric((2 * posPredValue(pred_tr, x_train_1$y_train_binary, positive="X1") * sensitivity(pred_tr, x_train_1$y_train_binary, positive="X1")) / (posPredValue(pred_tr, x_train_1$y_train_binary, positive="X1") + sensitivity(pred_tr, x_train_1$y_train_binary, positive="X1")))
tr_prob <- predict(weighted_fit, newdata=x_train_1, type="prob")
roc_obj <- roc(y_train_binary, tr_prob[[2]])
result_table[i,'AUC_score'] <- as.numeric(auc(roc_obj))

# CV output extraction
all_folds <- weighted_fit$pred
CV_results <- matrix(NA, nrow = 5, ncol = 14)
colnames(CV_results) <- c('Fold','TN','FP','FN','TP','ACC','BA','SN','SP','PPV','NPV','RC','F1','AUC_score')
for(j in 1:5){
	A <- paste('Fold',j,sep = '')
	B <- all_folds[all_folds$Resample == A,]
	CV_results[j,'Fold'] <- j
	CV_results[j,'TN'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[1])
	CV_results[j,'FP'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[2])
	CV_results[j,'FN'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[3])
	CV_results[j,'TP'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[4])
	CV_results[j,'ACC'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$overall['Accuracy'])
	CV_results[j,'BA'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Balanced Accuracy'])
	CV_results[j,'SN'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Sensitivity'])
	CV_results[j,'SP'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Specificity'])
	CV_results[j,'PPV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Pos Pred Value'])
	CV_results[j,'NPV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Neg Pred Value'])
	CV_results[j,'RC'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Recall'])
	CV_results[j,'F1'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['F1'])
	CV_results[j,'AUC_score'] <- as.numeric(auc(roc(B$obs, B$X1)))
}

CV_results_final <- colMeans(CV_results[,2:14], na.rm=TRUE)
CV_results_final[1:4] <- round(CV_results_final[1:4], 0)
CV_results_final[5:13] <- round(CV_results_final[5:13], 2)
# add to final result
result_table[i,15:27] <- CV_results_final

# test pred
pred_te = predict(weighted_fit, x_test)
y_test_binary_mod <- as.factor(y_test_binary)
levels(y_test_binary_mod) <- make.names(levels(factor(y_test_binary_mod)))
te_res <- confusionMatrix(data = pred_te, reference = y_test_binary_mod, positive = 'X1')

result_table[i,'TN_TE'] <- as.numeric(te_res$table[1])
result_table[i,'FP_TE'] <- as.numeric(te_res$table[2])
result_table[i,'FN_TE'] <- as.numeric(te_res$table[3])
result_table[i,'TP_TE'] <- as.numeric(te_res$table[4])
result_table[i,'ACC_TE'] <- round(as.numeric(te_res$overall['Accuracy']),2)
result_table[i,'BA_TE'] <- round(as.numeric(te_res$byClass['Balanced Accuracy']),2)
result_table[i,'SN_TE'] <- round(as.numeric(te_res$byClass['Sensitivity']),2)
result_table[i,'SP_TE'] <- round(as.numeric(te_res$byClass['Specificity']),2)
result_table[i,'PPV_TE'] <- round(as.numeric(te_res$byClass['Pos Pred Value']),2)
result_table[i,'NPV_TE'] <- round(as.numeric(te_res$byClass['Neg Pred Value']),2)
result_table[i,'RC_TE'] <- round(as.numeric(sensitivity(pred_te, y_test_binary_mod, positive="X1")),2)
result_table[i,'F1_TE'] <- round(as.numeric((2 * posPredValue(pred_te, y_test_binary_mod, positive="X1") * sensitivity(pred_te, y_test_binary_mod, positive="X1")) / (posPredValue(pred_te, y_test_binary_mod, positive="X1") + sensitivity(pred_te, y_test_binary_mod, positive="X1"))),2)
te_prob <- predict(weighted_fit, newdata=x_test, type="prob")
result_table[i,'AUC_score_TE'] <- round(as.numeric(auc(roc(y_test_binary, te_prob[[2]]))),2)

# external test set prediction

ext_test_set <- fread('avg_hybrid_descr_independent_test_set.csv')
# class
ext_test_set_class <- ifelse(ext_test_set$TOXICITY == "Hepatotoxic", 1, 0)

# replace '-' by underscore
colnames(ext_test_set) <- gsub('\\-','_',colnames(ext_test_set))

# extract descr 
ext_test_set_descr <- ext_test_set[,names(mean_train), with=F]
# standardize them using mean_train and std_train
ext_test_set_descr <- data.table(t(apply(ext_test_set_descr, 1, function(x) x - mean_train))) # centering
ext_test_set_descr <- data.table(t(apply(ext_test_set_descr, 1, function(x) x / std_train))) # scaling

# external predictions
pred_ext = predict(weighted_fit, ext_test_set_descr)
y_test_ext_binary_mod <- as.factor(ext_test_set_class)
levels(y_test_ext_binary_mod) <- make.names(levels(factor(y_test_ext_binary_mod)))
ext_res <- confusionMatrix(data = pred_ext, reference = y_test_ext_binary_mod, positive = 'X1')

result_table[i,'TN_EXT'] <- as.numeric(ext_res$table[1])
result_table[i,'FP_EXT'] <- as.numeric(ext_res$table[2])
result_table[i,'FN_EXT'] <- as.numeric(ext_res$table[3])
result_table[i,'TP_EXT'] <- as.numeric(ext_res$table[4])
result_table[i,'ACC_EXT'] <- round(as.numeric(ext_res$overall['Accuracy']),2)
result_table[i,'BA_EXT'] <- round(as.numeric(ext_res$byClass['Balanced Accuracy']),2)
result_table[i,'SN_EXT'] <- round(as.numeric(ext_res$byClass['Sensitivity']),2)
result_table[i,'SP_EXT'] <- round(as.numeric(ext_res$byClass['Specificity']),2)
result_table[i,'PPV_EXT'] <- round(as.numeric(ext_res$byClass['Pos Pred Value']),2)
result_table[i,'NPV_EXT'] <- round(as.numeric(ext_res$byClass['Neg Pred Value']),2)
result_table[i,'RC_EXT'] <- round(as.numeric(sensitivity(pred_ext, y_test_ext_binary_mod, positive="X1")),2)
result_table[i,'F1_EXT'] <- round(as.numeric((2 * posPredValue(pred_ext, y_test_ext_binary_mod, positive="X1") * sensitivity(pred_ext, y_test_ext_binary_mod, positive="X1")) / (posPredValue(pred_ext, y_test_ext_binary_mod, positive="X1") + sensitivity(pred_ext, y_test_ext_binary_mod, positive="X1"))),2)
ext_prob <- predict(weighted_fit, newdata=ext_test_set_descr, type="prob")
result_table[i,'AUC_score_EXT'] <- round(as.numeric(auc(roc(ext_test_set_class, ext_prob[[2]]))),2)
write.table(result_table,file = 'Results_HCI_NoHS_NoFS.csv', row.names = F, col.names = T, sep = ',')

# Var importance
ImpMeasure <- varImp(weighted_fit, scale = FALSE)

library(randomForest)
fm <- weighted_fit$finalModel
var_imp_details <- data.frame(randomForest::importance(fm, scale = FALSE))
var_imp_details$Vars <- row.names(var_imp_details)
var_imp_details <- var_imp_details[,c(5,1:4)]
var_imp_details <- var_imp_details[order(var_imp_details$MeanDecreaseGini, decreasing = T),]
write.table(var_imp_details,file = 'Imp_variables.csv', row.names = F, col.names = T, sep = ',')

# save model
mod_nm <- paste('HCI_descr_Model_NoHS_NoFS', '.Rdata',sep = '')
save.image(file=mod_nm)


# 2. Model building with Chemical descriptors

# Step - 1: Remove zero variance variables

rm(list = setdiff(ls(),'path'))
setwd(path)

avg_variables <- fread('avg_variables_346_inst_table.csv')

# keep only HCI phenotypes
hci_var_names <- names(avg_variables)[grep('Cells|Cytoplasm|Nuclei',names(avg_variables))]
x_data <- avg_variables[,-hci_var_names, with=F]
x_data <- x_data[,-c('Sample_id', 'TOXICITY', 'Replicate_count'), with=F]

# Remove descriptors with near zero variance
nzv <- nearZeroVar(x_data)
if (length(nzv) != 0){
	x_data <- x_data[,-nzv, with=F]
}
# 364 variable with zero or near zero variance
rm(nzv)

# add CLASS column
x_data[, Sample_id := avg_variables$Sample_id]
x_data[, CLASS := ifelse(avg_variables$TOXICITY == 'Hepatotoxic', 1, 0)]
x_data <- x_data[,c(982,983,1:981),with=F]

# replace '-' by underscore
colnames(x_data) <- gsub('\\-','_',colnames(x_data))

write.table(x_data, 'input_chemical.csv', row.names = F, col.names = T, sep = ',')

# Step - 2: RF model construction

rm(list = setdiff(ls(),'path'))

x_data <- fread('input_chemical.csv')
# to save all results
result_table <- matrix(NA,nrow = 1, ncol = 53)
colnames(result_table) <- c('Model','TN','FP','FN','TP','ACC','BA','SN','SP','PPV','NPV','RC','F1','AUC_score',
	'TN_CV','FP_CV','FN_CV','TP_CV','ACC_CV','BA_CV','SN_CV','SP_CV','PPV_CV','NPV_CV','RC_CV','F1_CV','AUC_score_CV',
	'TN_TE','FP_TE','FN_TE','TP_TE','ACC_TE','BA_TE','SN_TE','SP_TE','PPV_TE','NPV_TE','RC_TE','F1_TE','AUC_score_TE',
	'TN_EXT','FP_EXT','FN_EXT','TP_EXT','ACC_EXT','BA_EXT','SN_EXT','SP_EXT','PPV_EXT','NPV_EXT','RC_EXT','F1_EXT','AUC_score_EXT')


set.seed(47)
# Data splitting in training and test set
x_train_orig <- stratified(x_data, c('CLASS'), 0.8)
x_test_orig <- setdiff(x_data,x_train_orig)

# save train and test set
new_folder <- paste(path,'/chemical_NoHS_NoFS',sep = '')
dir.create(new_folder)
setwd(new_folder)
write.table(x_train_orig,file = 'Orig_train_set.csv', row.names = F, col.names = T, sep = ',')
write.table(x_test_orig,file = 'Orig_test_set.csv', row.names = F, col.names = T, sep = ',')

# pre-processing train set 
x_train <- as.data.frame(x_train_orig[,!c('Sample_id', 'CLASS')])
# save the MEAN & STD. of x_train, later will be used for normalization of test set
mean_train <- apply(x_train, 2, mean, na.rm = TRUE)
std_train <- apply(x_train, 2, sd, na.rm = TRUE)
write.table(mean_train,file = 'Mean_train_set.csv', row.names = F, col.names = T, sep = ',')
write.table(std_train,file = 'STD_train_set.csv', row.names = F, col.names = T, sep = ',')

# center
x_train <- scale(x_train, center = T, scale = F)
# scale
x_train <- scale(x_train, center = FALSE, scale = apply(x_train, 2, sd, na.rm = TRUE))

# pre-processing test set
x_test <- as.data.frame(x_test_orig[,!c('Sample_id', 'CLASS')])
# scale and center using mean and std of train set
x_test <- x_test[,names(mean_train)]        # Arrange columns as same as 'mean_train' dataframe
x_test <- data.table(t(apply(x_test, 1, function(x) x - mean_train))) # centering
x_test <- data.table(t(apply(x_test, 1, function(x) x / std_train))) # scaling

y_train_binary <- x_train_orig$CLASS
y_test_binary <- x_test_orig$CLASS

# save normalized train and test set
write.table(cbind(x_train_orig[,1:2],x_train),file = 'train_set_norm.csv', row.names = F, col.names = T, sep = ',')
write.table(cbind(x_test_orig[,1:2],x_test),file = 'test_set_norm.csv', row.names = F, col.names = T, sep = ',')

x_train_1 <- as.data.frame(cbind(y_train_binary,x_train))
x_train_1$y_train_binary <- as.factor(x_train_1$y_train_binary)
levels(x_train_1$y_train_binary) <- make.names(levels(factor(x_train_1$y_train_binary)))

# RF model building
model_weights <- ifelse(y_train_binary=='0', (1/table(y_train_binary)[1]) * 0.5, (1/table(y_train_binary)[2]) * 0.5)
seeds <- as.vector(c(1:6), mode = "list")
seeds[[6]] <- 1
mtry_val = as.integer(sqrt(dim(x_train_1)[2]))

set.seed(47)

library(doMC)
registerDoMC(cores = 24)

weighted_fit <- caret::train(y_train_binary ~ .,
	data = x_train_1,
	method = "rf",
	importance = TRUE,
	verbose = FALSE,
	ntree=100,
	replace=FALSE, 
	nodesize = 5,
	metric = "Accuracy",
	tuneGrid = data.frame(mtry = mtry_val),
	tuneLength = 1,
	weights = model_weights,
	trControl = trainControl(method = "cv", number = 5, classProbs=T, savePredictions = "final", seeds = seeds))

# training pred
pred_tr = predict(weighted_fit, x_train_1[,-1])
tr_res <- confusionMatrix(data = pred_tr, reference = x_train_1$y_train_binary, positive = 'X1')

i <- 1
result_table[i,'Model'] <- 'Chemical_NoHS_NoFS'
result_table[i,'TN'] <- as.numeric(tr_res$table[1])
result_table[i,'FP'] <- as.numeric(tr_res$table[2])
result_table[i,'FN'] <- as.numeric(tr_res$table[3])
result_table[i,'TP'] <- as.numeric(tr_res$table[4])
result_table[i,'ACC'] <- as.numeric(tr_res$overall['Accuracy'])
result_table[i,'BA'] <- as.numeric(tr_res$byClass['Balanced Accuracy'])
result_table[i,'SN'] <- as.numeric(tr_res$byClass['Sensitivity'])
result_table[i,'SP'] <- as.numeric(tr_res$byClass['Specificity'])
result_table[i,'PPV'] <- as.numeric(tr_res$byClass['Pos Pred Value'])
result_table[i,'NPV'] <- as.numeric(tr_res$byClass['Neg Pred Value'])
result_table[i,'RC'] <- as.numeric(sensitivity(pred_tr, x_train_1$y_train_binary, positive="X1"))
result_table[i,'F1'] <- as.numeric((2 * posPredValue(pred_tr, x_train_1$y_train_binary, positive="X1") * sensitivity(pred_tr, x_train_1$y_train_binary, positive="X1")) / (posPredValue(pred_tr, x_train_1$y_train_binary, positive="X1") + sensitivity(pred_tr, x_train_1$y_train_binary, positive="X1")))
tr_prob <- predict(weighted_fit, newdata=x_train_1, type="prob")
roc_obj <- roc(y_train_binary, tr_prob[[2]])
result_table[i,'AUC_score'] <- as.numeric(auc(roc_obj))

# CV output extraction
all_folds <- weighted_fit$pred
CV_results <- matrix(NA, nrow = 5, ncol = 14)
colnames(CV_results) <- c('Fold','TN','FP','FN','TP','ACC','BA','SN','SP','PPV','NPV','RC','F1','AUC_score')
for(j in 1:5){
	A <- paste('Fold',j,sep = '')
	B <- all_folds[all_folds$Resample == A,]
	CV_results[j,'Fold'] <- j
	CV_results[j,'TN'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[1])
	CV_results[j,'FP'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[2])
	CV_results[j,'FN'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[3])
	CV_results[j,'TP'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[4])
	CV_results[j,'ACC'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$overall['Accuracy'])
	CV_results[j,'BA'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Balanced Accuracy'])
	CV_results[j,'SN'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Sensitivity'])
	CV_results[j,'SP'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Specificity'])
	CV_results[j,'PPV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Pos Pred Value'])
	CV_results[j,'NPV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Neg Pred Value'])
	CV_results[j,'RC'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Recall'])
	CV_results[j,'F1'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['F1'])
	CV_results[j,'AUC_score'] <- as.numeric(auc(roc(B$obs, B$X1)))
}

CV_results_final <- colMeans(CV_results[,2:14], na.rm=TRUE)
CV_results_final[1:4] <- round(CV_results_final[1:4], 0)
CV_results_final[5:13] <- round(CV_results_final[5:13], 2)
# add to final result
result_table[i,15:27] <- CV_results_final

# test pred
pred_te = predict(weighted_fit, x_test)
y_test_binary_mod <- as.factor(y_test_binary)
levels(y_test_binary_mod) <- make.names(levels(factor(y_test_binary_mod)))
te_res <- confusionMatrix(data = pred_te, reference = y_test_binary_mod, positive = 'X1')

result_table[i,'TN_TE'] <- as.numeric(te_res$table[1])
result_table[i,'FP_TE'] <- as.numeric(te_res$table[2])
result_table[i,'FN_TE'] <- as.numeric(te_res$table[3])
result_table[i,'TP_TE'] <- as.numeric(te_res$table[4])
result_table[i,'ACC_TE'] <- round(as.numeric(te_res$overall['Accuracy']),2)
result_table[i,'BA_TE'] <- round(as.numeric(te_res$byClass['Balanced Accuracy']),2)
result_table[i,'SN_TE'] <- round(as.numeric(te_res$byClass['Sensitivity']),2)
result_table[i,'SP_TE'] <- round(as.numeric(te_res$byClass['Specificity']),2)
result_table[i,'PPV_TE'] <- round(as.numeric(te_res$byClass['Pos Pred Value']),2)
result_table[i,'NPV_TE'] <- round(as.numeric(te_res$byClass['Neg Pred Value']),2)
result_table[i,'RC_TE'] <- round(as.numeric(sensitivity(pred_te, y_test_binary_mod, positive="X1")),2)
result_table[i,'F1_TE'] <- round(as.numeric((2 * posPredValue(pred_te, y_test_binary_mod, positive="X1") * sensitivity(pred_te, y_test_binary_mod, positive="X1")) / (posPredValue(pred_te, y_test_binary_mod, positive="X1") + sensitivity(pred_te, y_test_binary_mod, positive="X1"))),2)
te_prob <- predict(weighted_fit, newdata=x_test, type="prob")
result_table[i,'AUC_score_TE'] <- round(as.numeric(auc(roc(y_test_binary, te_prob[[2]]))),2)

# external test set prediction

ext_test_set <- fread('avg_hybrid_descr_independent_test_set.csv')
# class
ext_test_set_class <- ifelse(ext_test_set$TOXICITY == "Hepatotoxic", 1, 0)

# replace '-' by underscore
colnames(ext_test_set) <- gsub('\\-','_',colnames(ext_test_set))

# extract descr 
ext_test_set_descr <- ext_test_set[,names(mean_train), with=F]
# standardize them using mean_train and std_train
ext_test_set_descr <- data.table(t(apply(ext_test_set_descr, 1, function(x) x - mean_train))) # centering
ext_test_set_descr <- data.table(t(apply(ext_test_set_descr, 1, function(x) x / std_train))) # scaling

# external predictions
pred_ext = predict(weighted_fit, ext_test_set_descr)
y_test_ext_binary_mod <- as.factor(ext_test_set_class)
levels(y_test_ext_binary_mod) <- make.names(levels(factor(y_test_ext_binary_mod)))
ext_res <- confusionMatrix(data = pred_ext, reference = y_test_ext_binary_mod, positive = 'X1')

result_table[i,'TN_EXT'] <- as.numeric(ext_res$table[1])
result_table[i,'FP_EXT'] <- as.numeric(ext_res$table[2])
result_table[i,'FN_EXT'] <- as.numeric(ext_res$table[3])
result_table[i,'TP_EXT'] <- as.numeric(ext_res$table[4])
result_table[i,'ACC_EXT'] <- round(as.numeric(ext_res$overall['Accuracy']),2)
result_table[i,'BA_EXT'] <- round(as.numeric(ext_res$byClass['Balanced Accuracy']),2)
result_table[i,'SN_EXT'] <- round(as.numeric(ext_res$byClass['Sensitivity']),2)
result_table[i,'SP_EXT'] <- round(as.numeric(ext_res$byClass['Specificity']),2)
result_table[i,'PPV_EXT'] <- round(as.numeric(ext_res$byClass['Pos Pred Value']),2)
result_table[i,'NPV_EXT'] <- round(as.numeric(ext_res$byClass['Neg Pred Value']),2)
result_table[i,'RC_EXT'] <- round(as.numeric(sensitivity(pred_ext, y_test_ext_binary_mod, positive="X1")),2)
result_table[i,'F1_EXT'] <- round(as.numeric((2 * posPredValue(pred_ext, y_test_ext_binary_mod, positive="X1") * sensitivity(pred_ext, y_test_ext_binary_mod, positive="X1")) / (posPredValue(pred_ext, y_test_ext_binary_mod, positive="X1") + sensitivity(pred_ext, y_test_ext_binary_mod, positive="X1"))),2)
ext_prob <- predict(weighted_fit, newdata=ext_test_set_descr, type="prob")
result_table[i,'AUC_score_EXT'] <- round(as.numeric(auc(roc(ext_test_set_class, ext_prob[[2]]))),2)
write.table(result_table,file = 'Results_chemical_NoHS_NoFS.csv', row.names = F, col.names = T, sep = ',')

# Var importance
ImpMeasure <- varImp(weighted_fit, scale = FALSE)

library(randomForest)
fm <- weighted_fit$finalModel
var_imp_details <- data.frame(randomForest::importance(fm, scale = FALSE))
var_imp_details$Vars <- row.names(var_imp_details)
var_imp_details <- var_imp_details[,c(5,1:4)]
var_imp_details <- var_imp_details[order(var_imp_details$MeanDecreaseGini, decreasing = T),]
write.table(var_imp_details,file = 'Imp_variables.csv', row.names = F, col.names = T, sep = ',')


# save model
mod_nm <- paste('chemical_NoHS_NoFS','.Rdata',sep = '')
save.image(file=mod_nm)


# 3.  Model building using hybrid variables

# Step - 1: Remove zero variance variables

rm(list = setdiff(ls(),'path'))
setwd(path)

avg_variables <- fread('avg_variables_346_inst_table.csv')

# drop first 3 cols
x_data <- avg_variables[,-c('Sample_id', 'TOXICITY', 'Replicate_count'), with=F]

# Remove descriptors with near zero variance
nzv <- nearZeroVar(x_data)
# 364 var with zero/mear zero variance
if (length(nzv) != 0){
	x_data <- x_data[,-nzv, with=F]
}
rm(nzv)

# add CLASS column
x_data[, Sample_id := avg_variables$Sample_id]
x_data[, CLASS := ifelse(avg_variables$TOXICITY == 'Hepatotoxic', 1, 0)]
x_data <- x_data[,c(2702,2703,1:2701),with=F]

# replace '-' by underscore
colnames(x_data) <- gsub('\\-','_',colnames(x_data))

write.table(x_data, 'input_hybrid.csv', row.names = F, col.names = T, sep = ',')

# Step - 2: RF model construction using "caret" package

rm(list = setdiff(ls(),'path'))

x_data <- fread('input_hybrid.csv')
# to save all results
result_table <- matrix(NA,nrow = 1, ncol = 53)
colnames(result_table) <- c('Model','TN','FP','FN','TP','ACC','BA','SN','SP','PPV','NPV','RC','F1','AUC_score',
	'TN_CV','FP_CV','FN_CV','TP_CV','ACC_CV','BA_CV','SN_CV','SP_CV','PPV_CV','NPV_CV','RC_CV','F1_CV','AUC_score_CV',
	'TN_TE','FP_TE','FN_TE','TP_TE','ACC_TE','BA_TE','SN_TE','SP_TE','PPV_TE','NPV_TE','RC_TE','F1_TE','AUC_score_TE',
	'TN_EXT','FP_EXT','FN_EXT','TP_EXT','ACC_EXT','BA_EXT','SN_EXT','SP_EXT','PPV_EXT','NPV_EXT','RC_EXT','F1_EXT','AUC_score_EXT')


set.seed(47)
# Data splitting in training and test set
x_train_orig <- stratified(x_data, c('CLASS'), 0.8)
x_test_orig <- setdiff(x_data,x_train_orig)

# save train and test set

new_folder <- paste(path,'/Hybrid_NoHS_NoFS',sep = '')
dir.create(new_folder)
setwd(new_folder)
write.table(x_train_orig,file = 'Orig_train_set.csv', row.names = F, col.names = T, sep = ',')
write.table(x_test_orig,file = 'Orig_test_set.csv', row.names = F, col.names = T, sep = ',')

# pre-processing train set 
x_train <- as.data.frame(x_train_orig[,!c('Sample_id', 'CLASS')])
# save the MEAN & STD. of x_train, later will be used for normalization of test set
mean_train <- apply(x_train, 2, mean, na.rm = TRUE)
std_train <- apply(x_train, 2, sd, na.rm = TRUE)
write.table(mean_train,file = 'Mean_train_set.csv', row.names = F, col.names = T, sep = ',')
write.table(std_train,file = 'STD_train_set.csv', row.names = F, col.names = T, sep = ',')

# center
x_train <- scale(x_train, center = T, scale = F)
# scale
x_train <- scale(x_train, center = FALSE, scale = apply(x_train, 2, sd, na.rm = TRUE))

# pre-processing test set
x_test <- as.data.frame(x_test_orig[,!c('Sample_id', 'CLASS')])
# scale and center using mean and std of train set
x_test <- x_test[,names(mean_train)]        # Arrange columns as same as 'mean_train' dataframe
x_test <- data.table(t(apply(x_test, 1, function(x) x - mean_train))) # centering
x_test <- data.table(t(apply(x_test, 1, function(x) x / std_train))) # scaling

y_train_binary <- x_train_orig$CLASS
y_test_binary <- x_test_orig$CLASS

# save normalized train and test set
write.table(cbind(x_train_orig[,1:2],x_train),file = 'train_set_norm.csv', row.names = F, col.names = T, sep = ',')
write.table(cbind(x_test_orig[,1:2],x_test),file = 'test_set_norm.csv', row.names = F, col.names = T, sep = ',')

x_train_1 <- as.data.frame(cbind(y_train_binary,x_train))
x_train_1$y_train_binary <- as.factor(x_train_1$y_train_binary)
levels(x_train_1$y_train_binary) <- make.names(levels(factor(x_train_1$y_train_binary)))

# RF model building
model_weights <- ifelse(y_train_binary=='0', (1/table(y_train_binary)[1]) * 0.5, (1/table(y_train_binary)[2]) * 0.5)
seeds <- as.vector(c(1:6), mode = "list")
seeds[[6]] <- 1
mtry_val = as.integer(sqrt(dim(x_train_1)[2]))

set.seed(47)

library(doMC)
registerDoMC(cores = 16)

weighted_fit <- caret::train(y_train_binary ~ .,
	data = x_train_1,
	method = "rf",
	importance = TRUE,
	verbose = FALSE,
	ntree=100,
	replace=FALSE, 
	nodesize = 5,
	metric = "Accuracy",
	tuneGrid = data.frame(mtry = mtry_val),
	tuneLength = 1,
	weights = model_weights,
	trControl = trainControl(method = "cv", number = 5, classProbs=T, savePredictions = "final", seeds = seeds))

# training pred
pred_tr = predict(weighted_fit, x_train_1[,-1])
tr_res <- confusionMatrix(data = pred_tr, reference = x_train_1$y_train_binary, positive = 'X1')

i <- 1
result_table[i,'Model'] <- 'Hybrid_NoHS_NoFS'
result_table[i,'TN'] <- as.numeric(tr_res$table[1])
result_table[i,'FP'] <- as.numeric(tr_res$table[2])
result_table[i,'FN'] <- as.numeric(tr_res$table[3])
result_table[i,'TP'] <- as.numeric(tr_res$table[4])
result_table[i,'ACC'] <- as.numeric(tr_res$overall['Accuracy'])
result_table[i,'BA'] <- as.numeric(tr_res$byClass['Balanced Accuracy'])
result_table[i,'SN'] <- as.numeric(tr_res$byClass['Sensitivity'])
result_table[i,'SP'] <- as.numeric(tr_res$byClass['Specificity'])
result_table[i,'PPV'] <- as.numeric(tr_res$byClass['Pos Pred Value'])
result_table[i,'NPV'] <- as.numeric(tr_res$byClass['Neg Pred Value'])
result_table[i,'RC'] <- as.numeric(sensitivity(pred_tr, x_train_1$y_train_binary, positive="X1"))
result_table[i,'F1'] <- as.numeric((2 * posPredValue(pred_tr, x_train_1$y_train_binary, positive="X1") * sensitivity(pred_tr, x_train_1$y_train_binary, positive="X1")) / (posPredValue(pred_tr, x_train_1$y_train_binary, positive="X1") + sensitivity(pred_tr, x_train_1$y_train_binary, positive="X1")))
tr_prob <- predict(weighted_fit, newdata=x_train_1, type="prob")
roc_obj <- roc(y_train_binary, tr_prob[[2]])
result_table[i,'AUC_score'] <- as.numeric(auc(roc_obj))

# CV output extraction
all_folds <- weighted_fit$pred
CV_results <- matrix(NA, nrow = 5, ncol = 14)
colnames(CV_results) <- c('Fold','TN','FP','FN','TP','ACC','BA','SN','SP','PPV','NPV','RC','F1','AUC_score')
for(j in 1:5){
	A <- paste('Fold',j,sep = '')
	B <- all_folds[all_folds$Resample == A,]
	CV_results[j,'Fold'] <- j
	CV_results[j,'TN'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[1])
	CV_results[j,'FP'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[2])
	CV_results[j,'FN'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[3])
	CV_results[j,'TP'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[4])
	CV_results[j,'ACC'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$overall['Accuracy'])
	CV_results[j,'BA'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Balanced Accuracy'])
	CV_results[j,'SN'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Sensitivity'])
	CV_results[j,'SP'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Specificity'])
	CV_results[j,'PPV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Pos Pred Value'])
	CV_results[j,'NPV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Neg Pred Value'])
	CV_results[j,'RC'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Recall'])
	CV_results[j,'F1'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['F1'])
	CV_results[j,'AUC_score'] <- as.numeric(auc(roc(B$obs, B$X1)))
}

CV_results_final <- colMeans(CV_results[,2:14], na.rm=TRUE)
CV_results_final[1:4] <- round(CV_results_final[1:4], 0)
CV_results_final[5:13] <- round(CV_results_final[5:13], 2)
# add to final result
result_table[i,15:27] <- CV_results_final

# test pred
pred_te = predict(weighted_fit, x_test)
y_test_binary_mod <- as.factor(y_test_binary)
levels(y_test_binary_mod) <- make.names(levels(factor(y_test_binary_mod)))
te_res <- confusionMatrix(data = pred_te, reference = y_test_binary_mod, positive = 'X1')

result_table[i,'TN_TE'] <- as.numeric(te_res$table[1])
result_table[i,'FP_TE'] <- as.numeric(te_res$table[2])
result_table[i,'FN_TE'] <- as.numeric(te_res$table[3])
result_table[i,'TP_TE'] <- as.numeric(te_res$table[4])
result_table[i,'ACC_TE'] <- round(as.numeric(te_res$overall['Accuracy']),2)
result_table[i,'BA_TE'] <- round(as.numeric(te_res$byClass['Balanced Accuracy']),2)
result_table[i,'SN_TE'] <- round(as.numeric(te_res$byClass['Sensitivity']),2)
result_table[i,'SP_TE'] <- round(as.numeric(te_res$byClass['Specificity']),2)
result_table[i,'PPV_TE'] <- round(as.numeric(te_res$byClass['Pos Pred Value']),2)
result_table[i,'NPV_TE'] <- round(as.numeric(te_res$byClass['Neg Pred Value']),2)
result_table[i,'RC_TE'] <- round(as.numeric(sensitivity(pred_te, y_test_binary_mod, positive="X1")),2)
result_table[i,'F1_TE'] <- round(as.numeric((2 * posPredValue(pred_te, y_test_binary_mod, positive="X1") * sensitivity(pred_te, y_test_binary_mod, positive="X1")) / (posPredValue(pred_te, y_test_binary_mod, positive="X1") + sensitivity(pred_te, y_test_binary_mod, positive="X1"))),2)
te_prob <- predict(weighted_fit, newdata=x_test, type="prob")
result_table[i,'AUC_score_TE'] <- round(as.numeric(auc(roc(y_test_binary, te_prob[[2]]))),2)

# external test set prediction

ext_test_set <- fread('avg_hybrid_descr_independent_test_set.csv')
# class
ext_test_set_class <- ifelse(ext_test_set$TOXICITY == "Hepatotoxic", 1, 0)

# replace '-' by underscore
colnames(ext_test_set) <- gsub('\\-','_',colnames(ext_test_set))

# extract descr 
ext_test_set_descr <- ext_test_set[,names(mean_train), with=F]
# standardize them using mean_train and std_train
ext_test_set_descr <- data.table(t(apply(ext_test_set_descr, 1, function(x) x - mean_train))) # centering
ext_test_set_descr <- data.table(t(apply(ext_test_set_descr, 1, function(x) x / std_train))) # scaling

# external predictions
pred_ext = predict(weighted_fit, ext_test_set_descr)
y_test_ext_binary_mod <- as.factor(ext_test_set_class)
levels(y_test_ext_binary_mod) <- make.names(levels(factor(y_test_ext_binary_mod)))
ext_res <- confusionMatrix(data = pred_ext, reference = y_test_ext_binary_mod, positive = 'X1')

result_table[i,'TN_EXT'] <- as.numeric(ext_res$table[1])
result_table[i,'FP_EXT'] <- as.numeric(ext_res$table[2])
result_table[i,'FN_EXT'] <- as.numeric(ext_res$table[3])
result_table[i,'TP_EXT'] <- as.numeric(ext_res$table[4])
result_table[i,'ACC_EXT'] <- round(as.numeric(ext_res$overall['Accuracy']),2)
result_table[i,'BA_EXT'] <- round(as.numeric(ext_res$byClass['Balanced Accuracy']),2)
result_table[i,'SN_EXT'] <- round(as.numeric(ext_res$byClass['Sensitivity']),2)
result_table[i,'SP_EXT'] <- round(as.numeric(ext_res$byClass['Specificity']),2)
result_table[i,'PPV_EXT'] <- round(as.numeric(ext_res$byClass['Pos Pred Value']),2)
result_table[i,'NPV_EXT'] <- round(as.numeric(ext_res$byClass['Neg Pred Value']),2)
result_table[i,'RC_EXT'] <- round(as.numeric(sensitivity(pred_ext, y_test_ext_binary_mod, positive="X1")),2)
result_table[i,'F1_EXT'] <- round(as.numeric((2 * posPredValue(pred_ext, y_test_ext_binary_mod, positive="X1") * sensitivity(pred_ext, y_test_ext_binary_mod, positive="X1")) / (posPredValue(pred_ext, y_test_ext_binary_mod, positive="X1") + sensitivity(pred_ext, y_test_ext_binary_mod, positive="X1"))),2)
ext_prob <- predict(weighted_fit, newdata=ext_test_set_descr, type="prob")
result_table[i,'AUC_score_EXT'] <- round(as.numeric(auc(roc(ext_test_set_class, ext_prob[[2]]))),2)
write.table(result_table,file = 'Results_hybrid_NoHS_NoFS.csv', row.names = F, col.names = T, sep = ',')

# Var importance
ImpMeasure <- varImp(weighted_fit, scale = FALSE)

library(randomForest)
fm <- weighted_fit$finalModel
var_imp_details <- data.frame(randomForest::importance(fm, scale = FALSE))
var_imp_details$Vars <- row.names(var_imp_details)
var_imp_details <- var_imp_details[,c(5,1:4)]
var_imp_details <- var_imp_details[order(var_imp_details$MeanDecreaseGini, decreasing = T),]
write.table(var_imp_details,file = 'Imp_variables.csv', row.names = F, col.names = T, sep = ',')

# save model
mod_nm <- paste('hybrid_NoHS_NoFS','.Rdata',sep = '')
save.image(file=mod_nm)


######################################################################################################
#             Approach - B:  Hyperparameter Selection & No Feature Selection (HS_NoFS)
######################################################################################################


# 1.   Model building using HCI variables

rm(list = setdiff(ls(),'path'))
setwd(path)

x_data <- fread('input_HCI.csv')
# to save all results
result_table <- matrix(NA,nrow = 1, ncol = 56)
colnames(result_table) <- c('Model','mtry','ntree','nodesize','TN','FP','FN','TP','ACC','BA','SN','SP','PPV','NPV','RC','F1','AUC_score',
	'TN_CV','FP_CV','FN_CV','TP_CV','ACC_CV','BA_CV','SN_CV','SP_CV','PPV_CV','NPV_CV','RC_CV','F1_CV','AUC_score_CV',
	'TN_TE','FP_TE','FN_TE','TP_TE','ACC_TE','BA_TE','SN_TE','SP_TE','PPV_TE','NPV_TE','RC_TE','F1_TE','AUC_score_TE',
	'TN_EXT','FP_EXT','FN_EXT','TP_EXT','ACC_EXT','BA_EXT','SN_EXT','SP_EXT','PPV_EXT','NPV_EXT','RC_EXT','F1_EXT','AUC_score_EXT')

set.seed(47)
# Data splitting in training and test set
x_train_orig <- stratified(x_data, c('CLASS'), 0.8)
x_test_orig <- setdiff(x_data,x_train_orig)

# save train and test set
new_folder <- paste(path,'/HCI_HS_NoFS',sep = '')
dir.create(new_folder)
setwd(new_folder)
write.table(x_train_orig,file = 'Orig_train_set.csv', row.names = F, col.names = T, sep = ',')
write.table(x_test_orig,file = 'Orig_test_set.csv', row.names = F, col.names = T, sep = ',')

# pre-processing train set 
x_train <- as.data.frame(x_train_orig[,!c('Sample_id', 'CLASS')])
# save the MEAN & STD. of x_train, later will be used for normalization of test set
mean_train <- apply(x_train, 2, mean, na.rm = TRUE)
std_train <- apply(x_train, 2, sd, na.rm = TRUE)
write.table(mean_train,file = 'Mean_train_set.csv', row.names = F, col.names = T, sep = ',')
write.table(std_train,file = 'STD_train_set.csv', row.names = F, col.names = T, sep = ',')

# center
x_train <- scale(x_train, center = T, scale = F)
# scale
x_train <- scale(x_train, center = FALSE, scale = apply(x_train, 2, sd, na.rm = TRUE))

# pre-processing test set
x_test <- as.data.frame(x_test_orig[,!c('Sample_id', 'CLASS')])
# scale and center using mean and std of train set
x_test <- x_test[,names(mean_train)]        # Arrange columns as same as 'mean_train' dataframe
x_test <- data.table(t(apply(x_test, 1, function(x) x - mean_train))) # centering
x_test <- data.table(t(apply(x_test, 1, function(x) x / std_train))) # scaling

y_train_binary <- x_train_orig$CLASS
y_test_binary <- x_test_orig$CLASS

# save normalized train and test set
write.table(cbind(x_train_orig[,1:2],x_train),file = 'train_set_norm.csv', row.names = F, col.names = T, sep = ',')
write.table(cbind(x_test_orig[,1:2],x_test),file = 'test_set_norm.csv', row.names = F, col.names = T, sep = ',')

x_train_1 <- as.data.frame(cbind(y_train_binary,x_train))
x_train_1$y_train_binary <- as.factor(x_train_1$y_train_binary)
levels(x_train_1$y_train_binary) <- make.names(levels(factor(x_train_1$y_train_binary)))

# RF model building
model_weights <- ifelse(y_train_binary=='0', (1/table(y_train_binary)[1]) * 0.5, (1/table(y_train_binary)[2]) * 0.5)

set.seed(47)

customRF <- list(type = "Classification", library = "randomForest", loop = NULL)
customRF$parameters <- data.frame(parameter = c("mtry", "ntree", "nodesize"), class = rep("numeric", 3), label = c("mtry", "ntree", "nodesize"))
customRF$grid <- function(x, y, len = NULL, search = "grid") {}
customRF$fit <- function(x, y, wts, param, lev, last, weights, classProbs, ...) {
	randomForest(x, y, mtry = param$mtry, ntree=param$ntree, ...)
}
customRF$predict <- function(modelFit, newdata, preProc = NULL, submodels = NULL) 
	predict(modelFit, newdata)
customRF$prob <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
	predict(modelFit, newdata, type = "prob")
customRF$sort <- function(x) x[order(x[,1]),]
customRF$levels <- function(x) x$classes

library(doMC)
registerDoMC(cores = 32)

tunegrid <- data.frame(expand.grid(.mtry=c(30,35,40,45,50), .ntree=c(50, 100,500), .nodesize=c(1:5)))
weighted_fit <- caret::train(y_train_binary ~ .,
	data = x_train_1,
	method = customRF,
	importance = TRUE,
	verbose = FALSE,
	replace=FALSE,
	metric = "Accuracy",
	tuneGrid = tunegrid,
	weights = model_weights,
	trControl = trainControl(method = "cv", number = 5, classProbs=T, savePredictions = "final", allowParallel = TRUE))

# save results
i <- 1
result_table[i,'Model'] <- 'HCI_HS_NoFS'
result_table[i,'mtry'] <- weighted_fit$bestTune$mtry
result_table[i,'ntree'] <- weighted_fit$bestTune$ntree
result_table[i,'nodesize'] <- weighted_fit$bestTune$nodesize

# training pred
pred_tr = predict(weighted_fit, x_train_1[,-1])
tr_res <- confusionMatrix(data = pred_tr, reference = x_train_1$y_train_binary, positive = 'X1')

result_table[i,'TN'] <- as.numeric(tr_res$table[1])
result_table[i,'FP'] <- as.numeric(tr_res$table[2])
result_table[i,'FN'] <- as.numeric(tr_res$table[3])
result_table[i,'TP'] <- as.numeric(tr_res$table[4])
result_table[i,'ACC'] <- as.numeric(tr_res$overall['Accuracy'])
result_table[i,'BA'] <- as.numeric(tr_res$byClass['Balanced Accuracy'])
result_table[i,'SN'] <- as.numeric(tr_res$byClass['Sensitivity'])
result_table[i,'SP'] <- as.numeric(tr_res$byClass['Specificity'])
result_table[i,'PPV'] <- as.numeric(tr_res$byClass['Pos Pred Value'])
result_table[i,'NPV'] <- as.numeric(tr_res$byClass['Neg Pred Value'])
result_table[i,'RC'] <- as.numeric(sensitivity(pred_tr, x_train_1$y_train_binary, positive="X1"))
result_table[i,'F1'] <- as.numeric((2 * posPredValue(pred_tr, x_train_1$y_train_binary, positive="X1") * sensitivity(pred_tr, x_train_1$y_train_binary, positive="X1")) / (posPredValue(pred_tr, x_train_1$y_train_binary, positive="X1") + sensitivity(pred_tr, x_train_1$y_train_binary, positive="X1")))
tr_prob <- predict(weighted_fit, newdata=x_train_1, type="prob")
roc_obj <- roc(y_train_binary, tr_prob[[2]])
result_table[i,'AUC_score'] <- as.numeric(auc(roc_obj))

# CV output extraction
all_folds <- weighted_fit$pred
CV_results <- matrix(NA, nrow = 5, ncol = 14)
colnames(CV_results) <- c('Fold','TN','FP','FN','TP','ACC','BA','SN','SP','PPV','NPV','RC','F1','AUC_score')
for(j in 1:5){
	A <- paste('Fold',j,sep = '')
	B <- all_folds[all_folds$Resample == A,]
	CV_results[j,'Fold'] <- j
	CV_results[j,'TN'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[1])
	CV_results[j,'FP'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[2])
	CV_results[j,'FN'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[3])
	CV_results[j,'TP'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[4])
	CV_results[j,'ACC'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$overall['Accuracy'])
	CV_results[j,'BA'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Balanced Accuracy'])
	CV_results[j,'SN'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Sensitivity'])
	CV_results[j,'SP'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Specificity'])
	CV_results[j,'PPV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Pos Pred Value'])
	CV_results[j,'NPV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Neg Pred Value'])
	CV_results[j,'RC'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Recall'])
	CV_results[j,'F1'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['F1'])
	CV_results[j,'AUC_score'] <- as.numeric(auc(roc(B$obs, B$X1)))
}

CV_results_final <- colMeans(CV_results[,2:14], na.rm=TRUE)
CV_results_final[1:4] <- round(CV_results_final[1:4], 0)
CV_results_final[5:13] <- round(CV_results_final[5:13], 2)
# add to final result
result_table[i,18:30] <- CV_results_final

# test pred
pred_te = predict(weighted_fit, x_test)
y_test_binary_mod <- as.factor(y_test_binary)
levels(y_test_binary_mod) <- make.names(levels(factor(y_test_binary_mod)))
te_res <- confusionMatrix(data = pred_te, reference = y_test_binary_mod, positive = 'X1')

result_table[i,'TN_TE'] <- as.numeric(te_res$table[1])
result_table[i,'FP_TE'] <- as.numeric(te_res$table[2])
result_table[i,'FN_TE'] <- as.numeric(te_res$table[3])
result_table[i,'TP_TE'] <- as.numeric(te_res$table[4])
result_table[i,'ACC_TE'] <- round(as.numeric(te_res$overall['Accuracy']),2)
result_table[i,'BA_TE'] <- round(as.numeric(te_res$byClass['Balanced Accuracy']),2)
result_table[i,'SN_TE'] <- round(as.numeric(te_res$byClass['Sensitivity']),2)
result_table[i,'SP_TE'] <- round(as.numeric(te_res$byClass['Specificity']),2)
result_table[i,'PPV_TE'] <- round(as.numeric(te_res$byClass['Pos Pred Value']),2)
result_table[i,'NPV_TE'] <- round(as.numeric(te_res$byClass['Neg Pred Value']),2)
result_table[i,'RC_TE'] <- round(as.numeric(sensitivity(pred_te, y_test_binary_mod, positive="X1")),2)
result_table[i,'F1_TE'] <- round(as.numeric((2 * posPredValue(pred_te, y_test_binary_mod, positive="X1") * sensitivity(pred_te, y_test_binary_mod, positive="X1")) / (posPredValue(pred_te, y_test_binary_mod, positive="X1") + sensitivity(pred_te, y_test_binary_mod, positive="X1"))),2)
te_prob <- predict(weighted_fit, newdata=x_test, type="prob")
result_table[i,'AUC_score_TE'] <- round(as.numeric(auc(roc(y_test_binary, te_prob[[2]]))),2)

# external test set prediction

ext_test_set <- fread('avg_hybrid_descr_independent_test_set.csv')
# class
ext_test_set_class <- ifelse(ext_test_set$TOXICITY == "Hepatotoxic", 1, 0)

# replace '-' by underscore
colnames(ext_test_set) <- gsub('\\-','_',colnames(ext_test_set))

# extract descr 
ext_test_set_descr <- ext_test_set[,names(mean_train), with=F]
# standardize them using mean_train and std_train
ext_test_set_descr <- data.table(t(apply(ext_test_set_descr, 1, function(x) x - mean_train))) # centering
ext_test_set_descr <- data.table(t(apply(ext_test_set_descr, 1, function(x) x / std_train))) # scaling

# external predictions
pred_ext = predict(weighted_fit, ext_test_set_descr)
y_test_ext_binary_mod <- as.factor(ext_test_set_class)
levels(y_test_ext_binary_mod) <- make.names(levels(factor(y_test_ext_binary_mod)))
ext_res <- confusionMatrix(data = pred_ext, reference = y_test_ext_binary_mod, positive = 'X1')

result_table[i,'TN_EXT'] <- as.numeric(ext_res$table[1])
result_table[i,'FP_EXT'] <- as.numeric(ext_res$table[2])
result_table[i,'FN_EXT'] <- as.numeric(ext_res$table[3])
result_table[i,'TP_EXT'] <- as.numeric(ext_res$table[4])
result_table[i,'ACC_EXT'] <- round(as.numeric(ext_res$overall['Accuracy']),2)
result_table[i,'BA_EXT'] <- round(as.numeric(ext_res$byClass['Balanced Accuracy']),2)
result_table[i,'SN_EXT'] <- round(as.numeric(ext_res$byClass['Sensitivity']),2)
result_table[i,'SP_EXT'] <- round(as.numeric(ext_res$byClass['Specificity']),2)
result_table[i,'PPV_EXT'] <- round(as.numeric(ext_res$byClass['Pos Pred Value']),2)
result_table[i,'NPV_EXT'] <- round(as.numeric(ext_res$byClass['Neg Pred Value']),2)
result_table[i,'RC_EXT'] <- round(as.numeric(sensitivity(pred_ext, y_test_ext_binary_mod, positive="X1")),2)
result_table[i,'F1_EXT'] <- round(as.numeric((2 * posPredValue(pred_ext, y_test_ext_binary_mod, positive="X1") * sensitivity(pred_ext, y_test_ext_binary_mod, positive="X1")) / (posPredValue(pred_ext, y_test_ext_binary_mod, positive="X1") + sensitivity(pred_ext, y_test_ext_binary_mod, positive="X1"))),2)
ext_prob <- predict(weighted_fit, newdata=ext_test_set_descr, type="prob")
result_table[i,'AUC_score_EXT'] <- round(as.numeric(auc(roc(ext_test_set_class, ext_prob[[2]]))),2)

write.table(result_table,file = 'Results_HCI_HS_NoFS.csv', row.names = F, col.names = T, sep = ',')

# Var importance
ImpMeasure <- varImp(weighted_fit, scale = FALSE)

library(randomForest)
fm <- weighted_fit$finalModel
var_imp_details <- data.frame(randomForest::importance(fm, scale = FALSE))
var_imp_details$Vars <- row.names(var_imp_details)
var_imp_details <- var_imp_details[,c(5,1:4)]
var_imp_details <- var_imp_details[order(var_imp_details$MeanDecreaseGini, decreasing = T),]
write.table(var_imp_details,file = 'Imp_variables.csv', row.names = F, col.names = T, sep = ',')

# save model
mod_nm <- paste('HCI_HS_NoFS','.Rdata',sep = '')
save.image(file=mod_nm)


# 2. Model building using chemical variables

rm(list = setdiff(ls(),'path'))
setwd(path)

x_data <- fread('input_chemical.csv')
# to save all results
result_table <- matrix(NA,nrow = 1, ncol = 56)
colnames(result_table) <- c('Model','mtry','ntree','nodesize','TN','FP','FN','TP','ACC','BA','SN','SP','PPV','NPV','RC','F1','AUC_score',
	'TN_CV','FP_CV','FN_CV','TP_CV','ACC_CV','BA_CV','SN_CV','SP_CV','PPV_CV','NPV_CV','RC_CV','F1_CV','AUC_score_CV',
	'TN_TE','FP_TE','FN_TE','TP_TE','ACC_TE','BA_TE','SN_TE','SP_TE','PPV_TE','NPV_TE','RC_TE','F1_TE','AUC_score_TE',
	'TN_EXT','FP_EXT','FN_EXT','TP_EXT','ACC_EXT','BA_EXT','SN_EXT','SP_EXT','PPV_EXT','NPV_EXT','RC_EXT','F1_EXT','AUC_score_EXT')

set.seed(47)
# Data splitting in training and test set
x_train_orig <- stratified(x_data, c('CLASS'), 0.8)
x_test_orig <- setdiff(x_data,x_train_orig)

# save train and test set

new_folder <- paste(path,'/chemical_HS_NoFS',sep = '')
dir.create(new_folder)
setwd(new_folder)
write.table(x_train_orig,file = 'Orig_train_set.csv', row.names = F, col.names = T, sep = ',')
write.table(x_test_orig,file = 'Orig_test_set.csv', row.names = F, col.names = T, sep = ',')

# pre-processing train set 
x_train <- as.data.frame(x_train_orig[,!c('Sample_id', 'CLASS')])
# save the MEAN & STD. of x_train, later will be used for normalization of test set
mean_train <- apply(x_train, 2, mean, na.rm = TRUE)
std_train <- apply(x_train, 2, sd, na.rm = TRUE)
write.table(mean_train,file = 'Mean_train_set.csv', row.names = F, col.names = T, sep = ',')
write.table(std_train,file = 'STD_train_set.csv', row.names = F, col.names = T, sep = ',')

# center
x_train <- scale(x_train, center = T, scale = F)
# scale
x_train <- scale(x_train, center = FALSE, scale = apply(x_train, 2, sd, na.rm = TRUE))

# pre-processing test set
x_test <- as.data.frame(x_test_orig[,!c('Sample_id', 'CLASS')])
# scale and center using mean and std of train set
x_test <- x_test[,names(mean_train)]        # Arrange columns as same as 'mean_train' dataframe
x_test <- data.table(t(apply(x_test, 1, function(x) x - mean_train))) # centering
x_test <- data.table(t(apply(x_test, 1, function(x) x / std_train))) # scaling

y_train_binary <- x_train_orig$CLASS
y_test_binary <- x_test_orig$CLASS

# save normalized train and test set
write.table(cbind(x_train_orig[,1:2],x_train),file = 'train_set_norm.csv', row.names = F, col.names = T, sep = ',')
write.table(cbind(x_test_orig[,1:2],x_test),file = 'test_set_norm.csv', row.names = F, col.names = T, sep = ',')

x_train_1 <- as.data.frame(cbind(y_train_binary,x_train))
x_train_1$y_train_binary <- as.factor(x_train_1$y_train_binary)
levels(x_train_1$y_train_binary) <- make.names(levels(factor(x_train_1$y_train_binary)))

# RF model building
model_weights <- ifelse(y_train_binary=='0', (1/table(y_train_binary)[1]) * 0.5, (1/table(y_train_binary)[2]) * 0.5)

set.seed(47)
customRF <- list(type = "Classification", library = "randomForest", loop = NULL)
customRF$parameters <- data.frame(parameter = c("mtry", "ntree", "nodesize"), class = rep("numeric", 3), label = c("mtry", "ntree", "nodesize"))
customRF$grid <- function(x, y, len = NULL, search = "grid") {}
customRF$fit <- function(x, y, wts, param, lev, last, weights, classProbs, ...) {
	randomForest(x, y, mtry = param$mtry, ntree=param$ntree, ...)
}
customRF$predict <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
	predict(modelFit, newdata)
customRF$prob <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
	predict(modelFit, newdata, type = "prob")
customRF$sort <- function(x) x[order(x[,1]),]
customRF$levels <- function(x) x$classes

library(doMC)
registerDoMC(cores = 32)

tunegrid <- data.frame(expand.grid(.mtry=c(20,25,30,35,40), .ntree=c(50, 100,500), .nodesize=c(1:5)))
weighted_fit <- caret::train(y_train_binary ~ .,
	data = x_train_1,
	method = customRF,
	importance = TRUE,
	verbose = FALSE,
	replace=FALSE,
	metric = "Accuracy",
	tuneGrid = tunegrid,
	weights = model_weights,
	trControl = trainControl(method = "cv", number = 5, classProbs=T, savePredictions = "final", allowParallel = TRUE))

# save results
i <- 1
result_table[i,'Model'] <- 'Chemical_HS_NoFS'
result_table[i,'mtry'] <- weighted_fit$bestTune$mtry
result_table[i,'ntree'] <- weighted_fit$bestTune$ntree
result_table[i,'nodesize'] <- weighted_fit$bestTune$nodesize

# training pred
pred_tr = predict(weighted_fit, x_train_1[,-1])
tr_res <- confusionMatrix(data = pred_tr, reference = x_train_1$y_train_binary, positive = 'X1')

result_table[i,'TN'] <- as.numeric(tr_res$table[1])
result_table[i,'FP'] <- as.numeric(tr_res$table[2])
result_table[i,'FN'] <- as.numeric(tr_res$table[3])
result_table[i,'TP'] <- as.numeric(tr_res$table[4])
result_table[i,'ACC'] <- as.numeric(tr_res$overall['Accuracy'])
result_table[i,'BA'] <- as.numeric(tr_res$byClass['Balanced Accuracy'])
result_table[i,'SN'] <- as.numeric(tr_res$byClass['Sensitivity'])
result_table[i,'SP'] <- as.numeric(tr_res$byClass['Specificity'])
result_table[i,'PPV'] <- as.numeric(tr_res$byClass['Pos Pred Value'])
result_table[i,'NPV'] <- as.numeric(tr_res$byClass['Neg Pred Value'])
result_table[i,'RC'] <- as.numeric(sensitivity(pred_tr, x_train_1$y_train_binary, positive="X1"))
result_table[i,'F1'] <- as.numeric((2 * posPredValue(pred_tr, x_train_1$y_train_binary, positive="X1") * sensitivity(pred_tr, x_train_1$y_train_binary, positive="X1")) / (posPredValue(pred_tr, x_train_1$y_train_binary, positive="X1") + sensitivity(pred_tr, x_train_1$y_train_binary, positive="X1")))
tr_prob <- predict(weighted_fit, newdata=x_train_1, type="prob")
roc_obj <- roc(y_train_binary, tr_prob[[2]])
result_table[i,'AUC_score'] <- as.numeric(auc(roc_obj))

# CV output extraction
all_folds <- weighted_fit$pred
CV_results <- matrix(NA, nrow = 5, ncol = 14)
colnames(CV_results) <- c('Fold','TN','FP','FN','TP','ACC','BA','SN','SP','PPV','NPV','RC','F1','AUC_score')
for(j in 1:5){
	A <- paste('Fold',j,sep = '')
	B <- all_folds[all_folds$Resample == A,]
	CV_results[j,'Fold'] <- j
	CV_results[j,'TN'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[1])
	CV_results[j,'FP'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[2])
	CV_results[j,'FN'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[3])
	CV_results[j,'TP'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[4])
	CV_results[j,'ACC'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$overall['Accuracy'])
	CV_results[j,'BA'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Balanced Accuracy'])
	CV_results[j,'SN'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Sensitivity'])
	CV_results[j,'SP'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Specificity'])
	CV_results[j,'PPV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Pos Pred Value'])
	CV_results[j,'NPV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Neg Pred Value'])
	CV_results[j,'RC'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Recall'])
	CV_results[j,'F1'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['F1'])
	CV_results[j,'AUC_score'] <- as.numeric(auc(roc(B$obs, B$X1)))
}

CV_results_final <- colMeans(CV_results[,2:14], na.rm=TRUE)
CV_results_final[1:4] <- round(CV_results_final[1:4], 0)
CV_results_final[5:13] <- round(CV_results_final[5:13], 2)
# add to final result
result_table[i,18:30] <- CV_results_final

# test pred
pred_te = predict(weighted_fit, x_test)
y_test_binary_mod <- as.factor(y_test_binary)
levels(y_test_binary_mod) <- make.names(levels(factor(y_test_binary_mod)))
te_res <- confusionMatrix(data = pred_te, reference = y_test_binary_mod, positive = 'X1')

result_table[i,'TN_TE'] <- as.numeric(te_res$table[1])
result_table[i,'FP_TE'] <- as.numeric(te_res$table[2])
result_table[i,'FN_TE'] <- as.numeric(te_res$table[3])
result_table[i,'TP_TE'] <- as.numeric(te_res$table[4])
result_table[i,'ACC_TE'] <- round(as.numeric(te_res$overall['Accuracy']),2)
result_table[i,'BA_TE'] <- round(as.numeric(te_res$byClass['Balanced Accuracy']),2)
result_table[i,'SN_TE'] <- round(as.numeric(te_res$byClass['Sensitivity']),2)
result_table[i,'SP_TE'] <- round(as.numeric(te_res$byClass['Specificity']),2)
result_table[i,'PPV_TE'] <- round(as.numeric(te_res$byClass['Pos Pred Value']),2)
result_table[i,'NPV_TE'] <- round(as.numeric(te_res$byClass['Neg Pred Value']),2)
result_table[i,'RC_TE'] <- round(as.numeric(sensitivity(pred_te, y_test_binary_mod, positive="X1")),2)
result_table[i,'F1_TE'] <- round(as.numeric((2 * posPredValue(pred_te, y_test_binary_mod, positive="X1") * sensitivity(pred_te, y_test_binary_mod, positive="X1")) / (posPredValue(pred_te, y_test_binary_mod, positive="X1") + sensitivity(pred_te, y_test_binary_mod, positive="X1"))),2)
te_prob <- predict(weighted_fit, newdata=x_test, type="prob")
result_table[i,'AUC_score_TE'] <- round(as.numeric(auc(roc(y_test_binary, te_prob[[2]]))),2)

# external test set prediction

ext_test_set <- fread('avg_hybrid_descr_independent_test_set.csv')
# class
ext_test_set_class <- ifelse(ext_test_set$TOXICITY == "Hepatotoxic", 1, 0)

# replace '-' by underscore
colnames(ext_test_set) <- gsub('\\-','_',colnames(ext_test_set))

# extract descr 
ext_test_set_descr <- ext_test_set[,names(mean_train), with=F]
# standardize them using mean_train and std_train
ext_test_set_descr <- data.table(t(apply(ext_test_set_descr, 1, function(x) x - mean_train))) # centering
ext_test_set_descr <- data.table(t(apply(ext_test_set_descr, 1, function(x) x / std_train))) # scaling

# external predictions
pred_ext = predict(weighted_fit, ext_test_set_descr)
y_test_ext_binary_mod <- as.factor(ext_test_set_class)
levels(y_test_ext_binary_mod) <- make.names(levels(factor(y_test_ext_binary_mod)))
ext_res <- confusionMatrix(data = pred_ext, reference = y_test_ext_binary_mod, positive = 'X1')

result_table[i,'TN_EXT'] <- as.numeric(ext_res$table[1])
result_table[i,'FP_EXT'] <- as.numeric(ext_res$table[2])
result_table[i,'FN_EXT'] <- as.numeric(ext_res$table[3])
result_table[i,'TP_EXT'] <- as.numeric(ext_res$table[4])
result_table[i,'ACC_EXT'] <- round(as.numeric(ext_res$overall['Accuracy']),2)
result_table[i,'BA_EXT'] <- round(as.numeric(ext_res$byClass['Balanced Accuracy']),2)
result_table[i,'SN_EXT'] <- round(as.numeric(ext_res$byClass['Sensitivity']),2)
result_table[i,'SP_EXT'] <- round(as.numeric(ext_res$byClass['Specificity']),2)
result_table[i,'PPV_EXT'] <- round(as.numeric(ext_res$byClass['Pos Pred Value']),2)
result_table[i,'NPV_EXT'] <- round(as.numeric(ext_res$byClass['Neg Pred Value']),2)
result_table[i,'RC_EXT'] <- round(as.numeric(sensitivity(pred_ext, y_test_ext_binary_mod, positive="X1")),2)
result_table[i,'F1_EXT'] <- round(as.numeric((2 * posPredValue(pred_ext, y_test_ext_binary_mod, positive="X1") * sensitivity(pred_ext, y_test_ext_binary_mod, positive="X1")) / (posPredValue(pred_ext, y_test_ext_binary_mod, positive="X1") + sensitivity(pred_ext, y_test_ext_binary_mod, positive="X1"))),2)
ext_prob <- predict(weighted_fit, newdata=ext_test_set_descr, type="prob")
result_table[i,'AUC_score_EXT'] <- round(as.numeric(auc(roc(ext_test_set_class, ext_prob[[2]]))),2)

write.table(result_table,file = 'Results_chemical_HS_NoFS.csv', row.names = F, col.names = T, sep = ',')

# Var importance
ImpMeasure <- varImp(weighted_fit, scale = FALSE)

library(randomForest)
fm <- weighted_fit$finalModel
var_imp_details <- data.frame(randomForest::importance(fm, scale = FALSE))
var_imp_details$Vars <- row.names(var_imp_details)
var_imp_details <- var_imp_details[,c(5,1:4)]
var_imp_details <- var_imp_details[order(var_imp_details$MeanDecreaseGini, decreasing = T),]
write.table(var_imp_details,file = 'Imp_variables.csv', row.names = F, col.names = T, sep = ',')

# save model
mod_nm <- paste('chemical_HS_NoFS','.Rdata',sep = '')
save.image(file=mod_nm)


# 3. Model building using hybrid variables

rm(list = setdiff(ls(),'path'))
setwd(path)

x_data <- fread('input_hybrid.csv')
# to save all results
result_table <- matrix(NA,nrow = 1, ncol = 56)
colnames(result_table) <- c('Model','mtry','ntree','nodesize','TN','FP','FN','TP','ACC','BA','SN','SP','PPV','NPV','RC','F1','AUC_score',
	'TN_CV','FP_CV','FN_CV','TP_CV','ACC_CV','BA_CV','SN_CV','SP_CV','PPV_CV','NPV_CV','RC_CV','F1_CV','AUC_score_CV',
	'TN_TE','FP_TE','FN_TE','TP_TE','ACC_TE','BA_TE','SN_TE','SP_TE','PPV_TE','NPV_TE','RC_TE','F1_TE','AUC_score_TE',
	'TN_EXT','FP_EXT','FN_EXT','TP_EXT','ACC_EXT','BA_EXT','SN_EXT','SP_EXT','PPV_EXT','NPV_EXT','RC_EXT','F1_EXT','AUC_score_EXT')

set.seed(47)

# Data splitting in training and test set
x_train_orig <- stratified(x_data, c('CLASS'), 0.8)
x_test_orig <- setdiff(x_data,x_train_orig)

# save train and test set

new_folder <- paste(path,'/Hybrid_HS_NoFS',sep = '')
dir.create(new_folder)
setwd(new_folder)
write.table(x_train_orig,file = 'Orig_train_set.csv', row.names = F, col.names = T, sep = ',')
write.table(x_test_orig,file = 'Orig_test_set.csv', row.names = F, col.names = T, sep = ',')

# pre-processing train set 
x_train <- as.data.frame(x_train_orig[,!c('Sample_id', 'CLASS')])
# save the MEAN & STD. of x_train, later will be used for normalization of test set
mean_train <- apply(x_train, 2, mean, na.rm = TRUE)
std_train <- apply(x_train, 2, sd, na.rm = TRUE)
write.table(mean_train,file = 'Mean_train_set.csv', row.names = F, col.names = T, sep = ',')
write.table(std_train,file = 'STD_train_set.csv', row.names = F, col.names = T, sep = ',')

# center
x_train <- scale(x_train, center = T, scale = F)
# scale
x_train <- scale(x_train, center = FALSE, scale = apply(x_train, 2, sd, na.rm = TRUE))

# pre-processing test set
x_test <- as.data.frame(x_test_orig[,!c('Sample_id', 'CLASS')])
# scale and center using mean and std of train set
x_test <- x_test[,names(mean_train)]        # Arrange columns as same as 'mean_train' dataframe
x_test <- data.table(t(apply(x_test, 1, function(x) x - mean_train))) # centering
x_test <- data.table(t(apply(x_test, 1, function(x) x / std_train))) # scaling

y_train_binary <- x_train_orig$CLASS
y_test_binary <- x_test_orig$CLASS

# save normalized train and test set
write.table(cbind(x_train_orig[,1:2],x_train),file = 'train_set_norm.csv', row.names = F, col.names = T, sep = ',')
write.table(cbind(x_test_orig[,1:2],x_test),file = 'test_set_norm.csv', row.names = F, col.names = T, sep = ',')

x_train_1 <- as.data.frame(cbind(y_train_binary,x_train))
x_train_1$y_train_binary <- as.factor(x_train_1$y_train_binary)
levels(x_train_1$y_train_binary) <- make.names(levels(factor(x_train_1$y_train_binary)))

# RF model building
model_weights <- ifelse(y_train_binary=='0', (1/table(y_train_binary)[1]) * 0.5, (1/table(y_train_binary)[2]) * 0.5)

set.seed(47)

customRF <- list(type = "Classification", library = "randomForest", loop = NULL)
customRF$parameters <- data.frame(parameter = c("mtry", "ntree", "nodesize"), class = rep("numeric", 3), label = c("mtry", "ntree", "nodesize"))
customRF$grid <- function(x, y, len = NULL, search = "grid") {}
customRF$fit <- function(x, y, wts, param, lev, last, weights, classProbs, ...) {
	randomForest(x, y, mtry = param$mtry, ntree=param$ntree, ...)
}
customRF$predict <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
	predict(modelFit, newdata)
customRF$prob <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
	predict(modelFit, newdata, type = "prob")
customRF$sort <- function(x) x[order(x[,1]),]
customRF$levels <- function(x) x$classes

library(doMC)
registerDoMC(cores = 32)

tunegrid <- data.frame(expand.grid(.mtry=c(40,45,50,55,60), .ntree=c(50, 100,500), .nodesize=c(1:5)))
weighted_fit <- caret::train(y_train_binary ~ .,
	data = x_train_1,
	method = customRF,
	importance = TRUE,
	verbose = FALSE,
	replace=FALSE,
	metric = "Accuracy",
	tuneGrid = tunegrid,
	weights = model_weights,
	trControl = trainControl(method = "cv", number = 5, classProbs=T, savePredictions = "final", allowParallel = TRUE))

# save results
i <- 1
result_table[i,'Model'] <- 'Hybrid_HS_NoFS'
result_table[i,'mtry'] <- weighted_fit$bestTune$mtry
result_table[i,'ntree'] <- weighted_fit$bestTune$ntree
result_table[i,'nodesize'] <- weighted_fit$bestTune$nodesize

# training pred
pred_tr = predict(weighted_fit, x_train_1[,-1])
tr_res <- confusionMatrix(data = pred_tr, reference = x_train_1$y_train_binary, positive = 'X1')

result_table[i,'TN'] <- as.numeric(tr_res$table[1])
result_table[i,'FP'] <- as.numeric(tr_res$table[2])
result_table[i,'FN'] <- as.numeric(tr_res$table[3])
result_table[i,'TP'] <- as.numeric(tr_res$table[4])
result_table[i,'ACC'] <- as.numeric(tr_res$overall['Accuracy'])
result_table[i,'BA'] <- as.numeric(tr_res$byClass['Balanced Accuracy'])
result_table[i,'SN'] <- as.numeric(tr_res$byClass['Sensitivity'])
result_table[i,'SP'] <- as.numeric(tr_res$byClass['Specificity'])
result_table[i,'PPV'] <- as.numeric(tr_res$byClass['Pos Pred Value'])
result_table[i,'NPV'] <- as.numeric(tr_res$byClass['Neg Pred Value'])
result_table[i,'RC'] <- as.numeric(sensitivity(pred_tr, x_train_1$y_train_binary, positive="X1"))
result_table[i,'F1'] <- as.numeric((2 * posPredValue(pred_tr, x_train_1$y_train_binary, positive="X1") * sensitivity(pred_tr, x_train_1$y_train_binary, positive="X1")) / (posPredValue(pred_tr, x_train_1$y_train_binary, positive="X1") + sensitivity(pred_tr, x_train_1$y_train_binary, positive="X1")))
tr_prob <- predict(weighted_fit, newdata=x_train_1, type="prob")
roc_obj <- roc(y_train_binary, tr_prob[[2]])
result_table[i,'AUC_score'] <- as.numeric(auc(roc_obj))

# CV output extraction
all_folds <- weighted_fit$pred
CV_results <- matrix(NA, nrow = 5, ncol = 14)
colnames(CV_results) <- c('Fold','TN','FP','FN','TP','ACC','BA','SN','SP','PPV','NPV','RC','F1','AUC_score')
for(j in 1:5){
	A <- paste('Fold',j,sep = '')
	B <- all_folds[all_folds$Resample == A,]
	CV_results[j,'Fold'] <- j
	CV_results[j,'TN'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[1])
	CV_results[j,'FP'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[2])
	CV_results[j,'FN'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[3])
	CV_results[j,'TP'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[4])
	CV_results[j,'ACC'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$overall['Accuracy'])
	CV_results[j,'BA'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Balanced Accuracy'])
	CV_results[j,'SN'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Sensitivity'])
	CV_results[j,'SP'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Specificity'])
	CV_results[j,'PPV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Pos Pred Value'])
	CV_results[j,'NPV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Neg Pred Value'])
	CV_results[j,'RC'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Recall'])
	CV_results[j,'F1'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['F1'])
	CV_results[j,'AUC_score'] <- as.numeric(auc(roc(B$obs, B$X1)))
}

CV_results_final <- colMeans(CV_results[,2:14], na.rm=TRUE)
CV_results_final[1:4] <- round(CV_results_final[1:4], 0)
CV_results_final[5:13] <- round(CV_results_final[5:13], 2)
# add to final result
result_table[i,18:30] <- CV_results_final

# test pred
pred_te = predict(weighted_fit, x_test)
y_test_binary_mod <- as.factor(y_test_binary)
levels(y_test_binary_mod) <- make.names(levels(factor(y_test_binary_mod)))
te_res <- confusionMatrix(data = pred_te, reference = y_test_binary_mod, positive = 'X1')

result_table[i,'TN_TE'] <- as.numeric(te_res$table[1])
result_table[i,'FP_TE'] <- as.numeric(te_res$table[2])
result_table[i,'FN_TE'] <- as.numeric(te_res$table[3])
result_table[i,'TP_TE'] <- as.numeric(te_res$table[4])
result_table[i,'ACC_TE'] <- round(as.numeric(te_res$overall['Accuracy']),2)
result_table[i,'BA_TE'] <- round(as.numeric(te_res$byClass['Balanced Accuracy']),2)
result_table[i,'SN_TE'] <- round(as.numeric(te_res$byClass['Sensitivity']),2)
result_table[i,'SP_TE'] <- round(as.numeric(te_res$byClass['Specificity']),2)
result_table[i,'PPV_TE'] <- round(as.numeric(te_res$byClass['Pos Pred Value']),2)
result_table[i,'NPV_TE'] <- round(as.numeric(te_res$byClass['Neg Pred Value']),2)
result_table[i,'RC_TE'] <- round(as.numeric(sensitivity(pred_te, y_test_binary_mod, positive="X1")),2)
result_table[i,'F1_TE'] <- round(as.numeric((2 * posPredValue(pred_te, y_test_binary_mod, positive="X1") * sensitivity(pred_te, y_test_binary_mod, positive="X1")) / (posPredValue(pred_te, y_test_binary_mod, positive="X1") + sensitivity(pred_te, y_test_binary_mod, positive="X1"))),2)
te_prob <- predict(weighted_fit, newdata=x_test, type="prob")
result_table[i,'AUC_score_TE'] <- round(as.numeric(auc(roc(y_test_binary, te_prob[[2]]))),2)

# external test set prediction

ext_test_set <- fread('avg_hybrid_descr_independent_test_set.csv')
# class
ext_test_set_class <- ifelse(ext_test_set$TOXICITY == "Hepatotoxic", 1, 0)

# replace '-' by underscore
colnames(ext_test_set) <- gsub('\\-','_',colnames(ext_test_set))

# extract descr 
ext_test_set_descr <- ext_test_set[,names(mean_train), with=F]
# standardize them using mean_train and std_train
ext_test_set_descr <- data.table(t(apply(ext_test_set_descr, 1, function(x) x - mean_train))) # centering
ext_test_set_descr <- data.table(t(apply(ext_test_set_descr, 1, function(x) x / std_train))) # scaling

# external predictions
pred_ext = predict(weighted_fit, ext_test_set_descr)
y_test_ext_binary_mod <- as.factor(ext_test_set_class)
levels(y_test_ext_binary_mod) <- make.names(levels(factor(y_test_ext_binary_mod)))
ext_res <- confusionMatrix(data = pred_ext, reference = y_test_ext_binary_mod, positive = 'X1')

result_table[i,'TN_EXT'] <- as.numeric(ext_res$table[1])
result_table[i,'FP_EXT'] <- as.numeric(ext_res$table[2])
result_table[i,'FN_EXT'] <- as.numeric(ext_res$table[3])
result_table[i,'TP_EXT'] <- as.numeric(ext_res$table[4])
result_table[i,'ACC_EXT'] <- round(as.numeric(ext_res$overall['Accuracy']),2)
result_table[i,'BA_EXT'] <- round(as.numeric(ext_res$byClass['Balanced Accuracy']),2)
result_table[i,'SN_EXT'] <- round(as.numeric(ext_res$byClass['Sensitivity']),2)
result_table[i,'SP_EXT'] <- round(as.numeric(ext_res$byClass['Specificity']),2)
result_table[i,'PPV_EXT'] <- round(as.numeric(ext_res$byClass['Pos Pred Value']),2)
result_table[i,'NPV_EXT'] <- round(as.numeric(ext_res$byClass['Neg Pred Value']),2)
result_table[i,'RC_EXT'] <- round(as.numeric(sensitivity(pred_ext, y_test_ext_binary_mod, positive="X1")),2)
result_table[i,'F1_EXT'] <- round(as.numeric((2 * posPredValue(pred_ext, y_test_ext_binary_mod, positive="X1") * sensitivity(pred_ext, y_test_ext_binary_mod, positive="X1")) / (posPredValue(pred_ext, y_test_ext_binary_mod, positive="X1") + sensitivity(pred_ext, y_test_ext_binary_mod, positive="X1"))),2)
ext_prob <- predict(weighted_fit, newdata=ext_test_set_descr, type="prob")
result_table[i,'AUC_score_EXT'] <- round(as.numeric(auc(roc(ext_test_set_class, ext_prob[[2]]))),2)

write.table(result_table,file = 'Results_Hybrid_HS_NoFS.csv', row.names = F, col.names = T, sep = ',')

# Var importance
ImpMeasure <- varImp(weighted_fit, scale = FALSE)

library(randomForest)
fm <- weighted_fit$finalModel
var_imp_details <- data.frame(randomForest::importance(fm, scale = FALSE))
var_imp_details$Vars <- row.names(var_imp_details)
var_imp_details <- var_imp_details[,c(5,1:4)]
var_imp_details <- var_imp_details[order(var_imp_details$MeanDecreaseGini, decreasing = T),]
write.table(var_imp_details,file = 'Imp_variables.csv', row.names = F, col.names = T, sep = ',')

# save model
mod_nm <- paste('Hybrid_HS_NoFS','.Rdata',sep = '')
save.image(file=mod_nm)


######################################################################################################
#             Approach - C:  No Hyperparameter Selection & Feature Selection (NoHS_FS)
######################################################################################################


# 1.   Model building using HCI variables

rm(list = setdiff(ls(),'path'))
setwd(path)

x_data <- fread('input_HCI.csv')
# to save all results
result_table <- matrix(NA,nrow = 1, ncol = 54)
colnames(result_table) <- c('Model','var_count','TN','FP','FN','TP','ACC','BA','SN','SP','PPV','NPV','RC','F1','AUC_score',
	'TN_CV','FP_CV','FN_CV','TP_CV','ACC_CV','BA_CV','SN_CV','SP_CV','PPV_CV','NPV_CV','RC_CV','F1_CV','AUC_score_CV',
	'TN_TE','FP_TE','FN_TE','TP_TE','ACC_TE','BA_TE','SN_TE','SP_TE','PPV_TE','NPV_TE','RC_TE','F1_TE','AUC_score_TE',
	'TN_EXT','FP_EXT','FN_EXT','TP_EXT','ACC_EXT','BA_EXT','SN_EXT','SP_EXT','PPV_EXT','NPV_EXT','RC_EXT','F1_EXT','AUC_score_EXT')

set.seed(47)

# Data splitting in training and test set
x_train_orig <- stratified(x_data, c('CLASS'), 0.8)
x_test_orig <- setdiff(x_data,x_train_orig)

# save train and test set

new_folder <- paste(path,'/HCI_NoHS_FS',sep = '')
dir.create(new_folder)
setwd(new_folder)
write.table(x_train_orig,file = 'Orig_train_set.csv', row.names = F, col.names = T, sep = ',')
write.table(x_test_orig,file = 'Orig_test_set.csv', row.names = F, col.names = T, sep = ',')

# pre-processing train set 
x_train <- as.data.frame(x_train_orig[,!c('Sample_id', 'CLASS')])

# save the MEAN & STD. of x_train, later will be used for normalization of test set
mean_train <- apply(x_train, 2, mean, na.rm = TRUE)
std_train <- apply(x_train, 2, sd, na.rm = TRUE)
write.table(mean_train,file = 'Mean_train_set.csv', row.names = F, col.names = T, sep = ',')
write.table(std_train,file = 'STD_train_set.csv', row.names = F, col.names = T, sep = ',')

# center
x_train <- scale(x_train, center = T, scale = F)
# scale
x_train <- scale(x_train, center = FALSE, scale = apply(x_train, 2, sd, na.rm = TRUE))

# find significant descriptors
library(Boruta)

set.seed(47)

FS_boruta_model <- Boruta(x_train, x_train_orig$CLASS, pValue = 0.05, mcAdj = TRUE, maxRuns = 100)
best_var <- colnames(x_train)[which(FS_boruta_model$finalDecision == 'Confirmed')]

x_train <- x_train[,best_var]

# pre-processing test set
x_test <- as.data.frame(x_test_orig[,!c('Sample_id', 'CLASS')])
# scale and center using mean and std of train set
x_test <- x_test[,names(mean_train)]        # Arrange columns as same as 'mean_train' dataframe
x_test <- data.table(t(apply(x_test, 1, function(x) x - mean_train))) # centering
x_test <- data.table(t(apply(x_test, 1, function(x) x / std_train))) # scaling
x_test <- x_test[,best_var,with=F]
y_train_binary <- x_train_orig$CLASS
y_test_binary <- x_test_orig$CLASS

# save normalized train and test set
write.table(cbind(x_train_orig[,1:2],x_train),file = 'train_set_norm.csv', row.names = F, col.names = T, sep = ',')
write.table(cbind(x_test_orig[,1:2],x_test),file = 'test_set_norm.csv', row.names = F, col.names = T, sep = ',')

x_train_1 <- as.data.frame(cbind(y_train_binary,x_train))
x_train_1$y_train_binary <- as.factor(x_train_1$y_train_binary)
levels(x_train_1$y_train_binary) <- make.names(levels(factor(x_train_1$y_train_binary)))

# RF model building
model_weights <- ifelse(y_train_binary=='0', (1/table(y_train_binary)[1]) * 0.5, (1/table(y_train_binary)[2]) * 0.5)
seeds <- as.vector(c(1:6), mode = "list")
seeds[[6]] <- 1
mtry_val = as.integer(sqrt(dim(x_train_1)[2]))

set.seed(47)

library(doMC)
registerDoMC(cores = 32)

weighted_fit <- caret::train(y_train_binary ~ .,
	data = x_train_1,
	method = "rf",
	importance = TRUE,
	verbose = FALSE,
	ntree=100,
	replace=FALSE, 
	nodesize = 5,
	metric = "Accuracy",
	tuneGrid = data.frame(mtry = mtry_val),
	tuneLength = 1,
	weights = model_weights,
	trControl = trainControl(method = "cv", number = 5, classProbs=T, savePredictions = "final", seeds = seeds))

# training pred
pred_tr = predict(weighted_fit, x_train)
tr_res <- confusionMatrix(data = pred_tr, reference = x_train_1$y_train_binary, positive = 'X1')

i <- 1
result_table[i,'Model'] <- 'HCI_NoHS_FS'
result_table[i,'var_count'] <- c(length(best_var))
result_table[i,'TN'] <- as.numeric(tr_res$table[1])
result_table[i,'FP'] <- as.numeric(tr_res$table[2])
result_table[i,'FN'] <- as.numeric(tr_res$table[3])
result_table[i,'TP'] <- as.numeric(tr_res$table[4])
result_table[i,'ACC'] <- as.numeric(tr_res$overall['Accuracy'])
result_table[i,'BA'] <- as.numeric(tr_res$byClass['Balanced Accuracy'])
result_table[i,'SN'] <- as.numeric(tr_res$byClass['Sensitivity'])
result_table[i,'SP'] <- as.numeric(tr_res$byClass['Specificity'])
result_table[i,'PPV'] <- as.numeric(tr_res$byClass['Pos Pred Value'])
result_table[i,'NPV'] <- as.numeric(tr_res$byClass['Neg Pred Value'])
result_table[i,'RC'] <- as.numeric(sensitivity(pred_tr, x_train_1$y_train_binary, positive="X1"))
result_table[i,'F1'] <- as.numeric((2 * posPredValue(pred_tr, x_train_1$y_train_binary, positive="X1") * sensitivity(pred_tr, x_train_1$y_train_binary, positive="X1")) / (posPredValue(pred_tr, x_train_1$y_train_binary, positive="X1") + sensitivity(pred_tr, x_train_1$y_train_binary, positive="X1")))
tr_prob <- predict(weighted_fit, newdata=x_train_1, type="prob")
roc_obj <- roc(y_train_binary, tr_prob[[2]])
result_table[i,'AUC_score'] <- as.numeric(auc(roc_obj))

# CV output extraction
all_folds <- weighted_fit$pred
CV_results <- matrix(NA, nrow = 5, ncol = 14)
colnames(CV_results) <- c('Fold','TN','FP','FN','TP','ACC','BA','SN','SP','PPV','NPV','RC','F1','AUC_score')
for(j in 1:5){
	A <- paste('Fold',j,sep = '')
	B <- all_folds[all_folds$Resample == A,]
	CV_results[j,'Fold'] <- j
	CV_results[j,'TN'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[1])
	CV_results[j,'FP'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[2])
	CV_results[j,'FN'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[3])
	CV_results[j,'TP'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[4])
	CV_results[j,'ACC'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$overall['Accuracy'])
	CV_results[j,'BA'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Balanced Accuracy'])
	CV_results[j,'SN'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Sensitivity'])
	CV_results[j,'SP'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Specificity'])
	CV_results[j,'PPV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Pos Pred Value'])
	CV_results[j,'NPV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Neg Pred Value'])
	CV_results[j,'RC'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Recall'])
	CV_results[j,'F1'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['F1'])
	CV_results[j,'AUC_score'] <- as.numeric(auc(roc(B$obs, B$X1)))
}

CV_results_final <- colMeans(CV_results[,2:14], na.rm=TRUE)
CV_results_final[1:4] <- round(CV_results_final[1:4], 0)
CV_results_final[5:13] <- round(CV_results_final[5:13], 2)
# add to final result
result_table[i,16:28] <- CV_results_final

# test pred
pred_te = predict(weighted_fit, x_test)
y_test_binary_mod <- as.factor(y_test_binary)
levels(y_test_binary_mod) <- make.names(levels(factor(y_test_binary_mod)))
te_res <- confusionMatrix(data = pred_te, reference = y_test_binary_mod, positive = 'X1')

result_table[i,'TN_TE'] <- as.numeric(te_res$table[1])
result_table[i,'FP_TE'] <- as.numeric(te_res$table[2])
result_table[i,'FN_TE'] <- as.numeric(te_res$table[3])
result_table[i,'TP_TE'] <- as.numeric(te_res$table[4])
result_table[i,'ACC_TE'] <- round(as.numeric(te_res$overall['Accuracy']),2)
result_table[i,'BA_TE'] <- round(as.numeric(te_res$byClass['Balanced Accuracy']),2)
result_table[i,'SN_TE'] <- round(as.numeric(te_res$byClass['Sensitivity']),2)
result_table[i,'SP_TE'] <- round(as.numeric(te_res$byClass['Specificity']),2)
result_table[i,'PPV_TE'] <- round(as.numeric(te_res$byClass['Pos Pred Value']),2)
result_table[i,'NPV_TE'] <- round(as.numeric(te_res$byClass['Neg Pred Value']),2)
result_table[i,'RC_TE'] <- round(as.numeric(sensitivity(pred_te, y_test_binary_mod, positive="X1")),2)
result_table[i,'F1_TE'] <- round(as.numeric((2 * posPredValue(pred_te, y_test_binary_mod, positive="X1") * sensitivity(pred_te, y_test_binary_mod, positive="X1")) / (posPredValue(pred_te, y_test_binary_mod, positive="X1") + sensitivity(pred_te, y_test_binary_mod, positive="X1"))),2)
te_prob <- predict(weighted_fit, newdata=x_test, type="prob")
result_table[i,'AUC_score_TE'] <- round(as.numeric(auc(roc(y_test_binary, te_prob[[2]]))),2)

# external test set prediction

ext_test_set <- fread('avg_hybrid_descr_independent_test_set.csv')
# class
ext_test_set_class <- ifelse(ext_test_set$TOXICITY == "Hepatotoxic", 1, 0)

# replace '-' by underscore
colnames(ext_test_set) <- gsub('\\-','_',colnames(ext_test_set))

# extract descr 
ext_test_set_descr <- ext_test_set[,names(mean_train), with=F]
# standardize them using mean_train and std_train
ext_test_set_descr <- data.table(t(apply(ext_test_set_descr, 1, function(x) x - mean_train))) # centering
ext_test_set_descr <- data.table(t(apply(ext_test_set_descr, 1, function(x) x / std_train))) # scaling

# external predictions
pred_ext = predict(weighted_fit, ext_test_set_descr)
y_test_ext_binary_mod <- as.factor(ext_test_set_class)
levels(y_test_ext_binary_mod) <- make.names(levels(factor(y_test_ext_binary_mod)))
ext_res <- confusionMatrix(data = pred_ext, reference = y_test_ext_binary_mod, positive = 'X1')

result_table[i,'TN_EXT'] <- as.numeric(ext_res$table[1])
result_table[i,'FP_EXT'] <- as.numeric(ext_res$table[2])
result_table[i,'FN_EXT'] <- as.numeric(ext_res$table[3])
result_table[i,'TP_EXT'] <- as.numeric(ext_res$table[4])
result_table[i,'ACC_EXT'] <- round(as.numeric(ext_res$overall['Accuracy']),2)
result_table[i,'BA_EXT'] <- round(as.numeric(ext_res$byClass['Balanced Accuracy']),2)
result_table[i,'SN_EXT'] <- round(as.numeric(ext_res$byClass['Sensitivity']),2)
result_table[i,'SP_EXT'] <- round(as.numeric(ext_res$byClass['Specificity']),2)
result_table[i,'PPV_EXT'] <- round(as.numeric(ext_res$byClass['Pos Pred Value']),2)
result_table[i,'NPV_EXT'] <- round(as.numeric(ext_res$byClass['Neg Pred Value']),2)
result_table[i,'RC_EXT'] <- round(as.numeric(sensitivity(pred_ext, y_test_ext_binary_mod, positive="X1")),2)
result_table[i,'F1_EXT'] <- round(as.numeric((2 * posPredValue(pred_ext, y_test_ext_binary_mod, positive="X1") * sensitivity(pred_ext, y_test_ext_binary_mod, positive="X1")) / (posPredValue(pred_ext, y_test_ext_binary_mod, positive="X1") + sensitivity(pred_ext, y_test_ext_binary_mod, positive="X1"))),2)
ext_prob <- predict(weighted_fit, newdata=ext_test_set_descr, type="prob")
result_table[i,'AUC_score_EXT'] <- round(as.numeric(auc(roc(ext_test_set_class, ext_prob[[2]]))),2)

write.table(result_table,file = 'Results_HCI_NoHS_FS.csv', row.names = F, col.names = T, sep = ',')

# Var importance
ImpMeasure <- varImp(weighted_fit, scale = FALSE)

library(randomForest)
fm <- weighted_fit$finalModel
var_imp_details <- data.frame(randomForest::importance(fm, scale = FALSE))
var_imp_details$Vars <- row.names(var_imp_details)
var_imp_details <- var_imp_details[,c(5,1:4)]
var_imp_details <- var_imp_details[order(var_imp_details$MeanDecreaseGini, decreasing = T),]
write.table(var_imp_details,file = 'Imp_variables.csv', row.names = F, col.names = T, sep = ',')

# save model
mod_nm <- paste('HCI_NoHS_FS','.Rdata',sep = '')
save.image(file=mod_nm)


# 2. Model building with chemical descr 

rm(list = setdiff(ls(),'path'))
setwd(path)

x_data <- fread('input_chemical.csv')
# to save all results
result_table <- matrix(NA,nrow = 1, ncol = 54)
colnames(result_table) <- c('Model','var_count','TN','FP','FN','TP','ACC','BA','SN','SP','PPV','NPV','RC','F1','AUC_score',
	'TN_CV','FP_CV','FN_CV','TP_CV','ACC_CV','BA_CV','SN_CV','SP_CV','PPV_CV','NPV_CV','RC_CV','F1_CV','AUC_score_CV',
	'TN_TE','FP_TE','FN_TE','TP_TE','ACC_TE','BA_TE','SN_TE','SP_TE','PPV_TE','NPV_TE','RC_TE','F1_TE','AUC_score_TE',
	'TN_EXT','FP_EXT','FN_EXT','TP_EXT','ACC_EXT','BA_EXT','SN_EXT','SP_EXT','PPV_EXT','NPV_EXT','RC_EXT','F1_EXT','AUC_score_EXT')


set.seed(47)
# Data splitting in training and test set
x_train_orig <- stratified(x_data, c('CLASS'), 0.8)
x_test_orig <- setdiff(x_data,x_train_orig)

# save train and test set

new_folder <- paste(path,'/chemical_NoHS_FS',sep = '')
dir.create(new_folder)
setwd(new_folder)
write.table(x_train_orig,file = 'Orig_train_set.csv', row.names = F, col.names = T, sep = ',')
write.table(x_test_orig,file = 'Orig_test_set.csv', row.names = F, col.names = T, sep = ',')

# pre-processing train set 
x_train <- as.data.frame(x_train_orig[,!c('Sample_id', 'CLASS')])

# save the MEAN & STD. of x_train, later will be used for normalization of test set
mean_train <- apply(x_train, 2, mean, na.rm = TRUE)
std_train <- apply(x_train, 2, sd, na.rm = TRUE)
write.table(mean_train,file = 'Mean_train_set.csv', row.names = F, col.names = T, sep = ',')
write.table(std_train,file = 'STD_train_set.csv', row.names = F, col.names = T, sep = ',')

# center
x_train <- scale(x_train, center = T, scale = F)
# scale
x_train <- scale(x_train, center = FALSE, scale = apply(x_train, 2, sd, na.rm = TRUE))

# find significant descriptors
library(Boruta)

set.seed(47)

FS_boruta_model <- Boruta(x_train, x_train_orig$CLASS, pValue = 0.05, mcAdj = TRUE, maxRuns = 100)
best_var <- colnames(x_train)[which(FS_boruta_model$finalDecision == 'Confirmed')]
# final train & test set
x_train <- x_train[,best_var]

# pre-processing test set
x_test <- as.data.frame(x_test_orig[,!c('Sample_id', 'CLASS')])
# scale and center using mean and std of train set
x_test <- x_test[,names(mean_train)]        # Arrange columns as same as 'mean_train' dataframe
x_test <- data.table(t(apply(x_test, 1, function(x) x - mean_train))) # centering
x_test <- data.table(t(apply(x_test, 1, function(x) x / std_train))) # scaling
x_test <- x_test[,best_var,with=F]
y_train_binary <- x_train_orig$CLASS
y_test_binary <- x_test_orig$CLASS

# save normalized train and test set
write.table(cbind(x_train_orig[,1:2],x_train),file = 'train_set_norm.csv', row.names = F, col.names = T, sep = ',')
write.table(cbind(x_test_orig[,1:2],x_test),file = 'test_set_norm.csv', row.names = F, col.names = T, sep = ',')

x_train_1 <- as.data.frame(cbind(y_train_binary,x_train))
x_train_1$y_train_binary <- as.factor(x_train_1$y_train_binary)
levels(x_train_1$y_train_binary) <- make.names(levels(factor(x_train_1$y_train_binary)))

# RF model building
model_weights <- ifelse(y_train_binary=='0', (1/table(y_train_binary)[1]) * 0.5, (1/table(y_train_binary)[2]) * 0.5)
seeds <- as.vector(c(1:6), mode = "list")
seeds[[6]] <- 1
mtry_val = as.integer(sqrt(dim(x_train_1)[2]))

set.seed(47)

library(doMC)
registerDoMC(cores = 32)

weighted_fit <- caret::train(y_train_binary ~ .,
	data = x_train_1,
	method = "rf",
	importance = TRUE,
	verbose = FALSE,
	ntree=100,
	replace=FALSE, 
	nodesize = 5,
	metric = "Accuracy",
	tuneGrid = data.frame(mtry = mtry_val),
	tuneLength = 1,
	weights = model_weights,
	trControl = trainControl(method = "cv", number = 5, classProbs=T, savePredictions = "final", seeds = seeds))

# training pred
pred_tr = predict(weighted_fit, x_train)
tr_res <- confusionMatrix(data = pred_tr, reference = x_train_1$y_train_binary, positive = 'X1')

i <- 1
result_table[i,'Model'] <- 'Chemical_NoHS_FS'
result_table[i,'var_count'] <- c(length(best_var))
result_table[i,'TN'] <- as.numeric(tr_res$table[1])
result_table[i,'FP'] <- as.numeric(tr_res$table[2])
result_table[i,'FN'] <- as.numeric(tr_res$table[3])
result_table[i,'TP'] <- as.numeric(tr_res$table[4])
result_table[i,'ACC'] <- as.numeric(tr_res$overall['Accuracy'])
result_table[i,'BA'] <- as.numeric(tr_res$byClass['Balanced Accuracy'])
result_table[i,'SN'] <- as.numeric(tr_res$byClass['Sensitivity'])
result_table[i,'SP'] <- as.numeric(tr_res$byClass['Specificity'])
result_table[i,'PPV'] <- as.numeric(tr_res$byClass['Pos Pred Value'])
result_table[i,'NPV'] <- as.numeric(tr_res$byClass['Neg Pred Value'])
result_table[i,'RC'] <- as.numeric(sensitivity(pred_tr, x_train_1$y_train_binary, positive="X1"))
result_table[i,'F1'] <- as.numeric((2 * posPredValue(pred_tr, x_train_1$y_train_binary, positive="X1") * sensitivity(pred_tr, x_train_1$y_train_binary, positive="X1")) / (posPredValue(pred_tr, x_train_1$y_train_binary, positive="X1") + sensitivity(pred_tr, x_train_1$y_train_binary, positive="X1")))
tr_prob <- predict(weighted_fit, newdata=x_train_1, type="prob")
roc_obj <- roc(y_train_binary, tr_prob[[2]])
result_table[i,'AUC_score'] <- as.numeric(auc(roc_obj))

# CV output extraction
all_folds <- weighted_fit$pred
CV_results <- matrix(NA, nrow = 5, ncol = 14)
colnames(CV_results) <- c('Fold','TN','FP','FN','TP','ACC','BA','SN','SP','PPV','NPV','RC','F1','AUC_score')
for(j in 1:5){
	A <- paste('Fold',j,sep = '')
	B <- all_folds[all_folds$Resample == A,]
	CV_results[j,'Fold'] <- j
	CV_results[j,'TN'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[1])
	CV_results[j,'FP'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[2])
	CV_results[j,'FN'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[3])
	CV_results[j,'TP'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[4])
	CV_results[j,'ACC'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$overall['Accuracy'])
	CV_results[j,'BA'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Balanced Accuracy'])
	CV_results[j,'SN'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Sensitivity'])
	CV_results[j,'SP'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Specificity'])
	CV_results[j,'PPV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Pos Pred Value'])
	CV_results[j,'NPV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Neg Pred Value'])
	CV_results[j,'RC'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Recall'])
	CV_results[j,'F1'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['F1'])
	CV_results[j,'AUC_score'] <- as.numeric(auc(roc(B$obs, B$X1)))
}

CV_results_final <- colMeans(CV_results[,2:14], na.rm=TRUE)
CV_results_final[1:4] <- round(CV_results_final[1:4], 0)
CV_results_final[5:13] <- round(CV_results_final[5:13], 2)
# add to final result
result_table[i,16:28] <- CV_results_final

# test pred
pred_te = predict(weighted_fit, x_test)
y_test_binary_mod <- as.factor(y_test_binary)
levels(y_test_binary_mod) <- make.names(levels(factor(y_test_binary_mod)))
te_res <- confusionMatrix(data = pred_te, reference = y_test_binary_mod, positive = 'X1')

result_table[i,'TN_TE'] <- as.numeric(te_res$table[1])
result_table[i,'FP_TE'] <- as.numeric(te_res$table[2])
result_table[i,'FN_TE'] <- as.numeric(te_res$table[3])
result_table[i,'TP_TE'] <- as.numeric(te_res$table[4])
result_table[i,'ACC_TE'] <- round(as.numeric(te_res$overall['Accuracy']),2)
result_table[i,'BA_TE'] <- round(as.numeric(te_res$byClass['Balanced Accuracy']),2)
result_table[i,'SN_TE'] <- round(as.numeric(te_res$byClass['Sensitivity']),2)
result_table[i,'SP_TE'] <- round(as.numeric(te_res$byClass['Specificity']),2)
result_table[i,'PPV_TE'] <- round(as.numeric(te_res$byClass['Pos Pred Value']),2)
result_table[i,'NPV_TE'] <- round(as.numeric(te_res$byClass['Neg Pred Value']),2)
result_table[i,'RC_TE'] <- round(as.numeric(sensitivity(pred_te, y_test_binary_mod, positive="X1")),2)
result_table[i,'F1_TE'] <- round(as.numeric((2 * posPredValue(pred_te, y_test_binary_mod, positive="X1") * sensitivity(pred_te, y_test_binary_mod, positive="X1")) / (posPredValue(pred_te, y_test_binary_mod, positive="X1") + sensitivity(pred_te, y_test_binary_mod, positive="X1"))),2)
te_prob <- predict(weighted_fit, newdata=x_test, type="prob")
result_table[i,'AUC_score_TE'] <- round(as.numeric(auc(roc(y_test_binary, te_prob[[2]]))),2)

# external test set prediction

ext_test_set <- fread('avg_hybrid_descr_independent_test_set.csv')
# class
ext_test_set_class <- ifelse(ext_test_set$TOXICITY == "Hepatotoxic", 1, 0)

# replace '-' by underscore
colnames(ext_test_set) <- gsub('\\-','_',colnames(ext_test_set))

# extract descr 
ext_test_set_descr <- ext_test_set[,names(mean_train), with=F]
# standardize them using mean_train and std_train
ext_test_set_descr <- data.table(t(apply(ext_test_set_descr, 1, function(x) x - mean_train))) # centering
ext_test_set_descr <- data.table(t(apply(ext_test_set_descr, 1, function(x) x / std_train))) # scaling

# external predictions
pred_ext = predict(weighted_fit, ext_test_set_descr)
y_test_ext_binary_mod <- as.factor(ext_test_set_class)
levels(y_test_ext_binary_mod) <- make.names(levels(factor(y_test_ext_binary_mod)))
ext_res <- confusionMatrix(data = pred_ext, reference = y_test_ext_binary_mod, positive = 'X1')

result_table[i,'TN_EXT'] <- as.numeric(ext_res$table[1])
result_table[i,'FP_EXT'] <- as.numeric(ext_res$table[2])
result_table[i,'FN_EXT'] <- as.numeric(ext_res$table[3])
result_table[i,'TP_EXT'] <- as.numeric(ext_res$table[4])
result_table[i,'ACC_EXT'] <- round(as.numeric(ext_res$overall['Accuracy']),2)
result_table[i,'BA_EXT'] <- round(as.numeric(ext_res$byClass['Balanced Accuracy']),2)
result_table[i,'SN_EXT'] <- round(as.numeric(ext_res$byClass['Sensitivity']),2)
result_table[i,'SP_EXT'] <- round(as.numeric(ext_res$byClass['Specificity']),2)
result_table[i,'PPV_EXT'] <- round(as.numeric(ext_res$byClass['Pos Pred Value']),2)
result_table[i,'NPV_EXT'] <- round(as.numeric(ext_res$byClass['Neg Pred Value']),2)
result_table[i,'RC_EXT'] <- round(as.numeric(sensitivity(pred_ext, y_test_ext_binary_mod, positive="X1")),2)
result_table[i,'F1_EXT'] <- round(as.numeric((2 * posPredValue(pred_ext, y_test_ext_binary_mod, positive="X1") * sensitivity(pred_ext, y_test_ext_binary_mod, positive="X1")) / (posPredValue(pred_ext, y_test_ext_binary_mod, positive="X1") + sensitivity(pred_ext, y_test_ext_binary_mod, positive="X1"))),2)
ext_prob <- predict(weighted_fit, newdata=ext_test_set_descr, type="prob")
result_table[i,'AUC_score_EXT'] <- round(as.numeric(auc(roc(ext_test_set_class, ext_prob[[2]]))),2)

write.table(result_table,file = 'Results_chemical_NoHS_FS.csv', row.names = F, col.names = T, sep = ',')

# Var importance
ImpMeasure <- varImp(weighted_fit, scale = FALSE)

library(randomForest)
fm <- weighted_fit$finalModel
var_imp_details <- data.frame(randomForest::importance(fm, scale = FALSE))
var_imp_details$Vars <- row.names(var_imp_details)
var_imp_details <- var_imp_details[,c(5,1:4)]
var_imp_details <- var_imp_details[order(var_imp_details$MeanDecreaseGini, decreasing = T),]
write.table(var_imp_details,file = 'Imp_variables.csv', row.names = F, col.names = T, sep = ',')

# save model
mod_nm <- paste('chemical_NoHS_FS','.Rdata',sep = '')
save.image(file=mod_nm)


# 3. Model building with hybrid descr

rm(list = setdiff(ls(),'path'))
setwd(path)

x_data <- fread('input_hybrid.csv')
# to save all results
result_table <- matrix(NA,nrow = 1, ncol = 54)
colnames(result_table) <- c('Model','var_count','TN','FP','FN','TP','ACC','BA','SN','SP','PPV','NPV','RC','F1','AUC_score',
	'TN_CV','FP_CV','FN_CV','TP_CV','ACC_CV','BA_CV','SN_CV','SP_CV','PPV_CV','NPV_CV','RC_CV','F1_CV','AUC_score_CV',
	'TN_TE','FP_TE','FN_TE','TP_TE','ACC_TE','BA_TE','SN_TE','SP_TE','PPV_TE','NPV_TE','RC_TE','F1_TE','AUC_score_TE',
	'TN_EXT','FP_EXT','FN_EXT','TP_EXT','ACC_EXT','BA_EXT','SN_EXT','SP_EXT','PPV_EXT','NPV_EXT','RC_EXT','F1_EXT','AUC_score_EXT')

set.seed(47)
# Data splitting in training and test set
x_train_orig <- stratified(x_data, c('CLASS'), 0.8)
x_test_orig <- setdiff(x_data,x_train_orig)

# save train and test set

new_folder <- paste(path,'/hybrid_NoHS_FS',sep = '')
dir.create(new_folder)
setwd(new_folder)
write.table(x_train_orig,file = 'Orig_train_set.csv', row.names = F, col.names = T, sep = ',')
write.table(x_test_orig,file = 'Orig_test_set.csv', row.names = F, col.names = T, sep = ',')

# pre-processing train set 
x_train <- as.data.frame(x_train_orig[,!c('Sample_id', 'CLASS')])

# save the MEAN & STD. of x_train, later will be used for normalization of test set
mean_train <- apply(x_train, 2, mean, na.rm = TRUE)
std_train <- apply(x_train, 2, sd, na.rm = TRUE)
write.table(mean_train,file = 'Mean_train_set.csv', row.names = F, col.names = T, sep = ',')
write.table(std_train,file = 'STD_train_set.csv', row.names = F, col.names = T, sep = ',')

# center
x_train <- scale(x_train, center = T, scale = F)
# scale
x_train <- scale(x_train, center = FALSE, scale = apply(x_train, 2, sd, na.rm = TRUE))

# find significant descriptors
library(Boruta)

set.seed(47)

FS_boruta_model <- Boruta(x_train, x_train_orig$CLASS, pValue = 0.05, mcAdj = TRUE, maxRuns = 100)
best_var <- colnames(x_train)[which(FS_boruta_model$finalDecision == 'Confirmed')]

# final train & test set
x_train <- x_train[,best_var]

# pre-processing test set
x_test <- as.data.frame(x_test_orig[,!c('Sample_id', 'CLASS')])
# scale and center using mean and std of train set
x_test <- x_test[,names(mean_train)]        # Arrange columns as same as 'mean_train' dataframe
x_test <- data.table(t(apply(x_test, 1, function(x) x - mean_train))) # centering
x_test <- data.table(t(apply(x_test, 1, function(x) x / std_train))) # scaling
x_test <- x_test[,best_var,with=F]
y_train_binary <- x_train_orig$CLASS
y_test_binary <- x_test_orig$CLASS

# save normalized train and test set
write.table(cbind(x_train_orig[,1:2],x_train),file = 'train_set_norm.csv', row.names = F, col.names = T, sep = ',')
write.table(cbind(x_test_orig[,1:2],x_test),file = 'test_set_norm.csv', row.names = F, col.names = T, sep = ',')

x_train_1 <- as.data.frame(cbind(y_train_binary,x_train))
x_train_1$y_train_binary <- as.factor(x_train_1$y_train_binary)
levels(x_train_1$y_train_binary) <- make.names(levels(factor(x_train_1$y_train_binary)))

# RF model building
model_weights <- ifelse(y_train_binary=='0', (1/table(y_train_binary)[1]) * 0.5, (1/table(y_train_binary)[2]) * 0.5)
seeds <- as.vector(c(1:6), mode = "list")
seeds[[6]] <- 1
mtry_val = as.integer(sqrt(dim(x_train_1)[2]))

set.seed(47)

library(doMC)
registerDoMC(cores = 32)

weighted_fit <- caret::train(y_train_binary ~ .,
	data = x_train_1,
	method = "rf",
	importance = TRUE,
	verbose = FALSE,
	ntree=100,
	replace=FALSE, 
	nodesize = 5,
	metric = "Accuracy",
	tuneGrid = data.frame(mtry = mtry_val),
	tuneLength = 1,
	weights = model_weights,
	trControl = trainControl(method = "cv", number = 5, classProbs=T, savePredictions = "final", seeds = seeds))

# training pred
pred_tr = predict(weighted_fit, x_train)
tr_res <- confusionMatrix(data = pred_tr, reference = x_train_1$y_train_binary, positive = 'X1')

i <- 1
result_table[i,'Model'] <- 'Hybrid_NoHS_FS'
result_table[i,'var_count'] <- c(length(best_var))
result_table[i,'TN'] <- as.numeric(tr_res$table[1])
result_table[i,'FP'] <- as.numeric(tr_res$table[2])
result_table[i,'FN'] <- as.numeric(tr_res$table[3])
result_table[i,'TP'] <- as.numeric(tr_res$table[4])
result_table[i,'ACC'] <- as.numeric(tr_res$overall['Accuracy'])
result_table[i,'BA'] <- as.numeric(tr_res$byClass['Balanced Accuracy'])
result_table[i,'SN'] <- as.numeric(tr_res$byClass['Sensitivity'])
result_table[i,'SP'] <- as.numeric(tr_res$byClass['Specificity'])
result_table[i,'PPV'] <- as.numeric(tr_res$byClass['Pos Pred Value'])
result_table[i,'NPV'] <- as.numeric(tr_res$byClass['Neg Pred Value'])
result_table[i,'RC'] <- as.numeric(sensitivity(pred_tr, x_train_1$y_train_binary, positive="X1"))
result_table[i,'F1'] <- as.numeric((2 * posPredValue(pred_tr, x_train_1$y_train_binary, positive="X1") * sensitivity(pred_tr, x_train_1$y_train_binary, positive="X1")) / (posPredValue(pred_tr, x_train_1$y_train_binary, positive="X1") + sensitivity(pred_tr, x_train_1$y_train_binary, positive="X1")))
tr_prob <- predict(weighted_fit, newdata=x_train_1, type="prob")
roc_obj <- roc(y_train_binary, tr_prob[[2]])
result_table[i,'AUC_score'] <- as.numeric(auc(roc_obj))

# CV output extraction
all_folds <- weighted_fit$pred
CV_results <- matrix(NA, nrow = 5, ncol = 14)
colnames(CV_results) <- c('Fold','TN','FP','FN','TP','ACC','BA','SN','SP','PPV','NPV','RC','F1','AUC_score')
for(j in 1:5){
	A <- paste('Fold',j,sep = '')
	B <- all_folds[all_folds$Resample == A,]
	CV_results[j,'Fold'] <- j
	CV_results[j,'TN'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[1])
	CV_results[j,'FP'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[2])
	CV_results[j,'FN'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[3])
	CV_results[j,'TP'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[4])
	CV_results[j,'ACC'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$overall['Accuracy'])
	CV_results[j,'BA'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Balanced Accuracy'])
	CV_results[j,'SN'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Sensitivity'])
	CV_results[j,'SP'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Specificity'])
	CV_results[j,'PPV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Pos Pred Value'])
	CV_results[j,'NPV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Neg Pred Value'])
	CV_results[j,'RC'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Recall'])
	CV_results[j,'F1'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['F1'])
	CV_results[j,'AUC_score'] <- as.numeric(auc(roc(B$obs, B$X1)))
}

CV_results_final <- colMeans(CV_results[,2:14], na.rm=TRUE)
CV_results_final[1:4] <- round(CV_results_final[1:4], 0)
CV_results_final[5:13] <- round(CV_results_final[5:13], 2)
# add to final result
result_table[i,16:28] <- CV_results_final

# test pred
pred_te = predict(weighted_fit, x_test)
y_test_binary_mod <- as.factor(y_test_binary)
levels(y_test_binary_mod) <- make.names(levels(factor(y_test_binary_mod)))
te_res <- confusionMatrix(data = pred_te, reference = y_test_binary_mod, positive = 'X1')

result_table[i,'TN_TE'] <- as.numeric(te_res$table[1])
result_table[i,'FP_TE'] <- as.numeric(te_res$table[2])
result_table[i,'FN_TE'] <- as.numeric(te_res$table[3])
result_table[i,'TP_TE'] <- as.numeric(te_res$table[4])
result_table[i,'ACC_TE'] <- round(as.numeric(te_res$overall['Accuracy']),2)
result_table[i,'BA_TE'] <- round(as.numeric(te_res$byClass['Balanced Accuracy']),2)
result_table[i,'SN_TE'] <- round(as.numeric(te_res$byClass['Sensitivity']),2)
result_table[i,'SP_TE'] <- round(as.numeric(te_res$byClass['Specificity']),2)
result_table[i,'PPV_TE'] <- round(as.numeric(te_res$byClass['Pos Pred Value']),2)
result_table[i,'NPV_TE'] <- round(as.numeric(te_res$byClass['Neg Pred Value']),2)
result_table[i,'RC_TE'] <- round(as.numeric(sensitivity(pred_te, y_test_binary_mod, positive="X1")),2)
result_table[i,'F1_TE'] <- round(as.numeric((2 * posPredValue(pred_te, y_test_binary_mod, positive="X1") * sensitivity(pred_te, y_test_binary_mod, positive="X1")) / (posPredValue(pred_te, y_test_binary_mod, positive="X1") + sensitivity(pred_te, y_test_binary_mod, positive="X1"))),2)
te_prob <- predict(weighted_fit, newdata=x_test, type="prob")
result_table[i,'AUC_score_TE'] <- round(as.numeric(auc(roc(y_test_binary, te_prob[[2]]))),2)

# external test set prediction

ext_test_set <- fread('avg_hybrid_descr_independent_test_set.csv')
# class
ext_test_set_class <- ifelse(ext_test_set$TOXICITY == "Hepatotoxic", 1, 0)

# replace '-' by underscore
colnames(ext_test_set) <- gsub('\\-','_',colnames(ext_test_set))

# extract descr 
ext_test_set_descr <- ext_test_set[,names(mean_train), with=F]
# standardize them using mean_train and std_train
ext_test_set_descr <- data.table(t(apply(ext_test_set_descr, 1, function(x) x - mean_train))) # centering
ext_test_set_descr <- data.table(t(apply(ext_test_set_descr, 1, function(x) x / std_train))) # scaling

# external predictions
pred_ext = predict(weighted_fit, ext_test_set_descr)
y_test_ext_binary_mod <- as.factor(ext_test_set_class)
levels(y_test_ext_binary_mod) <- make.names(levels(factor(y_test_ext_binary_mod)))
ext_res <- confusionMatrix(data = pred_ext, reference = y_test_ext_binary_mod, positive = 'X1')

result_table[i,'TN_EXT'] <- as.numeric(ext_res$table[1])
result_table[i,'FP_EXT'] <- as.numeric(ext_res$table[2])
result_table[i,'FN_EXT'] <- as.numeric(ext_res$table[3])
result_table[i,'TP_EXT'] <- as.numeric(ext_res$table[4])
result_table[i,'ACC_EXT'] <- round(as.numeric(ext_res$overall['Accuracy']),2)
result_table[i,'BA_EXT'] <- round(as.numeric(ext_res$byClass['Balanced Accuracy']),2)
result_table[i,'SN_EXT'] <- round(as.numeric(ext_res$byClass['Sensitivity']),2)
result_table[i,'SP_EXT'] <- round(as.numeric(ext_res$byClass['Specificity']),2)
result_table[i,'PPV_EXT'] <- round(as.numeric(ext_res$byClass['Pos Pred Value']),2)
result_table[i,'NPV_EXT'] <- round(as.numeric(ext_res$byClass['Neg Pred Value']),2)
result_table[i,'RC_EXT'] <- round(as.numeric(sensitivity(pred_ext, y_test_ext_binary_mod, positive="X1")),2)
result_table[i,'F1_EXT'] <- round(as.numeric((2 * posPredValue(pred_ext, y_test_ext_binary_mod, positive="X1") * sensitivity(pred_ext, y_test_ext_binary_mod, positive="X1")) / (posPredValue(pred_ext, y_test_ext_binary_mod, positive="X1") + sensitivity(pred_ext, y_test_ext_binary_mod, positive="X1"))),2)
ext_prob <- predict(weighted_fit, newdata=ext_test_set_descr, type="prob")
result_table[i,'AUC_score_EXT'] <- round(as.numeric(auc(roc(ext_test_set_class, ext_prob[[2]]))),2)

write.table(result_table,file = 'Results_Hybrid_NoHS_FS.csv', row.names = F, col.names = T, sep = ',')

# Var importance
ImpMeasure <- varImp(weighted_fit, scale = FALSE)

library(randomForest)
fm <- weighted_fit$finalModel
var_imp_details <- data.frame(randomForest::importance(fm, scale = FALSE))
var_imp_details$Vars <- row.names(var_imp_details)
var_imp_details <- var_imp_details[,c(5,1:4)]
var_imp_details <- var_imp_details[order(var_imp_details$MeanDecreaseGini, decreasing = T),]
write.table(var_imp_details,file = 'Imp_variables.csv', row.names = F, col.names = T, sep = ',')

# save model
mod_nm <- paste('Hybrid_NoHS_FS','.Rdata',sep = '')
save.image(file=mod_nm)



# ---------------------------------------------------------------------------------------------------
#                                 Atom-pair & MACCS FP based models 
# ---------------------------------------------------------------------------------------------------


# ----------------------------- Apporach-A: NoHS_NoFS ------------------------------------------------

# 1. FP based model

rm(list = setdiff(ls(),'path'))
setwd(path)

# Load FDA list 346 chemicals with their FP data

avg_variables <- fread('MACCS_Atompair_FP_346_chem.csv')
chem_info <- fread('all_346_chem_info.csv')
x_data <- avg_variables[,-'Name']
classes_346 <- fread('avg_data_class.csv')

# add CLASS column
x_data[, Sample_id := chem_info$Sample_id]
x_data[, CLASS := classes_346$CLASSES]
x_data <- x_data[,c(947,948,1:946),with=F]

setwd(path)
write.table(x_data, 'input_chemical_FP.csv', row.names = F, col.names = T, sep = ',')


# RF model construction: MACCS + Atompair FP model
rm(list = setdiff(ls(),'path'))
x_data <- fread('input_chemical_FP.csv')

set.seed(47)
# Data splitting in training and test set
x_train_orig <- stratified(x_data, c('CLASS'), 0.8)
x_test_orig <- setdiff(x_data,x_train_orig)

# save train and test set
new_folder <- paste(path,'/MACCS_Atompair_NoHS_NoFS',sep = '')
dir.create(new_folder)
setwd(new_folder)

write.table(x_train_orig,file = 'train_set.csv', row.names = F, col.names = T, sep = ',')
write.table(x_test_orig,file = 'test_set.csv', row.names = F, col.names = T, sep = ',')

result_table <- matrix(NA,nrow = 1, ncol = 53)
colnames(result_table) <- c('Model','TN','FP','FN','TP','ACC','BA','SN','SP','PPV','NPV','RC','F1','AUC_score',
	'TN_CV','FP_CV','FN_CV','TP_CV','ACC_CV','BA_CV','SN_CV','SP_CV','PPV_CV','NPV_CV','RC_CV','F1_CV','AUC_score_CV',
	'TN_TE','FP_TE','FN_TE','TP_TE','ACC_TE','BA_TE','SN_TE','SP_TE','PPV_TE','NPV_TE','RC_TE','F1_TE','AUC_score_TE',
	'TN_EXT','FP_EXT','FN_EXT','TP_EXT','ACC_EXT','BA_EXT','SN_EXT','SP_EXT','PPV_EXT','NPV_EXT','RC_EXT','F1_EXT','AUC_score_EXT')


x_train_1 <- as.data.frame(x_train_orig[,-'Sample_id'])
# convert to factors
x_train_1[,2:947] <- lapply(x_train_1[,2:947], as.factor)
x_train_1$CLASS <- as.factor(x_train_1$CLASS)

# drop levels
x_train_1 <- x_train_1[, sapply(x_train_1, nlevels) > 1]
levels(x_train_1$CLASS) <- make.names(levels(factor(x_train_1$CLASS)))

# RF model building
model_weights <- ifelse(x_train_1$CLASS=='0', (1/table(x_train_1$CLASS)[1]) * 0.5, (1/table(x_train_1$CLASS)[2]) * 0.5)
seeds <- as.vector(c(1:6), mode = "list")
seeds[[6]] <- 1
mtry_val = as.integer(sqrt(dim(x_train_1)[2]))

set.seed(47)
weighted_fit <- caret::train(CLASS ~ .,
	data = x_train_1,
	method = "rf",
	verbose = FALSE,
	ntree=100,
	replace=FALSE, 
	nodesize = 5,
	metric = "Accuracy",
	tuneGrid = data.frame(mtry = mtry_val),
	tuneLength = 1,
	weights = model_weights,
	trControl = trainControl(method = "cv", number = 5, classProbs=T, savePredictions = "final", seeds = seeds))

# training pred
pred_tr = predict(weighted_fit, x_train_1[,-1])
tr_res <- confusionMatrix(data = pred_tr, reference = x_train_1$CLASS, positive = 'X1')

i <- 1
result_table[i,'Model'] <- 'AD2D_MACCS_NoHS_NoFS'
result_table[i,'TN'] <- as.numeric(tr_res$table[1])
result_table[i,'FP'] <- as.numeric(tr_res$table[2])
result_table[i,'FN'] <- as.numeric(tr_res$table[3])
result_table[i,'TP'] <- as.numeric(tr_res$table[4])
result_table[i,'ACC'] <- as.numeric(tr_res$overall['Accuracy'])
result_table[i,'BA'] <- as.numeric(tr_res$byClass['Balanced Accuracy'])
result_table[i,'SN'] <- as.numeric(tr_res$byClass['Sensitivity'])
result_table[i,'SP'] <- as.numeric(tr_res$byClass['Specificity'])
result_table[i,'PPV'] <- as.numeric(tr_res$byClass['Pos Pred Value'])
result_table[i,'NPV'] <- as.numeric(tr_res$byClass['Neg Pred Value'])
result_table[i,'RC'] <- as.numeric(sensitivity(pred_tr, x_train_1$CLASS, positive="X1"))
result_table[i,'F1'] <- as.numeric((2 * posPredValue(pred_tr, x_train_1$CLASS, positive="X1") * sensitivity(pred_tr, x_train_1$CLASS, positive="X1")) / (posPredValue(pred_tr, x_train_1$CLASS, positive="X1") + sensitivity(pred_tr, x_train_1$CLASS, positive="X1")))
tr_prob <- predict(weighted_fit, newdata=x_train_1, type="prob")
roc_obj <- roc(x_train_1$CLASS, tr_prob[[2]])
result_table[i,'AUC_score'] <- as.numeric(auc(roc_obj))

# CV output extraction
all_folds <- weighted_fit$pred
CV_results <- matrix(NA, nrow = 5, ncol = 14)
colnames(CV_results) <- c('Fold','TN','FP','FN','TP','ACC','BA','SN','SP','PPV','NPV','RC','F1','AUC_score')
for(j in 1:5){
	A <- paste('Fold',j,sep = '')
	B <- all_folds[all_folds$Resample == A,]
	CV_results[j,'Fold'] <- j
	CV_results[j,'TN'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[1])
	CV_results[j,'FP'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[2])
	CV_results[j,'FN'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[3])
	CV_results[j,'TP'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[4])
	CV_results[j,'ACC'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$overall['Accuracy'])
	CV_results[j,'BA'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Balanced Accuracy'])
	CV_results[j,'SN'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Sensitivity'])
	CV_results[j,'SP'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Specificity'])
	CV_results[j,'PPV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Pos Pred Value'])
	CV_results[j,'NPV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Neg Pred Value'])
	CV_results[j,'RC'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Recall'])
	CV_results[j,'F1'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['F1'])
	CV_results[j,'AUC_score'] <- as.numeric(auc(roc(B$obs, B$X1)))
}

CV_results_final <- colMeans(CV_results[,2:14], na.rm=TRUE)
CV_results_final[1:4] <- round(CV_results_final[1:4], 0)
CV_results_final[5:13] <- round(CV_results_final[5:13], 2)
# add to final result
result_table[i,15:27] <- CV_results_final

# test pred
x_test_1 <- as.data.frame(x_test_orig[,-'Sample_id'])
# convert to factors
x_test_1[,2:947] <- lapply(x_test_1[,2:947], as.factor)
x_test_1$CLASS <- as.factor(x_test_1$CLASS)

# drop levels
x_test_1 <- x_test_1[, names(x_train_1)]

pred_te = predict(weighted_fit, x_test_1[,2:dim(x_test_1)[2]])
levels(x_test_1$CLASS) <- make.names(levels(factor(x_test_1$CLASS)))
te_res <- confusionMatrix(data = pred_te, reference = x_test_1$CLASS, positive = 'X1')

result_table[i,'TN_TE'] <- as.numeric(te_res$table[1])
result_table[i,'FP_TE'] <- as.numeric(te_res$table[2])
result_table[i,'FN_TE'] <- as.numeric(te_res$table[3])
result_table[i,'TP_TE'] <- as.numeric(te_res$table[4])
result_table[i,'ACC_TE'] <- round(as.numeric(te_res$overall['Accuracy']),2)
result_table[i,'BA_TE'] <- round(as.numeric(te_res$byClass['Balanced Accuracy']),2)
result_table[i,'SN_TE'] <- round(as.numeric(te_res$byClass['Sensitivity']),2)
result_table[i,'SP_TE'] <- round(as.numeric(te_res$byClass['Specificity']),2)
result_table[i,'PPV_TE'] <- round(as.numeric(te_res$byClass['Pos Pred Value']),2)
result_table[i,'NPV_TE'] <- round(as.numeric(te_res$byClass['Neg Pred Value']),2)
result_table[i,'RC_TE'] <- round(as.numeric(sensitivity(pred_te, x_test_1$CLASS, positive="X1")),2)
result_table[i,'F1_TE'] <- round(as.numeric((2 * posPredValue(pred_te, x_test_1$CLASS, positive="X1") * sensitivity(pred_te, x_test_1$CLASS, positive="X1")) / (posPredValue(pred_te, x_test_1$CLASS, positive="X1") + sensitivity(pred_te, x_test_1$CLASS, positive="X1"))),2)
te_prob <- predict(weighted_fit, newdata=x_test_1, type="prob")
result_table[i,'AUC_score_TE'] <- round(as.numeric(auc(roc(x_test_1$CLASS, te_prob[[2]]))),2)

# external test set prediction

ext_test_set <- fread('AD2D_MACCS_external_set.csv')
ext_test_set <- ext_test_set[,names(x_train_1)[2:dim(x_test_1)[2]], with=F]
ext_test_set_descr <- ext_test_set[,lapply(ext_test_set, as.factor)]

# class
ext_test_set_class <- fread('external_test_set_class.txt')

ext_test_set_class <- ifelse(ext_test_set_class$TOXICITY == "Hepatotoxic", 1, 0)

# external predictions
pred_ext = predict(weighted_fit, ext_test_set_descr)
y_test_ext_binary_mod <- as.factor(ext_test_set_class)
levels(y_test_ext_binary_mod) <- make.names(levels(factor(y_test_ext_binary_mod)))
ext_res <- confusionMatrix(data = pred_ext, reference = y_test_ext_binary_mod, positive = 'X1')

result_table[i,'TN_EXT'] <- as.numeric(ext_res$table[1])
result_table[i,'FP_EXT'] <- as.numeric(ext_res$table[2])
result_table[i,'FN_EXT'] <- as.numeric(ext_res$table[3])
result_table[i,'TP_EXT'] <- as.numeric(ext_res$table[4])
result_table[i,'ACC_EXT'] <- round(as.numeric(ext_res$overall['Accuracy']),2)
result_table[i,'BA_EXT'] <- round(as.numeric(ext_res$byClass['Balanced Accuracy']),2)
result_table[i,'SN_EXT'] <- round(as.numeric(ext_res$byClass['Sensitivity']),2)
result_table[i,'SP_EXT'] <- round(as.numeric(ext_res$byClass['Specificity']),2)
result_table[i,'PPV_EXT'] <- round(as.numeric(ext_res$byClass['Pos Pred Value']),2)
result_table[i,'NPV_EXT'] <- round(as.numeric(ext_res$byClass['Neg Pred Value']),2)
result_table[i,'RC_EXT'] <- round(as.numeric(sensitivity(pred_ext, y_test_ext_binary_mod, positive="X1")),2)
result_table[i,'F1_EXT'] <- round(as.numeric((2 * posPredValue(pred_ext, y_test_ext_binary_mod, positive="X1") * sensitivity(pred_ext, y_test_ext_binary_mod, positive="X1")) / (posPredValue(pred_ext, y_test_ext_binary_mod, positive="X1") + sensitivity(pred_ext, y_test_ext_binary_mod, positive="X1"))),2)
ext_prob <- predict(weighted_fit, newdata=ext_test_set_descr, type="prob")
result_table[i,'AUC_score_EXT'] <- round(as.numeric(auc(roc(ext_test_set_class, ext_prob[[2]]))),2)

write.table(result_table,file = 'Results_AD2D_MACCS_NoHS_NoFS.csv', row.names = F, col.names = T, sep = ',')

# Var importance
ImpMeasure <- varImp(weighted_fit, scale = FALSE)

library(randomForest)
fm <- weighted_fit$finalModel
var_imp_details <- data.frame(randomForest::importance(fm, scale = FALSE))
var_imp_details$Vars <- row.names(var_imp_details)
var_imp_details <- var_imp_details[,c(5,1:4)]
var_imp_details <- var_imp_details[order(var_imp_details$MeanDecreaseGini, decreasing = T),]
write.table(var_imp_details,file = 'Imp_variables.csv', row.names = F, col.names = T, sep = ',')

# save model
mod_nm <- paste('AD2D_MACCS_FP_Model_NoHS_NoFS', '.Rdata',sep = '')
save.image(file=mod_nm)


# 2. Hybrid (FP + HCI descr) based model

rm(list = setdiff(ls(),'path'))
setwd(path)

# Load FDA list 346 chemicals with their HCI data
avg_variables <- fread('avg_variables_346_inst_table.csv')

# keep only HCI phenotypes
hci_var_names <- names(avg_variables)[grep('Cells|Cytoplasm|Nuclei',names(avg_variables))]
x_data_hci <- avg_variables[,hci_var_names, with=F]

# Load FDA list 346 chemicals with their FP data
avg_variables <- fread('MACCS_Atompair_FP_346_chem.csv')
chem_info <- fread('all_346_chem_info.csv')
x_data_FP <- avg_variables[,-'Name']
# drop levels
x_data_FP <- x_data_FP[,lapply(x_data_FP, as.factor)]
x_data_FP <- x_data_FP[, sapply(x_data_FP, nlevels) > 1, with=F]
# load classes
classes_346 <- fread('avg_data_class.csv')

# add CLASS column
x_data <- cbind(x_data_hci,x_data_FP)
x_data[, Sample_id := chem_info$Sample_id]
x_data[, CLASS := classes_346$CLASSES]
x_data <- x_data[,c(2128,2129,1:2127),with=F]

write.table(x_data, 'input_HCI_FP.csv', row.names = F, col.names = T, sep = ',')


# RF model construction: MACCS + Atompair FP model
rm(list = setdiff(ls(),'path'))
x_data <- fread('input_HCI_FP.csv')

set.seed(47)
# Data splitting in training and test set
x_train_orig <- stratified(x_data, c('CLASS'), 0.8)
x_test_orig <- setdiff(x_data,x_train_orig)

# save train and test set
new_folder <- paste(path,'/HCI_MACCS_Atompair_NoHS_NoFS',sep = '')
dir.create(new_folder)
setwd(new_folder)

write.table(x_train_orig,file = 'train_set.csv', row.names = F, col.names = T, sep = ',')
write.table(x_test_orig,file = 'test_set.csv', row.names = F, col.names = T, sep = ',')

result_table <- matrix(NA,nrow = 1, ncol = 53)
colnames(result_table) <- c('Model','TN','FP','FN','TP','ACC','BA','SN','SP','PPV','NPV','RC','F1','AUC_score',
	'TN_CV','FP_CV','FN_CV','TP_CV','ACC_CV','BA_CV','SN_CV','SP_CV','PPV_CV','NPV_CV','RC_CV','F1_CV','AUC_score_CV',
	'TN_TE','FP_TE','FN_TE','TP_TE','ACC_TE','BA_TE','SN_TE','SP_TE','PPV_TE','NPV_TE','RC_TE','F1_TE','AUC_score_TE',
	'TN_EXT','FP_EXT','FN_EXT','TP_EXT','ACC_EXT','BA_EXT','SN_EXT','SP_EXT','PPV_EXT','NPV_EXT','RC_EXT','F1_EXT','AUC_score_EXT')


x_train_1 <- as.data.frame(x_train_orig[,-'Sample_id'])
# convert to factors

x_train_1$CLASS <- as.factor(x_train_1$CLASS)
levels(x_train_1$CLASS) <- make.names(levels(factor(x_train_1$CLASS)))
#x_train_1[,1722:2128] <- lapply(x_train_1[,1722:2128], as.factor)

# RF model building
model_weights <- ifelse(x_train_1$CLASS=='0', (1/table(x_train_1$CLASS)[1]) * 0.5, (1/table(x_train_1$CLASS)[2]) * 0.5)
seeds <- as.vector(c(1:6), mode = "list")
seeds[[6]] <- 1
mtry_val = as.integer(sqrt(dim(x_train_1)[2]))

set.seed(47)
weighted_fit <- caret::train(CLASS ~ .,
	data = x_train_1,
	method = "rf",
	verbose = FALSE,
	ntree=100,
	replace=FALSE, 
	nodesize = 5,
	metric = "Accuracy",
	tuneGrid = data.frame(mtry = mtry_val),
	tuneLength = 1,
	weights = model_weights,
	trControl = trainControl(method = "cv", number = 5, classProbs=T, savePredictions = "final", seeds = seeds))

# training pred
pred_tr = predict(weighted_fit, x_train_1[,-1])
tr_res <- confusionMatrix(data = pred_tr, reference = x_train_1$CLASS, positive = 'X1')

i <- 1
result_table[i,'Model'] <- 'HCI_AD2D_MACCS_NoHS_NoFS'
result_table[i,'TN'] <- as.numeric(tr_res$table[1])
result_table[i,'FP'] <- as.numeric(tr_res$table[2])
result_table[i,'FN'] <- as.numeric(tr_res$table[3])
result_table[i,'TP'] <- as.numeric(tr_res$table[4])
result_table[i,'ACC'] <- as.numeric(tr_res$overall['Accuracy'])
result_table[i,'BA'] <- as.numeric(tr_res$byClass['Balanced Accuracy'])
result_table[i,'SN'] <- as.numeric(tr_res$byClass['Sensitivity'])
result_table[i,'SP'] <- as.numeric(tr_res$byClass['Specificity'])
result_table[i,'PPV'] <- as.numeric(tr_res$byClass['Pos Pred Value'])
result_table[i,'NPV'] <- as.numeric(tr_res$byClass['Neg Pred Value'])
result_table[i,'RC'] <- as.numeric(sensitivity(pred_tr, x_train_1$CLASS, positive="X1"))
result_table[i,'F1'] <- as.numeric((2 * posPredValue(pred_tr, x_train_1$CLASS, positive="X1") * sensitivity(pred_tr, x_train_1$CLASS, positive="X1")) / (posPredValue(pred_tr, x_train_1$CLASS, positive="X1") + sensitivity(pred_tr, x_train_1$CLASS, positive="X1")))
tr_prob <- predict(weighted_fit, newdata=x_train_1, type="prob")
roc_obj <- roc(x_train_1$CLASS, tr_prob[[2]])
result_table[i,'AUC_score'] <- as.numeric(auc(roc_obj))

# CV output extraction
all_folds <- weighted_fit$pred
CV_results <- matrix(NA, nrow = 5, ncol = 14)
colnames(CV_results) <- c('Fold','TN','FP','FN','TP','ACC','BA','SN','SP','PPV','NPV','RC','F1','AUC_score')
for(j in 1:5){
	A <- paste('Fold',j,sep = '')
	B <- all_folds[all_folds$Resample == A,]
	CV_results[j,'Fold'] <- j
	CV_results[j,'TN'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[1])
	CV_results[j,'FP'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[2])
	CV_results[j,'FN'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[3])
	CV_results[j,'TP'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[4])
	CV_results[j,'ACC'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$overall['Accuracy'])
	CV_results[j,'BA'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Balanced Accuracy'])
	CV_results[j,'SN'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Sensitivity'])
	CV_results[j,'SP'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Specificity'])
	CV_results[j,'PPV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Pos Pred Value'])
	CV_results[j,'NPV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Neg Pred Value'])
	CV_results[j,'RC'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Recall'])
	CV_results[j,'F1'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['F1'])
	CV_results[j,'AUC_score'] <- as.numeric(auc(roc(B$obs, B$X1)))
}

CV_results_final <- colMeans(CV_results[,2:14], na.rm=TRUE)
CV_results_final[1:4] <- round(CV_results_final[1:4], 0)
CV_results_final[5:13] <- round(CV_results_final[5:13], 2)
# add to final result
result_table[i,15:27] <- CV_results_final

# test pred
x_test_1 <- as.data.frame(x_test_orig[,-'Sample_id'])
x_test_1$CLASS <- as.factor(x_test_1$CLASS)

# drop levels
x_test_1 <- x_test_1[, names(x_train_1)]

pred_te = predict(weighted_fit, x_test_1[,2:dim(x_test_1)[2]])
levels(x_test_1$CLASS) <- make.names(levels(factor(x_test_1$CLASS)))
te_res <- confusionMatrix(data = pred_te, reference = x_test_1$CLASS, positive = 'X1')

result_table[i,'TN_TE'] <- as.numeric(te_res$table[1])
result_table[i,'FP_TE'] <- as.numeric(te_res$table[2])
result_table[i,'FN_TE'] <- as.numeric(te_res$table[3])
result_table[i,'TP_TE'] <- as.numeric(te_res$table[4])
result_table[i,'ACC_TE'] <- round(as.numeric(te_res$overall['Accuracy']),2)
result_table[i,'BA_TE'] <- round(as.numeric(te_res$byClass['Balanced Accuracy']),2)
result_table[i,'SN_TE'] <- round(as.numeric(te_res$byClass['Sensitivity']),2)
result_table[i,'SP_TE'] <- round(as.numeric(te_res$byClass['Specificity']),2)
result_table[i,'PPV_TE'] <- round(as.numeric(te_res$byClass['Pos Pred Value']),2)
result_table[i,'NPV_TE'] <- round(as.numeric(te_res$byClass['Neg Pred Value']),2)
result_table[i,'RC_TE'] <- round(as.numeric(sensitivity(pred_te, x_test_1$CLASS, positive="X1")),2)
result_table[i,'F1_TE'] <- round(as.numeric((2 * posPredValue(pred_te, x_test_1$CLASS, positive="X1") * sensitivity(pred_te, x_test_1$CLASS, positive="X1")) / (posPredValue(pred_te, x_test_1$CLASS, positive="X1") + sensitivity(pred_te, x_test_1$CLASS, positive="X1"))),2)
te_prob <- predict(weighted_fit, newdata=x_test_1, type="prob")
result_table[i,'AUC_score_TE'] <- round(as.numeric(auc(roc(x_test_1$CLASS, te_prob[[2]]))),2)

# external test set prediction
ext_test_set_1 <- fread('avg_hybrid_descr_independent_test_set.csv')
ext_test_set_2 <- fread('AD2D_MACCS_external_set.csv')
ext_test_set <- cbind(ext_test_set_1,ext_test_set_2)
ext_test_set <- ext_test_set[,names(x_test_1)[2:dim(x_test_1)[2]], with=F]
# class
ext_test_set_class <- fread('external_test_set_class.txt')
ext_test_set_class <- ifelse(ext_test_set_class$TOXICITY == "Hepatotoxic", 1, 0)

# external predictions
pred_ext = predict(weighted_fit, ext_test_set)
y_test_ext_binary_mod <- as.factor(ext_test_set_class)
levels(y_test_ext_binary_mod) <- make.names(levels(factor(y_test_ext_binary_mod)))
ext_res <- confusionMatrix(data = pred_ext, reference = y_test_ext_binary_mod, positive = 'X1')

result_table[i,'TN_EXT'] <- as.numeric(ext_res$table[1])
result_table[i,'FP_EXT'] <- as.numeric(ext_res$table[2])
result_table[i,'FN_EXT'] <- as.numeric(ext_res$table[3])
result_table[i,'TP_EXT'] <- as.numeric(ext_res$table[4])
result_table[i,'ACC_EXT'] <- round(as.numeric(ext_res$overall['Accuracy']),2)
result_table[i,'BA_EXT'] <- round(as.numeric(ext_res$byClass['Balanced Accuracy']),2)
result_table[i,'SN_EXT'] <- round(as.numeric(ext_res$byClass['Sensitivity']),2)
result_table[i,'SP_EXT'] <- round(as.numeric(ext_res$byClass['Specificity']),2)
result_table[i,'PPV_EXT'] <- round(as.numeric(ext_res$byClass['Pos Pred Value']),2)
result_table[i,'NPV_EXT'] <- round(as.numeric(ext_res$byClass['Neg Pred Value']),2)
result_table[i,'RC_EXT'] <- round(as.numeric(sensitivity(pred_ext, y_test_ext_binary_mod, positive="X1")),2)
result_table[i,'F1_EXT'] <- round(as.numeric((2 * posPredValue(pred_ext, y_test_ext_binary_mod, positive="X1") * sensitivity(pred_ext, y_test_ext_binary_mod, positive="X1")) / (posPredValue(pred_ext, y_test_ext_binary_mod, positive="X1") + sensitivity(pred_ext, y_test_ext_binary_mod, positive="X1"))),2)
ext_prob <- predict(weighted_fit, newdata=ext_test_set, type="prob")
result_table[i,'AUC_score_EXT'] <- round(as.numeric(auc(roc(ext_test_set_class, ext_prob[[2]]))),2)

write.table(result_table,file = 'Results_HCI_AD2D_MACCS_NoHS_NoFS.csv', row.names = F, col.names = T, sep = ',')

# Var importance
ImpMeasure <- varImp(weighted_fit, scale = FALSE)

library(randomForest)
fm <- weighted_fit$finalModel
var_imp_details <- data.frame(randomForest::importance(fm, scale = FALSE))
var_imp_details$Vars <- row.names(var_imp_details)
var_imp_details <- var_imp_details[,c(5,1:4)]
var_imp_details <- var_imp_details[order(var_imp_details$MeanDecreaseGini, decreasing = T),]
write.table(var_imp_details,file = 'Imp_variables.csv', row.names = F, col.names = T, sep = ',')

# save model
mod_nm <- paste('HCI_AD2D_MACCS_FP_Model_NoHS_NoFS', '.Rdata',sep = '')
save.image(file=mod_nm)

# ---------------------------- Approach-B: HS_NoFS ---------------------------------------------------

# 1. FP 

rm(list = setdiff(ls(),'path'))
setwd(path)

# Load FDA list 346 chemicals with their FP data

avg_variables <- fread('MACCS_Atompair_FP_346_chem.csv')
chem_info <- fread('all_346_chem_info.csv')
x_data <- avg_variables[,-'Name']
classes_346 <- fread('avg_data_class.csv')

# add CLASS column
x_data[, Sample_id := chem_info$Sample_id]
x_data[, CLASS := classes_346$CLASSES]
x_data <- x_data[,c(947,948,1:946),with=F]

# RF model construction: MACCS + Atompair FP model

# Data splitting in training and test set
set.seed(47)
x_train_orig <- stratified(x_data, c('CLASS'), 0.8)
x_test_orig <- setdiff(x_data,x_train_orig)

# save train and test set
new_folder <- paste(path,'/MACCS_Atompair_HS_NoFS',sep = '')
dir.create(new_folder)
setwd(new_folder)

write.table(x_train_orig,file = 'train_set.csv', row.names = F, col.names = T, sep = ',')
write.table(x_test_orig,file = 'test_set.csv', row.names = F, col.names = T, sep = ',')

x_train_1 <- as.data.frame(x_train_orig[,-'Sample_id'])
# convert to factors
x_train_1[,2:947] <- lapply(x_train_1[,2:947], as.factor)
x_train_1$CLASS <- as.factor(x_train_1$CLASS)

# drop levels
x_train_1 <- x_train_1[, sapply(x_train_1, nlevels) > 1]
levels(x_train_1$CLASS) <- make.names(levels(factor(x_train_1$CLASS)))

# RF model building
model_weights <- ifelse(x_train_1$CLASS=='0', (1/table(x_train_1$CLASS)[1]) * 0.5, (1/table(x_train_1$CLASS)[2]) * 0.5)
seeds <- as.vector(c(1:6), mode = "list")
seeds[[6]] <- 1
mtry_val = as.integer(sqrt(dim(x_train_1)[2]))

# to save all results
result_table <- matrix(NA,nrow = 1, ncol = 56)
colnames(result_table) <- c('Model','mtry','ntree','nodesize','TN','FP','FN','TP','ACC','BA','SN','SP','PPV','NPV','RC','F1','AUC_score',
	'TN_CV','FP_CV','FN_CV','TP_CV','ACC_CV','BA_CV','SN_CV','SP_CV','PPV_CV','NPV_CV','RC_CV','F1_CV','AUC_score_CV',
	'TN_TE','FP_TE','FN_TE','TP_TE','ACC_TE','BA_TE','SN_TE','SP_TE','PPV_TE','NPV_TE','RC_TE','F1_TE','AUC_score_TE',
	'TN_EXT','FP_EXT','FN_EXT','TP_EXT','ACC_EXT','BA_EXT','SN_EXT','SP_EXT','PPV_EXT','NPV_EXT','RC_EXT','F1_EXT','AUC_score_EXT')

set.seed(47)
customRF <- list(type = "Classification", library = "randomForest", loop = NULL)
customRF$parameters <- data.frame(parameter = c("mtry", "ntree", "nodesize"), class = rep("numeric", 3), label = c("mtry", "ntree", "nodesize"))
customRF$grid <- function(x, y, len = NULL, search = "grid") {}
customRF$fit <- function(x, y, wts, param, lev, last, weights, classProbs, ...) {
	randomForest(x, y, mtry = param$mtry, ntree=param$ntree, ...)
}
customRF$predict <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
	predict(modelFit, newdata)
customRF$prob <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
	predict(modelFit, newdata, type = "prob")
customRF$sort <- function(x) x[order(x[,1]),]
customRF$levels <- function(x) x$classes

library(doMC)
registerDoMC(cores = 24)
set.seed(47)
tunegrid <- data.frame(expand.grid(.mtry=c(10,15,20,25,30), .ntree=c(50, 100,500), .nodesize=c(1:5)))
weighted_fit <- caret::train(CLASS ~ .,
	data = x_train_1,
	method = customRF,
	importance = TRUE,
	verbose = FALSE,
	replace=FALSE,
	metric = "Accuracy",
	tuneGrid = tunegrid,
	weights = model_weights,
	trControl = trainControl(method = "cv", number = 5, classProbs=T, savePredictions = "final", allowParallel = TRUE))

# training pred
pred_tr = predict(weighted_fit, x_train_1[,-1])
tr_res <- confusionMatrix(data = pred_tr, reference = x_train_1$CLASS, positive = 'X1')


# save results
i <- 1
result_table[i,'Model'] <- 'AD2D_MACCS_HS_NoFS'
result_table[i,'mtry'] <- weighted_fit$bestTune$mtry
result_table[i,'ntree'] <- weighted_fit$bestTune$ntree
result_table[i,'nodesize'] <- weighted_fit$bestTune$nodesize

result_table[i,'TN'] <- as.numeric(tr_res$table[1])
result_table[i,'FP'] <- as.numeric(tr_res$table[2])
result_table[i,'FN'] <- as.numeric(tr_res$table[3])
result_table[i,'TP'] <- as.numeric(tr_res$table[4])
result_table[i,'ACC'] <- as.numeric(tr_res$overall['Accuracy'])
result_table[i,'BA'] <- as.numeric(tr_res$byClass['Balanced Accuracy'])
result_table[i,'SN'] <- as.numeric(tr_res$byClass['Sensitivity'])
result_table[i,'SP'] <- as.numeric(tr_res$byClass['Specificity'])
result_table[i,'PPV'] <- as.numeric(tr_res$byClass['Pos Pred Value'])
result_table[i,'NPV'] <- as.numeric(tr_res$byClass['Neg Pred Value'])
result_table[i,'RC'] <- as.numeric(sensitivity(pred_tr, x_train_1$CLASS, positive="X1"))
result_table[i,'F1'] <- as.numeric((2 * posPredValue(pred_tr, x_train_1$CLASS, positive="X1") * sensitivity(pred_tr, x_train_1$CLASS, positive="X1")) / (posPredValue(pred_tr, x_train_1$CLASS, positive="X1") + sensitivity(pred_tr, x_train_1$CLASS, positive="X1")))
tr_prob <- predict(weighted_fit, newdata=x_train_1, type="prob")
roc_obj <- roc(x_train_1$CLASS, tr_prob[[2]])
result_table[i,'AUC_score'] <- as.numeric(auc(roc_obj))

# CV output extraction
all_folds <- weighted_fit$pred
CV_results <- matrix(NA, nrow = 5, ncol = 14)
colnames(CV_results) <- c('Fold','TN','FP','FN','TP','ACC','BA','SN','SP','PPV','NPV','RC','F1','AUC_score')
for(j in 1:5){
	A <- paste('Fold',j,sep = '')
	B <- all_folds[all_folds$Resample == A,]
	CV_results[j,'Fold'] <- j
	CV_results[j,'TN'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[1])
	CV_results[j,'FP'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[2])
	CV_results[j,'FN'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[3])
	CV_results[j,'TP'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[4])
	CV_results[j,'ACC'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$overall['Accuracy'])
	CV_results[j,'BA'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Balanced Accuracy'])
	CV_results[j,'SN'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Sensitivity'])
	CV_results[j,'SP'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Specificity'])
	CV_results[j,'PPV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Pos Pred Value'])
	CV_results[j,'NPV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Neg Pred Value'])
	CV_results[j,'RC'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Recall'])
	CV_results[j,'F1'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['F1'])
	CV_results[j,'AUC_score'] <- as.numeric(auc(roc(B$obs, B$X1)))
}

CV_results_final <- colMeans(CV_results[,2:14], na.rm=TRUE)
CV_results_final[1:4] <- round(CV_results_final[1:4], 0)
CV_results_final[5:13] <- round(CV_results_final[5:13], 2)
# add to final result
result_table[i,18:30] <- CV_results_final

# test pred
x_test_1 <- as.data.frame(x_test_orig[,-'Sample_id'])
# convert to factors
x_test_1[,2:947] <- lapply(x_test_1[,2:947], as.factor)
x_test_1$CLASS <- as.factor(x_test_1$CLASS)

# drop levels
x_test_1 <- x_test_1[, names(x_train_1)]

pred_te = predict(weighted_fit, x_test_1[,2:dim(x_test_1)[2]])
levels(x_test_1$CLASS) <- make.names(levels(factor(x_test_1$CLASS)))
te_res <- confusionMatrix(data = pred_te, reference = x_test_1$CLASS, positive = 'X1')

result_table[i,'TN_TE'] <- as.numeric(te_res$table[1])
result_table[i,'FP_TE'] <- as.numeric(te_res$table[2])
result_table[i,'FN_TE'] <- as.numeric(te_res$table[3])
result_table[i,'TP_TE'] <- as.numeric(te_res$table[4])
result_table[i,'ACC_TE'] <- round(as.numeric(te_res$overall['Accuracy']),2)
result_table[i,'BA_TE'] <- round(as.numeric(te_res$byClass['Balanced Accuracy']),2)
result_table[i,'SN_TE'] <- round(as.numeric(te_res$byClass['Sensitivity']),2)
result_table[i,'SP_TE'] <- round(as.numeric(te_res$byClass['Specificity']),2)
result_table[i,'PPV_TE'] <- round(as.numeric(te_res$byClass['Pos Pred Value']),2)
result_table[i,'NPV_TE'] <- round(as.numeric(te_res$byClass['Neg Pred Value']),2)
result_table[i,'RC_TE'] <- round(as.numeric(sensitivity(pred_te, x_test_1$CLASS, positive="X1")),2)
result_table[i,'F1_TE'] <- round(as.numeric((2 * posPredValue(pred_te, x_test_1$CLASS, positive="X1") * sensitivity(pred_te, x_test_1$CLASS, positive="X1")) / (posPredValue(pred_te, x_test_1$CLASS, positive="X1") + sensitivity(pred_te, x_test_1$CLASS, positive="X1"))),2)
te_prob <- predict(weighted_fit, newdata=x_test_1, type="prob")
result_table[i,'AUC_score_TE'] <- round(as.numeric(auc(roc(x_test_1$CLASS, te_prob[[2]]))),2)

# external test set prediction

ext_test_set <- fread('AD2D_MACCS_external_set.csv')
ext_test_set <- ext_test_set[,names(x_train_1)[2:dim(x_test_1)[2]], with=F]
ext_test_set_descr <- ext_test_set[,lapply(ext_test_set, as.factor)]

# class
ext_test_set_class <- fread('external_test_set_class.txt')

ext_test_set_class <- ifelse(ext_test_set_class$TOXICITY == "Hepatotoxic", 1, 0)

# external predictions
pred_ext = predict(weighted_fit, ext_test_set_descr)
y_test_ext_binary_mod <- as.factor(ext_test_set_class)
levels(y_test_ext_binary_mod) <- make.names(levels(factor(y_test_ext_binary_mod)))
ext_res <- confusionMatrix(data = pred_ext, reference = y_test_ext_binary_mod, positive = 'X1')

result_table[i,'TN_EXT'] <- as.numeric(ext_res$table[1])
result_table[i,'FP_EXT'] <- as.numeric(ext_res$table[2])
result_table[i,'FN_EXT'] <- as.numeric(ext_res$table[3])
result_table[i,'TP_EXT'] <- as.numeric(ext_res$table[4])
result_table[i,'ACC_EXT'] <- round(as.numeric(ext_res$overall['Accuracy']),2)
result_table[i,'BA_EXT'] <- round(as.numeric(ext_res$byClass['Balanced Accuracy']),2)
result_table[i,'SN_EXT'] <- round(as.numeric(ext_res$byClass['Sensitivity']),2)
result_table[i,'SP_EXT'] <- round(as.numeric(ext_res$byClass['Specificity']),2)
result_table[i,'PPV_EXT'] <- round(as.numeric(ext_res$byClass['Pos Pred Value']),2)
result_table[i,'NPV_EXT'] <- round(as.numeric(ext_res$byClass['Neg Pred Value']),2)
result_table[i,'RC_EXT'] <- round(as.numeric(sensitivity(pred_ext, y_test_ext_binary_mod, positive="X1")),2)
result_table[i,'F1_EXT'] <- round(as.numeric((2 * posPredValue(pred_ext, y_test_ext_binary_mod, positive="X1") * sensitivity(pred_ext, y_test_ext_binary_mod, positive="X1")) / (posPredValue(pred_ext, y_test_ext_binary_mod, positive="X1") + sensitivity(pred_ext, y_test_ext_binary_mod, positive="X1"))),2)
ext_prob <- predict(weighted_fit, newdata=ext_test_set_descr, type="prob")
result_table[i,'AUC_score_EXT'] <- round(as.numeric(auc(roc(ext_test_set_class, ext_prob[[2]]))),2)

write.table(result_table,file = 'Results_AD2D_MACCS_HS_NoFS.csv', row.names = F, col.names = T, sep = ',')

# Var importance
ImpMeasure <- varImp(weighted_fit, scale = FALSE)

library(randomForest)
fm <- weighted_fit$finalModel
var_imp_details <- data.frame(randomForest::importance(fm, scale = FALSE))
var_imp_details$Vars <- row.names(var_imp_details)
var_imp_details <- var_imp_details[,c(5,1:4)]
var_imp_details <- var_imp_details[order(var_imp_details$MeanDecreaseGini, decreasing = T),]
write.table(var_imp_details,file = 'Imp_variables.csv', row.names = F, col.names = T, sep = ',')

# save model
mod_nm <- paste('AD2D_MACCS_FP_Model_HS_NoFS', '.Rdata',sep = '')
save.image(file=mod_nm)


# 2. Hybrid (FP + HCI descr) based model

rm(list = setdiff(ls(),'path'))
setwd(path)

x_data <- fread('input_HCI_FP.csv')

set.seed(47)
# Data splitting in training and test set
x_train_orig <- stratified(x_data, c('CLASS'), 0.8)
x_test_orig <- setdiff(x_data,x_train_orig)

# save train and test set
new_folder <- paste(path,'/HCI_MACCS_Atompair_HS_NoFS',sep = '')
dir.create(new_folder)
setwd(new_folder)

write.table(x_train_orig,file = 'train_set.csv', row.names = F, col.names = T, sep = ',')
write.table(x_test_orig,file = 'test_set.csv', row.names = F, col.names = T, sep = ',')

result_table <- matrix(NA,nrow = 1, ncol = 56)
colnames(result_table) <- c('Model','mtry','ntree','nodesize','TN','FP','FN','TP','ACC','BA','SN','SP','PPV','NPV','RC','F1','AUC_score',
	'TN_CV','FP_CV','FN_CV','TP_CV','ACC_CV','BA_CV','SN_CV','SP_CV','PPV_CV','NPV_CV','RC_CV','F1_CV','AUC_score_CV',
	'TN_TE','FP_TE','FN_TE','TP_TE','ACC_TE','BA_TE','SN_TE','SP_TE','PPV_TE','NPV_TE','RC_TE','F1_TE','AUC_score_TE',
	'TN_EXT','FP_EXT','FN_EXT','TP_EXT','ACC_EXT','BA_EXT','SN_EXT','SP_EXT','PPV_EXT','NPV_EXT','RC_EXT','F1_EXT','AUC_score_EXT')

x_train_1 <- as.data.frame(x_train_orig[,-'Sample_id'])
# convert to factors

x_train_1$CLASS <- as.factor(x_train_1$CLASS)
levels(x_train_1$CLASS) <- make.names(levels(factor(x_train_1$CLASS)))
#x_train_1[,1722:2128] <- lapply(x_train_1[,1722:2128], as.factor)

# RF model building
model_weights <- ifelse(x_train_1$CLASS=='0', (1/table(x_train_1$CLASS)[1]) * 0.5, (1/table(x_train_1$CLASS)[2]) * 0.5)
seeds <- as.vector(c(1:6), mode = "list")
seeds[[6]] <- 1
mtry_val = as.integer(sqrt(dim(x_train_1)[2]))

set.seed(47)
customRF <- list(type = "Classification", library = "randomForest", loop = NULL)
customRF$parameters <- data.frame(parameter = c("mtry", "ntree", "nodesize"), class = rep("numeric", 3), label = c("mtry", "ntree", "nodesize"))
customRF$grid <- function(x, y, len = NULL, search = "grid") {}
customRF$fit <- function(x, y, wts, param, lev, last, weights, classProbs, ...) {
	randomForest(x, y, mtry = param$mtry, ntree=param$ntree, ...)
}
customRF$predict <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
	predict(modelFit, newdata)
customRF$prob <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
	predict(modelFit, newdata, type = "prob")
customRF$sort <- function(x) x[order(x[,1]),]
customRF$levels <- function(x) x$classes

library(doMC)
registerDoMC(cores = 24)

tunegrid <- data.frame(expand.grid(.mtry=c(35,40,45,50,55), .ntree=c(50, 100,500), .nodesize=c(1:5)))
weighted_fit <- caret::train(CLASS ~ .,
	data = x_train_1,
	method = customRF,
	importance = TRUE,
	verbose = FALSE,
	replace=FALSE,
	metric = "Accuracy",
	tuneGrid = tunegrid,
	weights = model_weights,
	trControl = trainControl(method = "cv", number = 5, classProbs=T, savePredictions = "final", allowParallel = TRUE))


# training pred
pred_tr = predict(weighted_fit, x_train_1[,-1])
tr_res <- confusionMatrix(data = pred_tr, reference = x_train_1$CLASS, positive = 'X1')

# save results
i <- 1
result_table[i,'Model'] <- 'HCI_AD2D_MACCS_HS_NoFS'
result_table[i,'mtry'] <- weighted_fit$bestTune$mtry
result_table[i,'ntree'] <- weighted_fit$bestTune$ntree
result_table[i,'nodesize'] <- weighted_fit$bestTune$nodesize

result_table[i,'TN'] <- as.numeric(tr_res$table[1])
result_table[i,'FP'] <- as.numeric(tr_res$table[2])
result_table[i,'FN'] <- as.numeric(tr_res$table[3])
result_table[i,'TP'] <- as.numeric(tr_res$table[4])
result_table[i,'ACC'] <- as.numeric(tr_res$overall['Accuracy'])
result_table[i,'BA'] <- as.numeric(tr_res$byClass['Balanced Accuracy'])
result_table[i,'SN'] <- as.numeric(tr_res$byClass['Sensitivity'])
result_table[i,'SP'] <- as.numeric(tr_res$byClass['Specificity'])
result_table[i,'PPV'] <- as.numeric(tr_res$byClass['Pos Pred Value'])
result_table[i,'NPV'] <- as.numeric(tr_res$byClass['Neg Pred Value'])
result_table[i,'RC'] <- as.numeric(sensitivity(pred_tr, x_train_1$CLASS, positive="X1"))
result_table[i,'F1'] <- as.numeric((2 * posPredValue(pred_tr, x_train_1$CLASS, positive="X1") * sensitivity(pred_tr, x_train_1$CLASS, positive="X1")) / (posPredValue(pred_tr, x_train_1$CLASS, positive="X1") + sensitivity(pred_tr, x_train_1$CLASS, positive="X1")))
tr_prob <- predict(weighted_fit, newdata=x_train_1, type="prob")
roc_obj <- roc(x_train_1$CLASS, tr_prob[[2]])
result_table[i,'AUC_score'] <- as.numeric(auc(roc_obj))

# CV output extraction
all_folds <- weighted_fit$pred
CV_results <- matrix(NA, nrow = 5, ncol = 14)
colnames(CV_results) <- c('Fold','TN','FP','FN','TP','ACC','BA','SN','SP','PPV','NPV','RC','F1','AUC_score')
for(j in 1:5){
	A <- paste('Fold',j,sep = '')
	B <- all_folds[all_folds$Resample == A,]
	CV_results[j,'Fold'] <- j
	CV_results[j,'TN'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[1])
	CV_results[j,'FP'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[2])
	CV_results[j,'FN'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[3])
	CV_results[j,'TP'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[4])
	CV_results[j,'ACC'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$overall['Accuracy'])
	CV_results[j,'BA'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Balanced Accuracy'])
	CV_results[j,'SN'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Sensitivity'])
	CV_results[j,'SP'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Specificity'])
	CV_results[j,'PPV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Pos Pred Value'])
	CV_results[j,'NPV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Neg Pred Value'])
	CV_results[j,'RC'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Recall'])
	CV_results[j,'F1'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['F1'])
	CV_results[j,'AUC_score'] <- as.numeric(auc(roc(B$obs, B$X1)))
}

CV_results_final <- colMeans(CV_results[,2:14], na.rm=TRUE)
CV_results_final[1:4] <- round(CV_results_final[1:4], 0)
CV_results_final[5:13] <- round(CV_results_final[5:13], 2)
# add to final result
result_table[i,18:30] <- CV_results_final

# test pred
x_test_1 <- as.data.frame(x_test_orig[,-'Sample_id'])
# convert to factors
x_test_1$CLASS <- as.factor(x_test_1$CLASS)


pred_te = predict(weighted_fit, x_test_1[,2:dim(x_test_1)[2]])
levels(x_test_1$CLASS) <- make.names(levels(factor(x_test_1$CLASS)))
te_res <- confusionMatrix(data = pred_te, reference = x_test_1$CLASS, positive = 'X1')

result_table[i,'TN_TE'] <- as.numeric(te_res$table[1])
result_table[i,'FP_TE'] <- as.numeric(te_res$table[2])
result_table[i,'FN_TE'] <- as.numeric(te_res$table[3])
result_table[i,'TP_TE'] <- as.numeric(te_res$table[4])
result_table[i,'ACC_TE'] <- round(as.numeric(te_res$overall['Accuracy']),2)
result_table[i,'BA_TE'] <- round(as.numeric(te_res$byClass['Balanced Accuracy']),2)
result_table[i,'SN_TE'] <- round(as.numeric(te_res$byClass['Sensitivity']),2)
result_table[i,'SP_TE'] <- round(as.numeric(te_res$byClass['Specificity']),2)
result_table[i,'PPV_TE'] <- round(as.numeric(te_res$byClass['Pos Pred Value']),2)
result_table[i,'NPV_TE'] <- round(as.numeric(te_res$byClass['Neg Pred Value']),2)
result_table[i,'RC_TE'] <- round(as.numeric(sensitivity(pred_te, x_test_1$CLASS, positive="X1")),2)
result_table[i,'F1_TE'] <- round(as.numeric((2 * posPredValue(pred_te, x_test_1$CLASS, positive="X1") * sensitivity(pred_te, x_test_1$CLASS, positive="X1")) / (posPredValue(pred_te, x_test_1$CLASS, positive="X1") + sensitivity(pred_te, x_test_1$CLASS, positive="X1"))),2)
te_prob <- predict(weighted_fit, newdata=x_test_1, type="prob")
result_table[i,'AUC_score_TE'] <- round(as.numeric(auc(roc(x_test_1$CLASS, te_prob[[2]]))),2)

# external test set prediction
ext_test_set_1 <- fread('Standardized_avg_hybrid_descr_independent_test_set.csv')
ext_test_set_2 <- fread('AD2D_MACCS_external_set.csv')
ext_test_set <- cbind(ext_test_set_1,ext_test_set_2)
ext_test_set <- ext_test_set[,names(x_test_1)[2:dim(x_test_1)[2]], with=F]
# class
ext_test_set_class <- fread('external_test_set_class.txt')
ext_test_set_class <- ifelse(ext_test_set_class$TOXICITY == "Hepatotoxic", 1, 0)

# external predictions
pred_ext = predict(weighted_fit, ext_test_set)
y_test_ext_binary_mod <- as.factor(ext_test_set_class)
levels(y_test_ext_binary_mod) <- make.names(levels(factor(y_test_ext_binary_mod)))
ext_res <- confusionMatrix(data = pred_ext, reference = y_test_ext_binary_mod, positive = 'X1')

result_table[i,'TN_EXT'] <- as.numeric(ext_res$table[1])
result_table[i,'FP_EXT'] <- as.numeric(ext_res$table[2])
result_table[i,'FN_EXT'] <- as.numeric(ext_res$table[3])
result_table[i,'TP_EXT'] <- as.numeric(ext_res$table[4])
result_table[i,'ACC_EXT'] <- round(as.numeric(ext_res$overall['Accuracy']),2)
result_table[i,'BA_EXT'] <- round(as.numeric(ext_res$byClass['Balanced Accuracy']),2)
result_table[i,'SN_EXT'] <- round(as.numeric(ext_res$byClass['Sensitivity']),2)
result_table[i,'SP_EXT'] <- round(as.numeric(ext_res$byClass['Specificity']),2)
result_table[i,'PPV_EXT'] <- round(as.numeric(ext_res$byClass['Pos Pred Value']),2)
result_table[i,'NPV_EXT'] <- round(as.numeric(ext_res$byClass['Neg Pred Value']),2)
result_table[i,'RC_EXT'] <- round(as.numeric(sensitivity(pred_ext, y_test_ext_binary_mod, positive="X1")),2)
result_table[i,'F1_EXT'] <- round(as.numeric((2 * posPredValue(pred_ext, y_test_ext_binary_mod, positive="X1") * sensitivity(pred_ext, y_test_ext_binary_mod, positive="X1")) / (posPredValue(pred_ext, y_test_ext_binary_mod, positive="X1") + sensitivity(pred_ext, y_test_ext_binary_mod, positive="X1"))),2)
ext_prob <- predict(weighted_fit, newdata=ext_test_set, type="prob")
result_table[i,'AUC_score_EXT'] <- round(as.numeric(auc(roc(ext_test_set_class, ext_prob[[2]]))),2)

write.table(result_table,file = 'Results_HCI_AD2D_MACCS_HS_NoFS.csv', row.names = F, col.names = T, sep = ',')

# Var importance
ImpMeasure <- varImp(weighted_fit, scale = FALSE)

library(randomForest)
fm <- weighted_fit$finalModel
var_imp_details <- data.frame(randomForest::importance(fm, scale = FALSE))
var_imp_details$Vars <- row.names(var_imp_details)
var_imp_details <- var_imp_details[,c(5,1:4)]
var_imp_details <- var_imp_details[order(var_imp_details$MeanDecreaseGini, decreasing = T),]
write.table(var_imp_details,file = 'Imp_variables.csv', row.names = F, col.names = T, sep = ',')

# save model
mod_nm <- paste('HCI_AD2D_MACCS_FP_Model_HS_NoFS', '.Rdata',sep = '')
save.image(file=mod_nm)


# ---------------------------- Approach-C: NoHS_FS ---------------------------------------------------

# 1. FP
rm(list = setdiff(ls(),'path'))
setwd(path)

# Load FDA list 346 chemicals with their FP data

avg_variables <- fread('MACCS_Atompair_FP_346_chem.csv')
chem_info <- fread('all_346_chem_info.csv')
classes_346 <- fread('avg_data_class.csv')

# conver x_data to factors
x_data <- avg_variables[,-'Name']
x_data[,1:946] <- lapply(x_data[,1:946], as.factor)

library(Boruta)

set.seed(47)

FS_boruta_model <- Boruta(x_data, classes_346$CLASSES, pValue = 0.05, mcAdj = TRUE, maxRuns = 100)
best_var <- colnames(x_data)[which(FS_boruta_model$finalDecision == 'Confirmed')]

# add CLASS column
x_data[, Sample_id := chem_info$Sample_id]
x_data[, CLASS := classes_346$CLASSES]

x_data <- x_data[,c('Sample_id','CLASS',best_var),with=F]

# Data splitting in training and test set
set.seed(47)
x_train_orig <- stratified(x_data, c('CLASS'), 0.8)
x_test_orig <- setdiff(x_data,x_train_orig)

# save train and test set
new_folder <- paste(path,'/MACCS_Atompair_NoHS_FS',sep = '')
dir.create(new_folder)
setwd(new_folder)

write.table(x_train_orig,file = 'train_set.csv', row.names = F, col.names = T, sep = ',')
write.table(x_test_orig,file = 'test_set.csv', row.names = F, col.names = T, sep = ',')

x_train_1 <- as.data.frame(x_train_orig[,-'Sample_id'])
# convert to factors
x_train_1[,2:17] <- lapply(x_train_1[,2:17], as.factor)
x_train_1$CLASS <- as.factor(x_train_1$CLASS)

# drop levels
x_train_1 <- x_train_1[, sapply(x_train_1, nlevels) > 1]
levels(x_train_1$CLASS) <- make.names(levels(factor(x_train_1$CLASS)))

# RF model building
model_weights <- ifelse(x_train_1$CLASS=='0', (1/table(x_train_1$CLASS)[1]) * 0.5, (1/table(x_train_1$CLASS)[2]) * 0.5)
seeds <- as.vector(c(1:6), mode = "list")
seeds[[6]] <- 1
mtry_val = as.integer(sqrt(dim(x_train_1)[2]))

# to save all results
result_table <- matrix(NA,nrow = 1, ncol = 54)
colnames(result_table) <- c('Model','var_count','TN','FP','FN','TP','ACC','BA','SN','SP','PPV','NPV','RC','F1','AUC_score',
	'TN_CV','FP_CV','FN_CV','TP_CV','ACC_CV','BA_CV','SN_CV','SP_CV','PPV_CV','NPV_CV','RC_CV','F1_CV','AUC_score_CV',
	'TN_TE','FP_TE','FN_TE','TP_TE','ACC_TE','BA_TE','SN_TE','SP_TE','PPV_TE','NPV_TE','RC_TE','F1_TE','AUC_score_TE',
	'TN_EXT','FP_EXT','FN_EXT','TP_EXT','ACC_EXT','BA_EXT','SN_EXT','SP_EXT','PPV_EXT','NPV_EXT','RC_EXT','F1_EXT','AUC_score_EXT')

set.seed(47)

library(doMC)
registerDoMC(cores = 32)

weighted_fit <- caret::train(CLASS ~ .,
	data = x_train_1,
	method = "rf",
	importance = TRUE,
	verbose = FALSE,
	ntree=100,
	replace=FALSE, 
	nodesize = 5,
	metric = "Accuracy",
	tuneGrid = data.frame(mtry = mtry_val),
	tuneLength = 1,
	weights = model_weights,
	trControl = trainControl(method = "cv", number = 5, classProbs=T, savePredictions = "final", seeds = seeds))

# training pred
pred_tr = predict(weighted_fit, x_train_1)
tr_res <- confusionMatrix(data = pred_tr, reference = x_train_1$CLASS, positive = 'X1')

i <- 1
result_table[i,'Model'] <- 'MACCS_Atompair_NoHS_FS'
result_table[i,'var_count'] <- c(length(best_var))
result_table[i,'TN'] <- as.numeric(tr_res$table[1])
result_table[i,'FP'] <- as.numeric(tr_res$table[2])
result_table[i,'FN'] <- as.numeric(tr_res$table[3])
result_table[i,'TP'] <- as.numeric(tr_res$table[4])
result_table[i,'ACC'] <- as.numeric(tr_res$overall['Accuracy'])
result_table[i,'BA'] <- as.numeric(tr_res$byClass['Balanced Accuracy'])
result_table[i,'SN'] <- as.numeric(tr_res$byClass['Sensitivity'])
result_table[i,'SP'] <- as.numeric(tr_res$byClass['Specificity'])
result_table[i,'PPV'] <- as.numeric(tr_res$byClass['Pos Pred Value'])
result_table[i,'NPV'] <- as.numeric(tr_res$byClass['Neg Pred Value'])
result_table[i,'RC'] <- as.numeric(sensitivity(pred_tr, x_train_1$CLASS, positive="X1"))
result_table[i,'F1'] <- as.numeric((2 * posPredValue(pred_tr, x_train_1$CLASS, positive="X1") * sensitivity(pred_tr, x_train_1$CLASS, positive="X1")) / (posPredValue(pred_tr, x_train_1$CLASS, positive="X1") + sensitivity(pred_tr, x_train_1$CLASS, positive="X1")))
tr_prob <- predict(weighted_fit, newdata=x_train_1, type="prob")
roc_obj <- roc(x_train_1$CLASS, tr_prob[[2]])
result_table[i,'AUC_score'] <- as.numeric(auc(roc_obj))

# CV output extraction
all_folds <- weighted_fit$pred
CV_results <- matrix(NA, nrow = 5, ncol = 14)
colnames(CV_results) <- c('Fold','TN','FP','FN','TP','ACC','BA','SN','SP','PPV','NPV','RC','F1','AUC_score')
for(j in 1:5){
	A <- paste('Fold',j,sep = '')
	B <- all_folds[all_folds$Resample == A,]
	CV_results[j,'Fold'] <- j
	CV_results[j,'TN'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[1])
	CV_results[j,'FP'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[2])
	CV_results[j,'FN'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[3])
	CV_results[j,'TP'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[4])
	CV_results[j,'ACC'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$overall['Accuracy'])
	CV_results[j,'BA'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Balanced Accuracy'])
	CV_results[j,'SN'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Sensitivity'])
	CV_results[j,'SP'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Specificity'])
	CV_results[j,'PPV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Pos Pred Value'])
	CV_results[j,'NPV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Neg Pred Value'])
	CV_results[j,'RC'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Recall'])
	CV_results[j,'F1'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['F1'])
	CV_results[j,'AUC_score'] <- as.numeric(auc(roc(B$obs, B$X1)))
}

CV_results_final <- colMeans(CV_results[,2:14], na.rm=TRUE)
CV_results_final[1:4] <- round(CV_results_final[1:4], 0)
CV_results_final[5:13] <- round(CV_results_final[5:13], 2)
# add to final result
result_table[i,16:28] <- CV_results_final

# test pred
x_test_1 <- as.data.frame(x_test_orig[,-'Sample_id'])
# convert to factors
x_test_1[,2:17] <- lapply(x_test_1[,2:17], as.factor)
x_test_1$CLASS <- as.factor(x_test_1$CLASS)

pred_te = predict(weighted_fit, x_test_1[,2:dim(x_test_1)[2]])
levels(x_test_1$CLASS) <- make.names(levels(factor(x_test_1$CLASS)))
te_res <- confusionMatrix(data = pred_te, reference = x_test_1$CLASS, positive = 'X1')

result_table[i,'TN_TE'] <- as.numeric(te_res$table[1])
result_table[i,'FP_TE'] <- as.numeric(te_res$table[2])
result_table[i,'FN_TE'] <- as.numeric(te_res$table[3])
result_table[i,'TP_TE'] <- as.numeric(te_res$table[4])
result_table[i,'ACC_TE'] <- round(as.numeric(te_res$overall['Accuracy']),2)
result_table[i,'BA_TE'] <- round(as.numeric(te_res$byClass['Balanced Accuracy']),2)
result_table[i,'SN_TE'] <- round(as.numeric(te_res$byClass['Sensitivity']),2)
result_table[i,'SP_TE'] <- round(as.numeric(te_res$byClass['Specificity']),2)
result_table[i,'PPV_TE'] <- round(as.numeric(te_res$byClass['Pos Pred Value']),2)
result_table[i,'NPV_TE'] <- round(as.numeric(te_res$byClass['Neg Pred Value']),2)
result_table[i,'RC_TE'] <- round(as.numeric(sensitivity(pred_te, x_test_1$CLASS, positive="X1")),2)
result_table[i,'F1_TE'] <- round(as.numeric((2 * posPredValue(pred_te, x_test_1$CLASS, positive="X1") * sensitivity(pred_te, x_test_1$CLASS, positive="X1")) / (posPredValue(pred_te, x_test_1$CLASS, positive="X1") + sensitivity(pred_te, x_test_1$CLASS, positive="X1"))),2)
te_prob <- predict(weighted_fit, newdata=x_test_1, type="prob")
result_table[i,'AUC_score_TE'] <- round(as.numeric(auc(roc(x_test_1$CLASS, te_prob[[2]]))),2)

# external test set prediction

ext_test_set <- fread('AD2D_MACCS_external_set.csv')
ext_test_set <- ext_test_set[,names(x_train_1)[2:dim(x_test_1)[2]], with=F]
ext_test_set_descr <- ext_test_set[,lapply(ext_test_set, as.factor)]

# class
ext_test_set_class <- fread('external_test_set_class.txt')

ext_test_set_class <- ifelse(ext_test_set_class$TOXICITY == "Hepatotoxic", 1, 0)

# external predictions
pred_ext = predict(weighted_fit, ext_test_set_descr)
y_test_ext_binary_mod <- as.factor(ext_test_set_class)
levels(y_test_ext_binary_mod) <- make.names(levels(factor(y_test_ext_binary_mod)))
ext_res <- confusionMatrix(data = pred_ext, reference = y_test_ext_binary_mod, positive = 'X1')

result_table[i,'TN_EXT'] <- as.numeric(ext_res$table[1])
result_table[i,'FP_EXT'] <- as.numeric(ext_res$table[2])
result_table[i,'FN_EXT'] <- as.numeric(ext_res$table[3])
result_table[i,'TP_EXT'] <- as.numeric(ext_res$table[4])
result_table[i,'ACC_EXT'] <- round(as.numeric(ext_res$overall['Accuracy']),2)
result_table[i,'BA_EXT'] <- round(as.numeric(ext_res$byClass['Balanced Accuracy']),2)
result_table[i,'SN_EXT'] <- round(as.numeric(ext_res$byClass['Sensitivity']),2)
result_table[i,'SP_EXT'] <- round(as.numeric(ext_res$byClass['Specificity']),2)
result_table[i,'PPV_EXT'] <- round(as.numeric(ext_res$byClass['Pos Pred Value']),2)
result_table[i,'NPV_EXT'] <- round(as.numeric(ext_res$byClass['Neg Pred Value']),2)
result_table[i,'RC_EXT'] <- round(as.numeric(sensitivity(pred_ext, y_test_ext_binary_mod, positive="X1")),2)
result_table[i,'F1_EXT'] <- round(as.numeric((2 * posPredValue(pred_ext, y_test_ext_binary_mod, positive="X1") * sensitivity(pred_ext, y_test_ext_binary_mod, positive="X1")) / (posPredValue(pred_ext, y_test_ext_binary_mod, positive="X1") + sensitivity(pred_ext, y_test_ext_binary_mod, positive="X1"))),2)
ext_prob <- predict(weighted_fit, newdata=ext_test_set_descr, type="prob")
result_table[i,'AUC_score_EXT'] <- round(as.numeric(auc(roc(ext_test_set_class, ext_prob[[2]]))),2)

write.table(result_table,file = 'Results_AD2D_MACCS_HS_NoFS.csv', row.names = F, col.names = T, sep = ',')

# Var importance
ImpMeasure <- varImp(weighted_fit, scale = FALSE)

library(randomForest)
fm <- weighted_fit$finalModel
var_imp_details <- data.frame(randomForest::importance(fm, scale = FALSE))
var_imp_details$Vars <- row.names(var_imp_details)
var_imp_details <- var_imp_details[,c(5,1:4)]
var_imp_details <- var_imp_details[order(var_imp_details$MeanDecreaseGini, decreasing = T),]
write.table(var_imp_details,file = 'Imp_variables.csv', row.names = F, col.names = T, sep = ',')

# save model
mod_nm <- paste('AD2D_MACCS_FP_Model_NoHS_FS', '.Rdata',sep = '')
save.image(file=mod_nm)


# 2. Hybrid (FP + HCI descr) based model

rm(list = setdiff(ls(),'path'))
setwd(path)

avg_variables <- fread('input_HCI_FP.csv')

# conver x_data to factors
x_data <- avg_variables[,!c('Sample_id','CLASS')]
x_data[,1721:2127] <- lapply(x_data[,1721:2127], as.factor)


library(Boruta)

set.seed(47)
FS_boruta_model <- Boruta(x_data, avg_variables$CLASS, pValue = 0.05, mcAdj = TRUE, maxRuns = 100)
best_var <- colnames(x_data)[which(FS_boruta_model$finalDecision == 'Confirmed')]

# add id & CLASS column
x_data[, Sample_id := avg_variables$Sample_id]
x_data[, CLASS := avg_variables$CLASS]

x_data <- x_data[,c('Sample_id','CLASS',best_var),with=F]

# Data splitting in training and test set
set.seed(47)
x_train_orig <- stratified(x_data, c('CLASS'), 0.8)
x_test_orig <- setdiff(x_data,x_train_orig)

# save train and test set
new_folder <- paste(path,'/HCI_MACCS_Atompair_NoHS_FS',sep = '')
dir.create(new_folder)
setwd(new_folder)

write.table(x_train_orig,file = 'train_set.csv', row.names = F, col.names = T, sep = ',')
write.table(x_test_orig,file = 'test_set.csv', row.names = F, col.names = T, sep = ',')

result_table <- matrix(NA,nrow = 1, ncol = 54)
colnames(result_table) <- c('Model','var_count','TN','FP','FN','TP','ACC','BA','SN','SP','PPV','NPV','RC','F1','AUC_score',
	'TN_CV','FP_CV','FN_CV','TP_CV','ACC_CV','BA_CV','SN_CV','SP_CV','PPV_CV','NPV_CV','RC_CV','F1_CV','AUC_score_CV',
	'TN_TE','FP_TE','FN_TE','TP_TE','ACC_TE','BA_TE','SN_TE','SP_TE','PPV_TE','NPV_TE','RC_TE','F1_TE','AUC_score_TE',
	'TN_EXT','FP_EXT','FN_EXT','TP_EXT','ACC_EXT','BA_EXT','SN_EXT','SP_EXT','PPV_EXT','NPV_EXT','RC_EXT','F1_EXT','AUC_score_EXT')

x_train_1 <- as.data.frame(x_train_orig[,-'Sample_id'])
# convert to factors
x_train_1$CLASS <- as.factor(x_train_1$CLASS)
levels(x_train_1$CLASS) <- make.names(levels(factor(x_train_1$CLASS)))


# RF model building
model_weights <- ifelse(x_train_1$CLASS=='0', (1/table(x_train_1$CLASS)[1]) * 0.5, (1/table(x_train_1$CLASS)[2]) * 0.5)
seeds <- as.vector(c(1:6), mode = "list")
seeds[[6]] <- 1
mtry_val = as.integer(sqrt(dim(x_train_1)[2]))

set.seed(47)

library(doMC)
registerDoMC(cores = 32)

weighted_fit <- caret::train(CLASS ~ .,
	data = x_train_1,
	method = "rf",
	importance = TRUE,
	verbose = FALSE,
	ntree=100,
	replace=FALSE, 
	nodesize = 5,
	metric = "Accuracy",
	tuneGrid = data.frame(mtry = mtry_val),
	tuneLength = 1,
	weights = model_weights,
	trControl = trainControl(method = "cv", number = 5, classProbs=T, savePredictions = "final", seeds = seeds))

# training pred
pred_tr = predict(weighted_fit, x_train_1[,-1])
tr_res <- confusionMatrix(data = pred_tr, reference = x_train_1$CLASS, positive = 'X1')

# save results
i <- 1
result_table[i,'Model'] <- 'HCI_MACCS_Atompair_NoHS_FS'
result_table[i,'var_count'] <- c(length(best_var))
result_table[i,'TN'] <- as.numeric(tr_res$table[1])
result_table[i,'FP'] <- as.numeric(tr_res$table[2])
result_table[i,'FN'] <- as.numeric(tr_res$table[3])
result_table[i,'TP'] <- as.numeric(tr_res$table[4])
result_table[i,'ACC'] <- as.numeric(tr_res$overall['Accuracy'])
result_table[i,'BA'] <- as.numeric(tr_res$byClass['Balanced Accuracy'])
result_table[i,'SN'] <- as.numeric(tr_res$byClass['Sensitivity'])
result_table[i,'SP'] <- as.numeric(tr_res$byClass['Specificity'])
result_table[i,'PPV'] <- as.numeric(tr_res$byClass['Pos Pred Value'])
result_table[i,'NPV'] <- as.numeric(tr_res$byClass['Neg Pred Value'])
result_table[i,'RC'] <- as.numeric(sensitivity(pred_tr, x_train_1$CLASS, positive="X1"))
result_table[i,'F1'] <- as.numeric((2 * posPredValue(pred_tr, x_train_1$CLASS, positive="X1") * sensitivity(pred_tr, x_train_1$CLASS, positive="X1")) / (posPredValue(pred_tr, x_train_1$CLASS, positive="X1") + sensitivity(pred_tr, x_train_1$CLASS, positive="X1")))
tr_prob <- predict(weighted_fit, newdata=x_train_1, type="prob")
roc_obj <- roc(x_train_1$CLASS, tr_prob[[2]])
result_table[i,'AUC_score'] <- as.numeric(auc(roc_obj))

# CV output extraction
all_folds <- weighted_fit$pred
CV_results <- matrix(NA, nrow = 5, ncol = 14)
colnames(CV_results) <- c('Fold','TN','FP','FN','TP','ACC','BA','SN','SP','PPV','NPV','RC','F1','AUC_score')
for(j in 1:5){
	A <- paste('Fold',j,sep = '')
	B <- all_folds[all_folds$Resample == A,]
	CV_results[j,'Fold'] <- j
	CV_results[j,'TN'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[1])
	CV_results[j,'FP'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[2])
	CV_results[j,'FN'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[3])
	CV_results[j,'TP'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[4])
	CV_results[j,'ACC'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$overall['Accuracy'])
	CV_results[j,'BA'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Balanced Accuracy'])
	CV_results[j,'SN'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Sensitivity'])
	CV_results[j,'SP'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Specificity'])
	CV_results[j,'PPV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Pos Pred Value'])
	CV_results[j,'NPV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Neg Pred Value'])
	CV_results[j,'RC'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Recall'])
	CV_results[j,'F1'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['F1'])
	CV_results[j,'AUC_score'] <- as.numeric(auc(roc(B$obs, B$X1)))
}

CV_results_final <- colMeans(CV_results[,2:14], na.rm=TRUE)
CV_results_final[1:4] <- round(CV_results_final[1:4], 0)
CV_results_final[5:13] <- round(CV_results_final[5:13], 2)
# add to final result
result_table[i,16:28] <- CV_results_final

# test pred
x_test_1 <- as.data.frame(x_test_orig[,-'Sample_id'])
# convert to factors
x_test_1$CLASS <- as.factor(x_test_1$CLASS)

pred_te = predict(weighted_fit, x_test_1[,2:dim(x_test_1)[2]])
levels(x_test_1$CLASS) <- make.names(levels(factor(x_test_1$CLASS)))
te_res <- confusionMatrix(data = pred_te, reference = x_test_1$CLASS, positive = 'X1')

result_table[i,'TN_TE'] <- as.numeric(te_res$table[1])
result_table[i,'FP_TE'] <- as.numeric(te_res$table[2])
result_table[i,'FN_TE'] <- as.numeric(te_res$table[3])
result_table[i,'TP_TE'] <- as.numeric(te_res$table[4])
result_table[i,'ACC_TE'] <- round(as.numeric(te_res$overall['Accuracy']),2)
result_table[i,'BA_TE'] <- round(as.numeric(te_res$byClass['Balanced Accuracy']),2)
result_table[i,'SN_TE'] <- round(as.numeric(te_res$byClass['Sensitivity']),2)
result_table[i,'SP_TE'] <- round(as.numeric(te_res$byClass['Specificity']),2)
result_table[i,'PPV_TE'] <- round(as.numeric(te_res$byClass['Pos Pred Value']),2)
result_table[i,'NPV_TE'] <- round(as.numeric(te_res$byClass['Neg Pred Value']),2)
result_table[i,'RC_TE'] <- round(as.numeric(sensitivity(pred_te, x_test_1$CLASS, positive="X1")),2)
result_table[i,'F1_TE'] <- round(as.numeric((2 * posPredValue(pred_te, x_test_1$CLASS, positive="X1") * sensitivity(pred_te, x_test_1$CLASS, positive="X1")) / (posPredValue(pred_te, x_test_1$CLASS, positive="X1") + sensitivity(pred_te, x_test_1$CLASS, positive="X1"))),2)
te_prob <- predict(weighted_fit, newdata=x_test_1, type="prob")
result_table[i,'AUC_score_TE'] <- round(as.numeric(auc(roc(x_test_1$CLASS, te_prob[[2]]))),2)

# external test set prediction

# external test set prediction
ext_test_set_1 <- fread('Standardized_avg_hybrid_descr_independent_test_set.csv')
ext_test_set_2 <- fread('AD2D_MACCS_external_set.csv')
ext_test_set <- cbind(ext_test_set_1,ext_test_set_2)
ext_test_set <- ext_test_set[,names(x_test_1)[2:dim(x_test_1)[2]], with=F]
# convertr FP to factor
ext_test_set$MACCSFP17 <- as.factor(ext_test_set$MACCSFP17)
# class
ext_test_set_class <- ifelse(ext_test_set_1$TOXICITY == "Hepatotoxic", 1, 0)

# external predictions
pred_ext = predict(weighted_fit, ext_test_set)
y_test_ext_binary_mod <- as.factor(ext_test_set_class)
levels(y_test_ext_binary_mod) <- make.names(levels(factor(y_test_ext_binary_mod)))
ext_res <- confusionMatrix(data = pred_ext, reference = y_test_ext_binary_mod, positive = 'X1')

result_table[i,'TN_EXT'] <- as.numeric(ext_res$table[1])
result_table[i,'FP_EXT'] <- as.numeric(ext_res$table[2])
result_table[i,'FN_EXT'] <- as.numeric(ext_res$table[3])
result_table[i,'TP_EXT'] <- as.numeric(ext_res$table[4])
result_table[i,'ACC_EXT'] <- round(as.numeric(ext_res$overall['Accuracy']),2)
result_table[i,'BA_EXT'] <- round(as.numeric(ext_res$byClass['Balanced Accuracy']),2)
result_table[i,'SN_EXT'] <- round(as.numeric(ext_res$byClass['Sensitivity']),2)
result_table[i,'SP_EXT'] <- round(as.numeric(ext_res$byClass['Specificity']),2)
result_table[i,'PPV_EXT'] <- round(as.numeric(ext_res$byClass['Pos Pred Value']),2)
result_table[i,'NPV_EXT'] <- round(as.numeric(ext_res$byClass['Neg Pred Value']),2)
result_table[i,'RC_EXT'] <- round(as.numeric(sensitivity(pred_ext, y_test_ext_binary_mod, positive="X1")),2)
result_table[i,'F1_EXT'] <- round(as.numeric((2 * posPredValue(pred_ext, y_test_ext_binary_mod, positive="X1") * sensitivity(pred_ext, y_test_ext_binary_mod, positive="X1")) / (posPredValue(pred_ext, y_test_ext_binary_mod, positive="X1") + sensitivity(pred_ext, y_test_ext_binary_mod, positive="X1"))),2)
ext_prob <- predict(weighted_fit, newdata=ext_test_set, type="prob")
result_table[i,'AUC_score_EXT'] <- round(as.numeric(auc(roc(ext_test_set_class, ext_prob[[2]]))),2)

write.table(result_table,file = 'Results_HCI_AD2D_MACCS_NoHS_FS.csv', row.names = F, col.names = T, sep = ',')

# Test within AD
test_AD <- fread('test_set_norm_within_AD.csv')
test_AD <- test_AD[,-'Index']
test_AD$MACCSFP17 <- as.factor(test_AD$MACCSFP17)

test_AD_class_mod <- as.factor(test_AD$CLASS)
levels(test_AD_class_mod) <- make.names(levels(factor(test_AD_class_mod)))

pred_test_AD = predict(weighted_fit, test_AD[,3:5,with=F])
test_AD_res <- confusionMatrix(data = pred_test_AD, reference = test_AD_class_mod, positive = 'X1')
print(test_AD_res)

# External test within AD
ext_AD <- fread('external_set_norm_within_AD.csv')
ext_AD$MACCSFP17 <- as.factor(ext_AD$MACCSFP17)

ext_AD_class <- ifelse(ext_AD$CLASS == "Hepatotoxic", 1, 0)
ext_AD_class_mod <- as.factor(ext_AD_class)
levels(ext_AD_class_mod) <- make.names(levels(factor(ext_AD_class_mod)))

pred_ext_AD = predict(weighted_fit, ext_AD[,3:5,with=F])
ext_AD_res <- confusionMatrix(data = pred_ext_AD, reference = ext_AD_class_mod, positive = 'X1')
print(ext_AD_res)

# Var importance
ImpMeasure <- varImp(weighted_fit, scale = FALSE)

library(randomForest)
fm <- weighted_fit$finalModel
var_imp_details <- data.frame(randomForest::importance(fm, scale = FALSE))
var_imp_details$Vars <- row.names(var_imp_details)
var_imp_details <- var_imp_details[,c(5,1:4)]
var_imp_details <- var_imp_details[order(var_imp_details$MeanDecreaseGini, decreasing = T),]
write.table(var_imp_details,file = 'Imp_variables.csv', row.names = F, col.names = T, sep = ',')

# save model
mod_nm <- paste('HCI_AD2D_MACCS_FP_Model_NoHS_FS', '.Rdata',sep = '')
save.image(file=mod_nm)


# ---------------------------------------------------------------------------------------------------
#                                  McNemar test 
# ---------------------------------------------------------------------------------------------------

# Independent test set

load("~/chemical_HS_NoFS.Rdata")
Chemical_HS_NoFS_test_pred <- pred_te
rm(list = setdiff(ls(),c('Chemical_HS_NoFS_test_pred')))

load("~/chemical_NoHS_FS.Rdata")
Chemical_NoHS_FS_test_pred <- pred_te
rm(list = setdiff(ls(),c('Chemical_HS_NoFS_test_pred','Chemical_NoHS_FS_test_pred')))

load("~/chemical_NoHS_NoFS.Rdata")
Chemical_NoHS_NoFS_test_pred <- pred_te
rm(list = setdiff(ls(),c('Chemical_HS_NoFS_test_pred','Chemical_NoHS_FS_test_pred','Chemical_NoHS_NoFS_test_pred')))

load('~/HCI_HS_NoFS.Rdata')
HCI_HS_NoFS_test_pred <- pred_te
rm(list = setdiff(ls(),c('Chemical_HS_NoFS_test_pred','Chemical_NoHS_FS_test_pred','Chemical_NoHS_NoFS_test_pred', 
	'HCI_HS_NoFS_test_pred')))

load('~/HCI_NoHS_FS.Rdata')
HCI_NoHS_FS_test_pred <- pred_te
rm(list = setdiff(ls(),c('Chemical_HS_NoFS_test_pred','Chemical_NoHS_FS_test_pred','Chemical_NoHS_NoFS_test_pred', 
	'HCI_HS_NoFS_test_pred','HCI_NoHS_FS_test_pred')))

load('~/HCI_descr_Model_NoHS_NoFS.Rdata')
HCI_NoHS_NoFS_test_pred <- pred_te
rm(list = setdiff(ls(),c('Chemical_HS_NoFS_test_pred','Chemical_NoHS_FS_test_pred','Chemical_NoHS_NoFS_test_pred', 
	'HCI_HS_NoFS_test_pred','HCI_NoHS_FS_test_pred', 'HCI_NoHS_NoFS_test_pred')))

load('~/Hybrid_HS_NoFS.Rdata')
Hybrid_HS_NoFS_test_pred <- pred_te
rm(list = setdiff(ls(),c('Chemical_HS_NoFS_test_pred','Chemical_NoHS_FS_test_pred','Chemical_NoHS_NoFS_test_pred', 
	'HCI_HS_NoFS_test_pred','HCI_NoHS_FS_test_pred', 'HCI_NoHS_NoFS_test_pred', 'Hybrid_HS_NoFS_test_pred')))

load('~/Hybrid_NoHS_FS.Rdata')
Hybrid_NoHS_FS_test_pred <- pred_te
rm(list = setdiff(ls(),c('Chemical_HS_NoFS_test_pred','Chemical_NoHS_FS_test_pred','Chemical_NoHS_NoFS_test_pred', 
	'HCI_HS_NoFS_test_pred','HCI_NoHS_FS_test_pred', 'HCI_NoHS_NoFS_test_pred', 'Hybrid_HS_NoFS_test_pred',
	'Hybrid_NoHS_FS_test_pred')))

load('~/hybrid_NoHS_NoFS.Rdata')
Hybrid_NoHS_NoFS_test_pred <- pred_te
rm(list = setdiff(ls(),c('Chemical_HS_NoFS_test_pred','Chemical_NoHS_FS_test_pred','Chemical_NoHS_NoFS_test_pred', 
	'HCI_HS_NoFS_test_pred','HCI_NoHS_FS_test_pred', 'HCI_NoHS_NoFS_test_pred', 'Hybrid_HS_NoFS_test_pred',
	'Hybrid_NoHS_FS_test_pred', 'Hybrid_NoHS_NoFS_test_pred')))


make_mcnemar_test <- function(first_model_pred, second_model_pred){
	common_correct <- which(first_model_pred == second_model_pred)
	A <- length(which(first_model_pred[common_correct]=='X1'))
	D <- length(which(first_model_pred[common_correct]=='X0'))
	common_wrong <- which(first_model_pred != second_model_pred)
	B <- length(which(first_model_pred[common_wrong]=='X1'))
	C <- length(which(first_model_pred[common_wrong]=='X0'))
	
	my_mat <- matrix(c(A,B,C,D), nrow=2, 
		dimnames=list(first_model_pred=c('Correct','Incorrect'),second_model_pred=c('Correct','Incorrect')))
	
	#print(my_mat)
	return (mcnemar.test(my_mat)$p.value)
}

list_all <- ls()
list_all <- list_all[list_all != "make_mcnemar_test"]

result_table <- matrix(NA,nrow = 9, ncol = 10)
colnames(result_table) <- c(list_all, 'Model_name')

for (i in 1:length(list_all)){
	col_name <- list_all[i]
	for (j in 1:length(list_all)){
		result_table[j,'Model_name'] <- c(list_all[j])
		if(i==j){
			result_table[j,col_name] <- c(NA)
		}else{
			result_table[j,col_name] <- c(make_mcnemar_test(get(list_all[i]), get(list_all[j])))
		}
	}
	print(i)
}

result_table_Test <- as.data.frame(result_table[,1:9], stringsAsFactors = F)
result_table_Test <- apply(result_table_Test, 2, function(x) as.numeric(x))
colnames(result_table_Test) <- gsub('_test_pred','',colnames(result_table_Test))
row.names(result_table_Test) <- c(gsub('_test_pred','',colnames(result_table_Test)[1:9]))

# External set

rm(list = setdiff(ls(),c('result_table_Test')))

# Get all test set predictions for all 9 models

load("~/chemical_HS_NoFS.Rdata")
Chemical_HS_NoFS_ext_pred <- pred_ext
rm(list = setdiff(ls(),c('Chemical_HS_NoFS_ext_pred','result_table_Test')))

load("~/chemical_NoHS_FS.Rdata")
Chemical_NoHS_FS_ext_pred <- pred_ext
rm(list = setdiff(ls(),c('Chemical_HS_NoFS_ext_pred','Chemical_NoHS_FS_ext_pred','result_table_Test')))

load("~/chemical_NoHS_NoFS.Rdata")
Chemical_NoHS_NoFS_ext_pred <- pred_ext
rm(list = setdiff(ls(),c('Chemical_HS_NoFS_ext_pred','Chemical_NoHS_FS_ext_pred','Chemical_NoHS_NoFS_ext_pred','result_table_Test')))

load('~/HCI_HS_NoFS.Rdata')
HCI_HS_NoFS_ext_pred <- pred_ext
rm(list = setdiff(ls(),c('Chemical_HS_NoFS_ext_pred','Chemical_NoHS_FS_ext_pred','Chemical_NoHS_NoFS_ext_pred', 
	'HCI_HS_NoFS_ext_pred','result_table_Test')))

load('~/HCI_NoHS_FS.Rdata')
HCI_NoHS_FS_ext_pred <- pred_ext
rm(list = setdiff(ls(),c('Chemical_HS_NoFS_ext_pred','Chemical_NoHS_FS_ext_pred','Chemical_NoHS_NoFS_ext_pred', 
	'HCI_HS_NoFS_ext_pred','HCI_NoHS_FS_ext_pred','result_table_Test')))

load('~/HCI_descr_Model_NoHS_NoFS.Rdata')
HCI_NoHS_NoFS_ext_pred <- pred_ext
rm(list = setdiff(ls(),c('Chemical_HS_NoFS_ext_pred','Chemical_NoHS_FS_ext_pred','Chemical_NoHS_NoFS_ext_pred', 
	'HCI_HS_NoFS_ext_pred','HCI_NoHS_FS_ext_pred', 'HCI_NoHS_NoFS_ext_pred','result_table_Test')))

load('~/Hybrid_HS_NoFS.Rdata')
Hybrid_HS_NoFS_ext_pred <- pred_ext
rm(list = setdiff(ls(),c('Chemical_HS_NoFS_ext_pred','Chemical_NoHS_FS_ext_pred','Chemical_NoHS_NoFS_ext_pred', 
	'HCI_HS_NoFS_ext_pred','HCI_NoHS_FS_ext_pred', 'HCI_NoHS_NoFS_ext_pred', 'Hybrid_HS_NoFS_ext_pred','result_table_Test')))

load('~/Hybrid_NoHS_FS.Rdata')
Hybrid_NoHS_FS_ext_pred <- pred_ext
rm(list = setdiff(ls(),c('Chemical_HS_NoFS_ext_pred','Chemical_NoHS_FS_ext_pred','Chemical_NoHS_NoFS_ext_pred', 
	'HCI_HS_NoFS_ext_pred','HCI_NoHS_FS_ext_pred', 'HCI_NoHS_NoFS_ext_pred', 'Hybrid_HS_NoFS_ext_pred',
	'Hybrid_NoHS_FS_ext_pred','result_table_Test')))

load('~/hybrid_NoHS_NoFS.Rdata')
Hybrid_NoHS_NoFS_ext_pred <- pred_ext
rm(list = setdiff(ls(),c('Chemical_HS_NoFS_ext_pred','Chemical_NoHS_FS_ext_pred','Chemical_NoHS_NoFS_ext_pred', 
	'HCI_HS_NoFS_ext_pred','HCI_NoHS_FS_ext_pred', 'HCI_NoHS_NoFS_ext_pred', 'Hybrid_HS_NoFS_ext_pred',
	'Hybrid_NoHS_FS_ext_pred', 'Hybrid_NoHS_NoFS_ext_pred','result_table_Test')))

make_mcnemar_test <- function(first_model_pred, second_model_pred){
	common_correct <- which(first_model_pred == second_model_pred)
	A <- length(which(first_model_pred[common_correct]=='X1'))
	D <- length(which(first_model_pred[common_correct]=='X0'))
	common_wrong <- which(first_model_pred != second_model_pred)
	B <- length(which(first_model_pred[common_wrong]=='X1'))
	C <- length(which(first_model_pred[common_wrong]=='X0'))
	
	my_mat <- matrix(c(A,B,C,D), nrow=2, 
		dimnames=list(first_model_pred=c('Correct','Incorrect'),second_model_pred=c('Correct','Incorrect')))
	
	#print(my_mat)
	return (mcnemar.test(my_mat)$p.value)
}


list_all <- ls()
list_all <- list_all[list_all != "make_mcnemar_test"]
list_all <- list_all[list_all != 'result_table_Test']

result_table <- matrix(NA,nrow = 9, ncol = 10)
colnames(result_table) <- c(list_all, 'Model_name')

for (i in 1:length(list_all)){
	col_name <- list_all[i]
	for (j in 1:length(list_all)){
		result_table[j,'Model_name'] <- c(list_all[j])
		if(i==j){
			result_table[j,col_name] <- c(NA)
		}else{
			result_table[j,col_name] <- c(make_mcnemar_test(get(list_all[i]), get(list_all[j])))
		}
	}
	print(i)
}

result_table_EXT <- as.data.frame(result_table[,1:9], stringsAsFactors = F)
result_table_EXT <- apply(result_table_EXT, 2, function(x) as.numeric(x))
colnames(result_table_EXT) <- gsub('_ext_pred','',colnames(result_table_EXT))
row.names(result_table_EXT) <- c(gsub('_ext_pred','',colnames(result_table)[1:9]))



# ---------------------------------------------------------------------------------------------------
#                                  Y-randomization 
# ---------------------------------------------------------------------------------------------------


library(data.table)
library(dplyr)
library(splitstackshape)
library(caret)

# A.1: Y-randomization of HCI_NoHS_NoFS

rm(list = setdiff(ls(),'path'))

setwd('~/HCI_NoHS_NoFS')

x_train <- fread("train_set_norm.csv")

permutation_results <- matrix(NA, nrow = 100, ncol = 13)
temp_names <- c('Entry','TN_CV','FP_CV','FN_CV','TP_CV','ACC_CV','BA_CV','SN_CV','SP_CV','PPV_CV','NPV_CV','RC_CV','F1_CV')
colnames(permutation_results) <- temp_names

path <- getwd()

for (i in 1:100){
	set.seed(i)
	print(i)
	path_1 <- paste(path,'permutation_',i,sep = '')
	dir.create(path_1)
	setwd(path_1)
	y_train_binary <- sample(x_train$CLASS)
	write.table(y_train_binary,file = 'permuted_labels.csv', row.names = F, col.names = F, sep = ',')
	x_train_1 <- as.data.frame(cbind(y_train_binary, x_train[,!c('Sample_id','CLASS')]))
	x_train_1$y_train_binary <- as.factor(x_train_1$y_train_binary)
	levels(x_train_1$y_train_binary) <- make.names(levels(factor(x_train_1$y_train_binary)))
	
	# RF model building
	model_weights <- ifelse(y_train_binary=='0', (1/table(y_train_binary)[1]) * 0.5, (1/table(y_train_binary)[2]) * 0.5)
	seeds <- as.vector(c(1:6), mode = "list")
	seeds[[6]] <- 1
	mtry_val = as.integer(sqrt(dim(x_train_1)[2]))
	set.seed(47)
	library(doMC)
	registerDoMC(cores = 16)
	weighted_fit <- caret::train(y_train_binary ~ .,
		data = x_train_1,
		method = "rf",
		verbose = FALSE,
		ntree=100,
		replace=FALSE, 
		nodesize = 5,
		metric = "Accuracy",
		tuneGrid = data.frame(mtry = mtry_val),
		tuneLength = 1,
		weights = model_weights,
		trControl = trainControl(method = "cv", number = 5, classProbs=T, savePredictions = "final", seeds = seeds))
	
	# CV output extraction
	all_folds <- weighted_fit$pred
	CV_results <- matrix(NA, nrow = 5, ncol = 13)
	colnames(CV_results) <- temp_names
	for(j in 1:5){
		A <- paste('Fold',j,sep = '')
		B <- all_folds[all_folds$Resample == A,]
		CV_results[j,'Entry'] <- j
		CV_results[j,'TN_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[1])
		CV_results[j,'FP_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[2])
		CV_results[j,'FN_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[3])
		CV_results[j,'TP_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[4])
		CV_results[j,'ACC_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$overall['Accuracy'])
		CV_results[j,'BA_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Balanced Accuracy'])
		CV_results[j,'SN_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Sensitivity'])
		CV_results[j,'SP_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Specificity'])
		CV_results[j,'PPV_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Pos Pred Value'])
		CV_results[j,'NPV_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Neg Pred Value'])
		CV_results[j,'RC_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Recall'])
		CV_results[j,'F1_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['F1'])
	}
	
	CV_results_final <- c(apply(CV_results[,2:5], 2, function(x) ceiling(mean(x))),apply(CV_results[,6:13], 2, mean))
	CV_results_final <-round(CV_results_final,2)
	permutation_results[i,'Entry'] <- c(i)
	permutation_results[i,2:13] <- CV_results_final
	setwd(path)
	write.table(permutation_results,file = 'permutation_results_HCI_NoHS_NoFS.csv', row.names = F, col.names = T, sep = ',')
}

# p value
#count <- length(which(permutation_results[,'ACC_CV']>=0.77))
count <- length(which(permutation_results[,'BA_CV']>=0.56))
pvalue = (count + 1.0) / (100 + 1)


# A.2: Y-randomization of HCI_HS_NoFS

library(randomForest)

setwd('~/HCI_HS_NoFS')

x_train <- fread("train_set_norm.csv")

path <- getwd()

result_table <- matrix(NA,nrow = 100, ncol = 16)
colnames(result_table) <- c('Model','mtry','ntree','nodesize','TN_CV','FP_CV','FN_CV','TP_CV','ACC_CV','BA_CV','SN_CV','SP_CV','PPV_CV','NPV_CV','RC_CV','F1_CV')

for (i in 1:100){
	set.seed(i)
	print(i)
	path_1 <- paste(path,'permutation_',i,sep = '')
	dir.create(path_1)
	setwd(path_1)
	y_train_binary <- sample(x_train$CLASS)
	write.table(y_train_binary,file = 'permuted_labels.csv', row.names = F, col.names = F, sep = ',')
	x_train_1 <- as.data.frame(cbind(y_train_binary, x_train[,!c('Sample_id','CLASS')]))
	x_train_1$y_train_binary <- as.factor(x_train_1$y_train_binary)
	levels(x_train_1$y_train_binary) <- make.names(levels(factor(x_train_1$y_train_binary)))
	
	# RF model building
	model_weights <- ifelse(y_train_binary=='0', (1/table(y_train_binary)[1]) * 0.5, (1/table(y_train_binary)[2]) * 0.5)
	
	set.seed(47)
	customRF <- list(type = "Classification", library = "randomForest", loop = NULL)
	customRF$parameters <- data.frame(parameter = c("mtry", "ntree", "nodesize"), class = rep("numeric", 3), label = c("mtry", "ntree", "nodesize"))
	customRF$grid <- function(x, y, len = NULL, search = "grid") {}
	customRF$fit <- function(x, y, wts, param, lev, last, weights, classProbs, ...) {
		randomForest(x, y, mtry = param$mtry, ntree=param$ntree, ...)
	}
	customRF$predict <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
		predict(modelFit, newdata)
	customRF$prob <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
		predict(modelFit, newdata, type = "prob")
	customRF$sort <- function(x) x[order(x[,1]),]
	customRF$levels <- function(x) x$classes
	
	library(doMC)
	registerDoMC(cores = 32)
	tunegrid <- data.frame(expand.grid(.mtry=c(30,35,40,45,50), .ntree=c(50, 100,500), .nodesize=c(1:5)))
	weighted_fit <- caret::train(y_train_binary ~ .,
		data = x_train_1,
		method = customRF,
		importance = TRUE,
		verbose = FALSE,
		replace=FALSE,
		metric = "Accuracy",
		tuneGrid = tunegrid,
		weights = model_weights,
		trControl = trainControl(method = "cv", number = 5, classProbs=T, savePredictions = "final", allowParallel = TRUE))
	
	# save results
	result_table[i,'Model'] <- i
	result_table[i,'mtry'] <- weighted_fit$bestTune$mtry
	result_table[i,'ntree'] <- weighted_fit$bestTune$ntree
	result_table[i,'nodesize'] <- weighted_fit$bestTune$nodesize
	
	# CV output extraction
	all_folds <- weighted_fit$pred
	CV_results <- matrix(NA, nrow = 5, ncol = 13)
	colnames(CV_results) <- c('Fold','TN_CV','FP_CV','FN_CV','TP_CV','ACC_CV','BA_CV','SN_CV','SP_CV','PPV_CV','NPV_CV','RC_CV','F1_CV')
	for(j in 1:5){
		A <- paste('Fold',j,sep = '')
		B <- all_folds[all_folds$Resample == A,]
		CV_results[j,'Fold'] <- j
		CV_results[j,'TN_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[1])
		CV_results[j,'FP_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[2])
		CV_results[j,'FN_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[3])
		CV_results[j,'TP_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[4])
		CV_results[j,'ACC_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$overall['Accuracy'])
		CV_results[j,'BA_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Balanced Accuracy'])
		CV_results[j,'SN_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Sensitivity'])
		CV_results[j,'SP_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Specificity'])
		CV_results[j,'PPV_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Pos Pred Value'])
		CV_results[j,'NPV_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Neg Pred Value'])
		CV_results[j,'RC_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Recall'])
		CV_results[j,'F1_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['F1'])
	}
	
	CV_results_final <- colMeans(CV_results[,2:13], na.rm=TRUE)
	CV_results_final[1:4] <- round(CV_results_final[1:4], 0)
	CV_results_final[5:12] <- round(CV_results_final[5:12], 2)
	
	result_table[i,5:16] <- CV_results_final
	setwd(path)
	write.table(result_table,file = 'perm_results_HCI_HS_NoFS.csv', row.names = F, col.names = T, sep = ',')
}

# p value
#count <- length(which(result_table[,'ACC_CV']>=0.77))
count <- length(which(permutation_results[,'BA_CV']>=0.54))
pvalue = (count + 1.0) / (100 + 1)


# A.3: Y-randomization of HCI_NoHS_FS

library(randomForest)
library(Boruta)

setwd('~/HCI_NoHS_FS')
x_train_orig <- fread("Orig_train_set.csv")
# pre-processing train set 
x_train <- as.data.frame(x_train_orig[,!c('Sample_id', 'CLASS')])

# save the MEAN & STD. of x_train, later will be used for normalization of test set
mean_train <- apply(x_train, 2, mean, na.rm = TRUE)
std_train <- apply(x_train, 2, sd, na.rm = TRUE)

# center
x_train <- scale(x_train, center = T, scale = F)
# scale
x_train <- scale(x_train, center = FALSE, scale = apply(x_train, 2, sd, na.rm = TRUE))

path <- setwd(~HCI_NoHS_FS)

result_table <- matrix(NA,nrow = 100, ncol = 14)
colnames(result_table) <- c('Model','var_count','TN_CV','FP_CV','FN_CV','TP_CV','ACC_CV','BA_CV','SN_CV','SP_CV','PPV_CV','NPV_CV','RC_CV','F1_CV')

for(i in 1:100){
	set.seed(i)
	print(i)
	path_1 <- paste(path,'permutation_',i,sep = '')
	dir.create(path_1)
	setwd(path_1)
	y_train_binary <- sample(x_train_orig$CLASS)
	write.table(y_train_binary,file = 'permuted_labels.csv', row.names = F, col.names = F, sep = ',')
	
	library(doMC)
	registerDoMC(cores = 32)
	
	FS_boruta_model <- Boruta(x_train, y_train_binary, pValue = 0.05, mcAdj = TRUE, maxRuns = 100)
	best_var <- colnames(x_train)[which(FS_boruta_model$finalDecision == 'Confirmed')]
	# final train & test set
	if(length(best_var) != 0){
		if(length(best_var)==1){
			x_train <- data.frame(x_train[,best_var])
			colnames(x_train) <- best_var
		}else{
			x_train <- x_train[,best_var]
		}
		x_train_1 <- as.data.frame(cbind(y_train_binary,x_train))
		x_train_1$y_train_binary <- as.factor(x_train_1$y_train_binary)
		levels(x_train_1$y_train_binary) <- make.names(levels(factor(x_train_1$y_train_binary)))
		
		# RF model building
		model_weights <- ifelse(y_train_binary=='0', (1/table(y_train_binary)[1]) * 0.5, (1/table(y_train_binary)[2]) * 0.5)
		seeds <- as.vector(c(1:6), mode = "list")
		seeds[[6]] <- 1
		mtry_val = as.integer(sqrt(dim(x_train_1)[2]))
		
		set.seed(47)
		
		library(doMC)
		registerDoMC(cores = 32)
		
		weighted_fit <- caret::train(y_train_binary ~ .,
			data = x_train_1,
			method = "rf",
			importance = TRUE,
			verbose = FALSE,
			ntree=100,
			replace=FALSE, 
			nodesize = 5,
			metric = "Accuracy",
			tuneGrid = data.frame(mtry = mtry_val),
			tuneLength = 1,
			weights = model_weights,
			trControl = trainControl(method = "cv", number = 5, classProbs=T, savePredictions = "final", seeds = seeds))
		
		# CV output extraction
		all_folds <- weighted_fit$pred
		CV_results <- matrix(NA, nrow = 5, ncol = 13)
		colnames(CV_results) <- c('Fold','TN_CV','FP_CV','FN_CV','TP_CV','ACC_CV','BA_CV','SN_CV','SP_CV',
			'PPV_CV','NPV_CV','RC_CV','F1_CV')
		for(j in 1:5){
			A <- paste('Fold',j,sep = '')
			B <- all_folds[all_folds$Resample == A,]
			CV_results[j,'Fold'] <- j
			CV_results[j,'TN_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[1])
			CV_results[j,'FP_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[2])
			CV_results[j,'FN_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[3])
			CV_results[j,'TP_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[4])
			CV_results[j,'ACC_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$overall['Accuracy'])
			CV_results[j,'BA_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Balanced Accuracy'])
			CV_results[j,'SN_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Sensitivity'])
			CV_results[j,'SP_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Specificity'])
			CV_results[j,'PPV_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Pos Pred Value'])
			CV_results[j,'NPV_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Neg Pred Value'])
			CV_results[j,'RC_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Recall'])
			CV_results[j,'F1_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['F1'])
		}
		
		CV_results_final <- colMeans(CV_results[,2:13], na.rm=TRUE)
		CV_results_final[1:4] <- round(CV_results_final[1:4], 0)
		CV_results_final[5:12] <- round(CV_results_final[5:12], 2)
		# add to final result
		result_table[i,'Model'] <- i
		result_table[i,'var_count'] <- length(best_var)
		result_table[i,3:14] <- CV_results_final
	}else{
		result_table[i,'Model'] <- i
		result_table[i,2:dim(result_table)[2]] <- NA
	}
	setwd(path)
	write.table(result_table,file = 'perm_results_HCI_NoHS_FS.csv', row.names = F, col.names = T, sep = ',')
}

# p value
#count <- length(which(result_table[,'ACC_CV']>=0.77))
count <- length(which(permutation_results[,'BA_CV']>=0.59))
pvalue = (count + 1.0) / (100 + 1)


# B.1: Y-randomization of chemical_NoHS_NoFS

setwd('~/chemical_NoHS_NoFS')

x_train <- fread("train_set_norm.csv")

path <- getwd()

permutation_results <- matrix(NA, nrow = 100, ncol = 13)
temp_names <- c('Entry','TN_CV','FP_CV','FN_CV','TP_CV','ACC_CV','BA_CV','SN_CV','SP_CV','PPV_CV','NPV_CV','RC_CV','F1_CV')
colnames(permutation_results) <- temp_names

for (i in 1:100){
	set.seed(i)
	print(i)
	path_1 <- paste(path,'permutation_',i,sep = '')
	dir.create(path_1)
	setwd(path_1)
	y_train_binary <- sample(x_train$CLASS)
	write.table(y_train_binary,file = 'permuted_labels.csv', row.names = F, col.names = F, sep = ',')
	x_train_1 <- as.data.frame(cbind(y_train_binary, x_train[,!c('Sample_id','CLASS')]))
	x_train_1$y_train_binary <- as.factor(x_train_1$y_train_binary)
	levels(x_train_1$y_train_binary) <- make.names(levels(factor(x_train_1$y_train_binary)))
	
	# RF model building
	model_weights <- ifelse(y_train_binary=='0', (1/table(y_train_binary)[1]) * 0.5, (1/table(y_train_binary)[2]) * 0.5)
	seeds <- as.vector(c(1:6), mode = "list")
	seeds[[6]] <- 1
	mtry_val = as.integer(sqrt(dim(x_train_1)[2]))
	set.seed(47)
	library(doMC)
	registerDoMC(cores = 32)
	weighted_fit <- caret::train(y_train_binary ~ .,
		data = x_train_1,
		method = "rf",
		verbose = FALSE,
		ntree=100,
		replace=FALSE, 
		nodesize = 5,
		metric = "Accuracy",
		tuneGrid = data.frame(mtry = mtry_val),
		tuneLength = 1,
		weights = model_weights,
		trControl = trainControl(method = "cv", number = 5, classProbs=T, savePredictions = "final", seeds = seeds))
	
	# CV output extraction
	all_folds <- weighted_fit$pred
	CV_results <- matrix(NA, nrow = 5, ncol = 13)
	colnames(CV_results) <- temp_names
	for(j in 1:5){
		A <- paste('Fold',j,sep = '')
		B <- all_folds[all_folds$Resample == A,]
		CV_results[j,'Entry'] <- j
		CV_results[j,'TN_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[1])
		CV_results[j,'FP_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[2])
		CV_results[j,'FN_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[3])
		CV_results[j,'TP_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[4])
		CV_results[j,'ACC_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$overall['Accuracy'])
		CV_results[j,'BA_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Balanced Accuracy'])
		CV_results[j,'SN_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Sensitivity'])
		CV_results[j,'SP_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Specificity'])
		CV_results[j,'PPV_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Pos Pred Value'])
		CV_results[j,'NPV_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Neg Pred Value'])
		CV_results[j,'RC_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Recall'])
		CV_results[j,'F1_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['F1'])
	}
	
	CV_results_final <- c(apply(CV_results[,2:5], 2, function(x) ceiling(mean(x))),apply(CV_results[,6:13], 2, mean))
	CV_results_final <-round(CV_results_final,2)
	permutation_results[i,'Entry'] <- c(i)
	permutation_results[i,2:13] <- CV_results_final
	setwd(path)
	write.table(permutation_results,file = 'permutation_results_chemical_NoHS_NoFS', row.names = F, col.names = T, sep = ',')
}

# p value
#count <- length(which(permutation_results[,'ACC_CV']>=0.80))
count <- length(which(permutation_results[,'BA_CV']>=0.60))
pvalue = (count + 1.0) / (100 + 1)


# B.2: Y-randomization of chemical_HS_NoFS

library(randomForest)

setwd('~/chemical_HS_NoFS')

x_train <- fread("train_set_norm.csv")

path <- getwd()

result_table <- matrix(NA,nrow = 100, ncol = 16)
colnames(result_table) <- c('Model','mtry','ntree','nodesize','TN_CV','FP_CV','FN_CV','TP_CV','ACC_CV','BA_CV','SN_CV','SP_CV','PPV_CV','NPV_CV','RC_CV','F1_CV')

for (i in 1:100){
	set.seed(i)
	print(i)
	path_1 <- paste(path,'permutation_',i,sep = '')
	dir.create(path_1)
	setwd(path_1)
	y_train_binary <- sample(x_train$CLASS)
	write.table(y_train_binary,file = 'permuted_labels.csv', row.names = F, col.names = F, sep = ',')
	x_train_1 <- as.data.frame(cbind(y_train_binary, x_train[,!c('Sample_id','CLASS')]))
	x_train_1$y_train_binary <- as.factor(x_train_1$y_train_binary)
	levels(x_train_1$y_train_binary) <- make.names(levels(factor(x_train_1$y_train_binary)))
	
	# RF model building
	model_weights <- ifelse(y_train_binary=='0', (1/table(y_train_binary)[1]) * 0.5, (1/table(y_train_binary)[2]) * 0.5)
	
	set.seed(47)
	customRF <- list(type = "Classification", library = "randomForest", loop = NULL)
	customRF$parameters <- data.frame(parameter = c("mtry", "ntree", "nodesize"), class = rep("numeric", 3), label = c("mtry", "ntree", "nodesize"))
	customRF$grid <- function(x, y, len = NULL, search = "grid") {}
	customRF$fit <- function(x, y, wts, param, lev, last, weights, classProbs, ...) {
		randomForest(x, y, mtry = param$mtry, ntree=param$ntree, ...)
	}
	customRF$predict <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
		predict(modelFit, newdata)
	customRF$prob <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
		predict(modelFit, newdata, type = "prob")
	customRF$sort <- function(x) x[order(x[,1]),]
	customRF$levels <- function(x) x$classes
	
	library(doMC)
	registerDoMC(cores = 32)
	tunegrid <- data.frame(expand.grid(.mtry=c(20,25,30,35,40), .ntree=c(50, 100,500), .nodesize=c(1:5)))
	weighted_fit <- caret::train(y_train_binary ~ .,
		data = x_train_1,
		method = customRF,
		importance = TRUE,
		verbose = FALSE,
		replace=FALSE,
		metric = "Accuracy",
		tuneGrid = tunegrid,
		weights = model_weights,
		trControl = trainControl(method = "cv", number = 5, classProbs=T, savePredictions = "final", allowParallel = TRUE))
	
	# save results
	result_table[i,'Model'] <- i
	result_table[i,'mtry'] <- weighted_fit$bestTune$mtry
	result_table[i,'ntree'] <- weighted_fit$bestTune$ntree
	result_table[i,'nodesize'] <- weighted_fit$bestTune$nodesize
	
	# CV output extraction
	all_folds <- weighted_fit$pred
	CV_results <- matrix(NA, nrow = 5, ncol = 13)
	colnames(CV_results) <- c('Fold','TN_CV','FP_CV','FN_CV','TP_CV','ACC_CV','BA_CV','SN_CV','SP_CV','PPV_CV','NPV_CV','RC_CV','F1_CV')
	for(j in 1:5){
		A <- paste('Fold',j,sep = '')
		B <- all_folds[all_folds$Resample == A,]
		CV_results[j,'Fold'] <- j
		CV_results[j,'TN_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[1])
		CV_results[j,'FP_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[2])
		CV_results[j,'FN_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[3])
		CV_results[j,'TP_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[4])
		CV_results[j,'ACC_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$overall['Accuracy'])
		CV_results[j,'BA_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Balanced Accuracy'])
		CV_results[j,'SN_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Sensitivity'])
		CV_results[j,'SP_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Specificity'])
		CV_results[j,'PPV_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Pos Pred Value'])
		CV_results[j,'NPV_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Neg Pred Value'])
		CV_results[j,'RC_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Recall'])
		CV_results[j,'F1_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['F1'])
	}
	
	CV_results_final <- colMeans(CV_results[,2:13], na.rm=TRUE)
	CV_results_final[1:4] <- round(CV_results_final[1:4], 0)
	CV_results_final[5:12] <- round(CV_results_final[5:12], 2)
	
	result_table[i,5:16] <- CV_results_final
	setwd(path)
	write.table(result_table,file = 'perm_results_chemical_HS_NoFS.csv', row.names = F, col.names = T, sep = ',')
}

# p value
#count <- length(which(result_table[,'ACC_CV']>=0.83))
count <- length(which(permutation_results[,'BA_CV']>=0.66))
pvalue = (count + 1.0) / (100 + 1)


# B.3: Y-randomization of Chemical_NoHS_FS

library(randomForest)
library(Boruta)

setwd('~/Chemical_NoHS_FS')
x_train_orig <- fread("Orig_train_set.csv")
# pre-processing train set 
x_train <- as.data.frame(x_train_orig[,!c('Sample_id', 'CLASS')])

# save the MEAN & STD. of x_train, later will be used for normalization of test set
mean_train <- apply(x_train, 2, mean, na.rm = TRUE)
std_train <- apply(x_train, 2, sd, na.rm = TRUE)

# center
x_train <- scale(x_train, center = T, scale = F)
# scale
x_train <- scale(x_train, center = FALSE, scale = apply(x_train, 2, sd, na.rm = TRUE))

path <- setwd()

result_table <- matrix(NA,nrow = 100, ncol = 14)
colnames(result_table) <- c('Model','var_count','TN_CV','FP_CV','FN_CV','TP_CV','ACC_CV','BA_CV','SN_CV','SP_CV','PPV_CV','NPV_CV','RC_CV','F1_CV')

for(i in 1:100){
	set.seed(i)
	print(i)
	path_1 <- paste(path,'permutation_',i,sep = '')
	dir.create(path_1)
	setwd(path_1)
	y_train_binary <- sample(x_train_orig$CLASS)
	write.table(y_train_binary,file = 'permuted_labels.csv', row.names = F, col.names = F, sep = ',')
	
	library(doMC)
	registerDoMC(cores = 32)
	
	FS_boruta_model <- Boruta(x_train, y_train_binary, pValue = 0.05, mcAdj = TRUE, maxRuns = 100)
	best_var <- colnames(x_train)[which(FS_boruta_model$finalDecision == 'Confirmed')]
	# final train & test set
	if(length(best_var) != 0){
		if(length(best_var)==1){
			x_train <- data.frame(x_train[,best_var])
			colnames(x_train) <- best_var
		}else{
			x_train <- x_train[,best_var]
		}
		x_train_1 <- as.data.frame(cbind(y_train_binary,x_train))
		x_train_1$y_train_binary <- as.factor(x_train_1$y_train_binary)
		levels(x_train_1$y_train_binary) <- make.names(levels(factor(x_train_1$y_train_binary)))
		
		# RF model building
		model_weights <- ifelse(y_train_binary=='0', (1/table(y_train_binary)[1]) * 0.5, (1/table(y_train_binary)[2]) * 0.5)
		seeds <- as.vector(c(1:6), mode = "list")
		seeds[[6]] <- 1
		mtry_val = as.integer(sqrt(dim(x_train_1)[2]))
		
		set.seed(47)
		
		library(doMC)
		registerDoMC(cores = 32)
		
		weighted_fit <- caret::train(y_train_binary ~ .,
			data = x_train_1,
			method = "rf",
			importance = TRUE,
			verbose = FALSE,
			ntree=100,
			replace=FALSE, 
			nodesize = 5,
			metric = "Accuracy",
			tuneGrid = data.frame(mtry = mtry_val),
			tuneLength = 1,
			weights = model_weights,
			trControl = trainControl(method = "cv", number = 5, classProbs=T, savePredictions = "final", seeds = seeds))
		
		# CV output extraction
		all_folds <- weighted_fit$pred
		CV_results <- matrix(NA, nrow = 5, ncol = 13)
		colnames(CV_results) <- c('Fold','TN_CV','FP_CV','FN_CV','TP_CV','ACC_CV','BA_CV','SN_CV','SP_CV',
			'PPV_CV','NPV_CV','RC_CV','F1_CV')
		for(j in 1:5){
			A <- paste('Fold',j,sep = '')
			B <- all_folds[all_folds$Resample == A,]
			CV_results[j,'Fold'] <- j
			CV_results[j,'TN_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[1])
			CV_results[j,'FP_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[2])
			CV_results[j,'FN_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[3])
			CV_results[j,'TP_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[4])
			CV_results[j,'ACC_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$overall['Accuracy'])
			CV_results[j,'BA_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Balanced Accuracy'])
			CV_results[j,'SN_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Sensitivity'])
			CV_results[j,'SP_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Specificity'])
			CV_results[j,'PPV_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Pos Pred Value'])
			CV_results[j,'NPV_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Neg Pred Value'])
			CV_results[j,'RC_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Recall'])
			CV_results[j,'F1_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['F1'])
		}
		
		CV_results_final <- colMeans(CV_results[,2:13], na.rm=TRUE)
		CV_results_final[1:4] <- round(CV_results_final[1:4], 0)
		CV_results_final[5:12] <- round(CV_results_final[5:12], 2)
		# add to final result
		result_table[i,'Model'] <- i
		result_table[i,'var_count'] <- length(best_var)
		result_table[i,3:14] <- CV_results_final
	}else{
		result_table[i,'Model'] <- i
		result_table[i,2:dim(result_table)[2]] <- NA
	}
	setwd(path)
	write.table(result_table,file = 'perm_results_Chemical_NoHS_FS.csv', row.names = F, col.names = T, sep = ',')
}

# p value
#count <- length(which(result_table[,'ACC_CV']>=0.82))
count <- length(which(permutation_results[,'BA_CV']>=0.65))
pvalue = (count + 1.0) / (100 + 1)


# C.1: Y-randomization of Hybrid_NoHS_NoFS

setwd('~/Hybrid_NoHS_NoFS')

x_train <- fread("train_set_norm.csv")

path <- getwd()

permutation_results <- matrix(NA, nrow = 100, ncol = 13)
temp_names <- c('Entry','TN_CV','FP_CV','FN_CV','TP_CV','ACC_CV','BA_CV','SN_CV','SP_CV','PPV_CV','NPV_CV','RC_CV','F1_CV')
colnames(permutation_results) <- temp_names

for (i in 1:100){
	set.seed(i)
	print(i)
	path_1 <- paste(path,'permutation_',i,sep = '')
	dir.create(path_1)
	setwd(path_1)
	y_train_binary <- sample(x_train$CLASS)
	write.table(y_train_binary,file = 'permuted_labels.csv', row.names = F, col.names = F, sep = ',')
	x_train_1 <- as.data.frame(cbind(y_train_binary, x_train[,!c('Sample_id','CLASS')]))
	x_train_1$y_train_binary <- as.factor(x_train_1$y_train_binary)
	levels(x_train_1$y_train_binary) <- make.names(levels(factor(x_train_1$y_train_binary)))
	
	# RF model building
	model_weights <- ifelse(y_train_binary=='0', (1/table(y_train_binary)[1]) * 0.5, (1/table(y_train_binary)[2]) * 0.5)
	seeds <- as.vector(c(1:6), mode = "list")
	seeds[[6]] <- 1
	mtry_val = as.integer(sqrt(dim(x_train_1)[2]))
	set.seed(47)
	library(doMC)
	registerDoMC(cores = 32)
	weighted_fit <- caret::train(y_train_binary ~ .,
		data = x_train_1,
		method = "rf",
		verbose = FALSE,
		ntree=100,
		replace=FALSE, 
		nodesize = 5,
		metric = "Accuracy",
		tuneGrid = data.frame(mtry = mtry_val),
		tuneLength = 1,
		weights = model_weights,
		trControl = trainControl(method = "cv", number = 5, classProbs=T, savePredictions = "final", seeds = seeds))
	
	# CV output extraction
	all_folds <- weighted_fit$pred
	CV_results <- matrix(NA, nrow = 5, ncol = 13)
	colnames(CV_results) <- temp_names
	for(j in 1:5){
		A <- paste('Fold',j,sep = '')
		B <- all_folds[all_folds$Resample == A,]
		CV_results[j,'Entry'] <- j
		CV_results[j,'TN_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[1])
		CV_results[j,'FP_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[2])
		CV_results[j,'FN_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[3])
		CV_results[j,'TP_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[4])
		CV_results[j,'ACC_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$overall['Accuracy'])
		CV_results[j,'BA_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Balanced Accuracy'])
		CV_results[j,'SN_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Sensitivity'])
		CV_results[j,'SP_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Specificity'])
		CV_results[j,'PPV_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Pos Pred Value'])
		CV_results[j,'NPV_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Neg Pred Value'])
		CV_results[j,'RC_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Recall'])
		CV_results[j,'F1_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['F1'])
	}
	
	CV_results_final <- c(apply(CV_results[,2:5], 2, function(x) ceiling(mean(x))),apply(CV_results[,6:13], 2, mean))
	CV_results_final <-round(CV_results_final,2)
	permutation_results[i,'Entry'] <- c(i)
	permutation_results[i,2:13] <- CV_results_final
	setwd(path)
	write.table(permutation_results,file = 'permutation_results_Hybrid_NoHS_NoFS', row.names = F, col.names = T, sep = ',')
}

# p value
#count <- length(which(permutation_results[,'ACC_CV']>=0.78))
count <- length(which(permutation_results[,'BA_CV']>=0.53))
pvalue = (count + 1.0) / (100 + 1)


# C.2: Y-randomization of Hybrid_HS_NoFS

library(randomForest)

setwd('~/Hybrid_HS_NoFS')

x_train <- fread("train_set_norm.csv")

path <- setwd()

result_table <- matrix(NA,nrow = 100, ncol = 16)
colnames(result_table) <- c('Model','mtry','ntree','nodesize','TN_CV','FP_CV','FN_CV','TP_CV','ACC_CV','BA_CV','SN_CV','SP_CV','PPV_CV','NPV_CV','RC_CV','F1_CV')

for (i in 1:100){
	set.seed(i)
	print(i)
	path_1 <- paste(path,'permutation_',i,sep = '')
	dir.create(path_1)
	setwd(path_1)
	y_train_binary <- sample(x_train$CLASS)
	write.table(y_train_binary,file = 'permuted_labels.csv', row.names = F, col.names = F, sep = ',')
	x_train_1 <- as.data.frame(cbind(y_train_binary, x_train[,!c('Sample_id','CLASS')]))
	x_train_1$y_train_binary <- as.factor(x_train_1$y_train_binary)
	levels(x_train_1$y_train_binary) <- make.names(levels(factor(x_train_1$y_train_binary)))
	
	# RF model building
	model_weights <- ifelse(y_train_binary=='0', (1/table(y_train_binary)[1]) * 0.5, (1/table(y_train_binary)[2]) * 0.5)
	
	set.seed(47)
	customRF <- list(type = "Classification", library = "randomForest", loop = NULL)
	customRF$parameters <- data.frame(parameter = c("mtry", "ntree", "nodesize"), class = rep("numeric", 3), label = c("mtry", "ntree", "nodesize"))
	customRF$grid <- function(x, y, len = NULL, search = "grid") {}
	customRF$fit <- function(x, y, wts, param, lev, last, weights, classProbs, ...) {
		randomForest(x, y, mtry = param$mtry, ntree=param$ntree, ...)
	}
	customRF$predict <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
		predict(modelFit, newdata)
	customRF$prob <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
		predict(modelFit, newdata, type = "prob")
	customRF$sort <- function(x) x[order(x[,1]),]
	customRF$levels <- function(x) x$classes
	
	library(doMC)
	registerDoMC(cores = 32)
	tunegrid <- data.frame(expand.grid(.mtry=c(40,45,50,55,60), .ntree=c(50, 100,500), .nodesize=c(1:5)))
	weighted_fit <- caret::train(y_train_binary ~ .,
		data = x_train_1,
		method = customRF,
		importance = TRUE,
		verbose = FALSE,
		replace=FALSE,
		metric = "Accuracy",
		tuneGrid = tunegrid,
		weights = model_weights,
		trControl = trainControl(method = "cv", number = 5, classProbs=T, savePredictions = "final", allowParallel = TRUE))
	
	# save results
	result_table[i,'Model'] <- i
	result_table[i,'mtry'] <- weighted_fit$bestTune$mtry
	result_table[i,'ntree'] <- weighted_fit$bestTune$ntree
	result_table[i,'nodesize'] <- weighted_fit$bestTune$nodesize
	
	# CV output extraction
	all_folds <- weighted_fit$pred
	CV_results <- matrix(NA, nrow = 5, ncol = 13)
	colnames(CV_results) <- c('Fold','TN_CV','FP_CV','FN_CV','TP_CV','ACC_CV','BA_CV','SN_CV','SP_CV','PPV_CV','NPV_CV','RC_CV','F1_CV')
	for(j in 1:5){
		A <- paste('Fold',j,sep = '')
		B <- all_folds[all_folds$Resample == A,]
		CV_results[j,'Fold'] <- j
		CV_results[j,'TN_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[1])
		CV_results[j,'FP_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[2])
		CV_results[j,'FN_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[3])
		CV_results[j,'TP_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[4])
		CV_results[j,'ACC_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$overall['Accuracy'])
		CV_results[j,'BA_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Balanced Accuracy'])
		CV_results[j,'SN_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Sensitivity'])
		CV_results[j,'SP_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Specificity'])
		CV_results[j,'PPV_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Pos Pred Value'])
		CV_results[j,'NPV_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Neg Pred Value'])
		CV_results[j,'RC_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Recall'])
		CV_results[j,'F1_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['F1'])
	}
	
	CV_results_final <- colMeans(CV_results[,2:13], na.rm=TRUE)
	CV_results_final[1:4] <- round(CV_results_final[1:4], 0)
	CV_results_final[5:12] <- round(CV_results_final[5:12], 2)
	
	result_table[i,5:16] <- CV_results_final
	setwd(path)
	write.table(result_table,file = 'perm_results_Hybrid_HS_NoFS.csv', row.names = F, col.names = T, sep = ',')
}

# p value
#count <- length(which(result_table[,'ACC_CV']>=0.79))
count <- length(which(permutation_results[,'BA_CV']>=0.59))
pvalue = (count + 1.0) / (100 + 1)


# C.3: Y-randomization of Hybrid_NoHS_FS

library(randomForest)
library(Boruta)

setwd('~/Hybrid_NoHS_FS')

x_train_orig <- fread("Orig_train_set.csv")
# pre-processing train set 
x_train <- as.data.frame(x_train_orig[,!c('Sample_id', 'CLASS')])

# save the MEAN & STD. of x_train, later will be used for normalization of test set
mean_train <- apply(x_train, 2, mean, na.rm = TRUE)
std_train <- apply(x_train, 2, sd, na.rm = TRUE)

# center
x_train <- scale(x_train, center = T, scale = F)
# scale
x_train <- scale(x_train, center = FALSE, scale = apply(x_train, 2, sd, na.rm = TRUE))

path <- getwd()

result_table <- matrix(NA,nrow = 100, ncol = 14)
colnames(result_table) <- c('Model','var_count','TN_CV','FP_CV','FN_CV','TP_CV','ACC_CV','BA_CV','SN_CV','SP_CV','PPV_CV','NPV_CV','RC_CV','F1_CV')

for(i in 1:100){
	set.seed(i)
	print(i)
	path_1 <- paste(path,'permutation_',i,sep = '')
	dir.create(path_1)
	setwd(path_1)
	y_train_binary <- sample(x_train_orig$CLASS)
	write.table(y_train_binary,file = 'permuted_labels.csv', row.names = F, col.names = F, sep = ',')
	
	library(doMC)
	registerDoMC(cores = 32)
	
	FS_boruta_model <- Boruta(x_train, y_train_binary, pValue = 0.05, mcAdj = TRUE, maxRuns = 100)
	best_var <- colnames(x_train)[which(FS_boruta_model$finalDecision == 'Confirmed')]
	# final train & test set
	if(length(best_var) != 0){
		if(length(best_var)==1){
			x_train <- data.frame(x_train[,best_var])
			colnames(x_train) <- best_var
		}else{
			x_train <- x_train[,best_var]
		}
		x_train_1 <- as.data.frame(cbind(y_train_binary,x_train))
		x_train_1$y_train_binary <- as.factor(x_train_1$y_train_binary)
		levels(x_train_1$y_train_binary) <- make.names(levels(factor(x_train_1$y_train_binary)))
		
		# RF model building
		model_weights <- ifelse(y_train_binary=='0', (1/table(y_train_binary)[1]) * 0.5, (1/table(y_train_binary)[2]) * 0.5)
		seeds <- as.vector(c(1:6), mode = "list")
		seeds[[6]] <- 1
		mtry_val = as.integer(sqrt(dim(x_train_1)[2]))
		
		set.seed(47)
		
		library(doMC)
		registerDoMC(cores = 32)
		
		weighted_fit <- caret::train(y_train_binary ~ .,
			data = x_train_1,
			method = "rf",
			importance = TRUE,
			verbose = FALSE,
			ntree=100,
			replace=FALSE, 
			nodesize = 5,
			metric = "Accuracy",
			tuneGrid = data.frame(mtry = mtry_val),
			tuneLength = 1,
			weights = model_weights,
			trControl = trainControl(method = "cv", number = 5, classProbs=T, savePredictions = "final", seeds = seeds))
		
		# CV output extraction
		all_folds <- weighted_fit$pred
		CV_results <- matrix(NA, nrow = 5, ncol = 13)
		colnames(CV_results) <- c('Fold','TN_CV','FP_CV','FN_CV','TP_CV','ACC_CV','BA_CV','SN_CV','SP_CV',
			'PPV_CV','NPV_CV','RC_CV','F1_CV')
		for(j in 1:5){
			A <- paste('Fold',j,sep = '')
			B <- all_folds[all_folds$Resample == A,]
			CV_results[j,'Fold'] <- j
			CV_results[j,'TN_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[1])
			CV_results[j,'FP_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[2])
			CV_results[j,'FN_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[3])
			CV_results[j,'TP_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$table[4])
			CV_results[j,'ACC_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$overall['Accuracy'])
			CV_results[j,'BA_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Balanced Accuracy'])
			CV_results[j,'SN_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Sensitivity'])
			CV_results[j,'SP_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Specificity'])
			CV_results[j,'PPV_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Pos Pred Value'])
			CV_results[j,'NPV_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Neg Pred Value'])
			CV_results[j,'RC_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['Recall'])
			CV_results[j,'F1_CV'] <- as.numeric(confusionMatrix(B$pred, B$obs, positive = 'X1')$byClass['F1'])
		}
		
		CV_results_final <- colMeans(CV_results[,2:13], na.rm=TRUE)
		CV_results_final[1:4] <- round(CV_results_final[1:4], 0)
		CV_results_final[5:12] <- round(CV_results_final[5:12], 2)
		# add to final result
		result_table[i,'Model'] <- i
		result_table[i,'var_count'] <- length(best_var)
		result_table[i,3:14] <- CV_results_final
	}else{
		result_table[i,'Model'] <- i
		result_table[i,2:dim(result_table)[2]] <- NA
	}
	setwd(path)
	write.table(result_table,file = 'perm_results_Hybrid_NoHS_FS.csv', row.names = F, col.names = T, sep = ',')
}

# p value
#count <- length(which(result_table[,'ACC_CV']>=0.86))
count <- length(which(result_table[,'BA_CV']>=0.71))
pvalue = (count + 1.0) / (100 + 1)

