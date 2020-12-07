##########  回归建模  #######
rm(list = ls())


#########   1.预处理  -------------------
descriptive <- read.csv("/Users/mac/Desktop/快乐研一/数据挖掘/留学推荐code&data/留学数据code&data/data/descriptive.csv", header = T, stringsAsFactors = F)   

## 调整因子变量、提取变量
descriptive$major_apply_new <- relevel(as.factor(descriptive$major_apply_new), ref = "Others")
descriptive$type <- relevel(as.factor(descriptive$type), ref = "混合")
colnames(descriptive)
features <- c("gpa_dis", "toefl_dis", "gre_total_dis", "type", "major_apply_new",
              "season", "cross",  "intern", "research", "rl","sci", "exchange","first",
              "Before_CollegeRank", "Districts", "CollegeRankTop50")
data <- descriptive[, features]  # 提取变量
data$y <- ifelse(descriptive$offertype == "Admitted", 1, 0)       
summary(data)  # 得到整理之后的数据

### 平衡数据试一试：（欠采样）
set.seed(123)
data_1 = data[data$y == 1,]
data_0 = data[data$y == 0,]
data_11 = data_1[sample(x = 1:dim(data_1)[1], size = dim(data_0)[1],replace = TRUE),]
data = rbind(data_11, data_0)
dim(data)




#########   2.logistic 回归  -------------------
lr_all <- glm(y ~ . , data, family = binomial())
lr_step <- step(lr_all)
summary(lr_step)


## 交叉验证
n <- dim(data)[1]
k <- 5
random_index <- sample(x = 1: k, size = n, replace = T)
accuracy_lr <- sapply(1: k ,function(i){
  # 分割数据集
  test_x <- data[random_index==i, ]
  train_x <- data[random_index!=i, ]
  
  # glm 和预测
  #lr_train <- glm(y ~ gpa_dis+type+major_apply_new+season+intern+ sci+exchange+Before_CollegeRank+CollegeRankTop50, train_x, family = binomial())
  lr_train <- glm(y ~ ., train_x, family = binomial())
  
  y_hat <- predict(lr_train, test_x)
  y_pred <- ifelse(y_hat > 0.5, 1, 0)
  table_tmp <- table(test_x$y, y_pred)
  (right_ratio <- sum(diag(table_tmp)) / sum(table_tmp))
  }
)
accuracy_lr


#########   3.决策树  -------------------
library(rpart)         # 决策树
library(rpart.plot)    # 决策树结构可视化

# 决策树的可视化解释（用中文更清晰）
fit.tree <- rpart(as.factor(y) ~ . , data = data,
                  control = rpart.control(cp = 0.002, minbucket = 80))
fit.tree <- rpart(as.factor(y) ~ . , data = data)

rpart.plot(fit.tree, main = "CART", type = 5, extra = 2, tweak = 1.2)  # 可视化

# 交叉验证
accuracy_dt <- sapply(1: k ,function(i){
  # 分割数据集
  test_x <- data[random_index==i, ]
  train_x <- data[random_index!=i, ]
  
  # 决策树和预测
  dt_train <- rpart(as.factor(y) ~ . , data = train_x,
                    control = rpart.control(cp = 0.005, minbucket = 100))
  y_pred <- apply(predict(dt_train, test_x), MARGIN = 1, FUN = function(x){
    return(as.numeric(names(x[order(x)][2])))
  })
  table_tmp <- table(test_x$y, y_pred)
  right_ratio <- sum(diag(table_tmp)) / sum(table_tmp)
  right_ratio
  }
)
accuracy_dt




#########   4.random forest 和预测  -------------------
library(randomForest)
fit.rf <- randomForest(as.factor(y) ~ . , data = data, ntree = 20)
fit.rf$importance
accuracy_rf <- sapply(1: k ,function(i){
  test_x <- data[random_index==i, ]
  train_x <- data[random_index!=i, ]
  
  # rf和预测
  dt_train <- randomForest(as.factor(y) ~ . , data = train_x, ntree = 20)
  y_pred <- predict(dt_train, test_x)
  table_tmp <- table(test_x$y, y_pred)
  right_ratio <- sum(diag(table_tmp)) / sum(table_tmp)
  right_ratio
  }
)
accuracy_rf


#########   5.xgboost 和预测-------------------
library(xgboost)       # xgboost
library(Matrix)        # 数据格式转换
library(Ckmeans.1d.dp)
# 由于建立xgboost模型需要特殊格式数据，对训练集和预测集进行格式转换
data_xgb <-  sparse.model.matrix(y ~ . , data = data) 
d_data <- xgb.DMatrix(data_xgb, label = data$y)    # label不能为空
xgb <- xgboost(data = d_data, max_depth = 6, gamma = 0.2, subsample = 0.8, eta = 0.1,booster = "gbtree",
               objective='binary:logistic', nround = 50)
# xgb_param <- xgb.cv(params = list( booster = "gbtree",     # 基于树模型 （下同）
#                                    eval_metric = "error",    # 度量模型误差（下同）
#                                    objective = "binary:logistic",   # 定义目标函数（下同）
#                                    gamma = 0.1,    # 定义树节点分裂所需的最小损失函数下降值（下同）
#                                    max_depth = 5,  # 定义树最大深度、学习率（下同）
#                                    eta = 0.1,      # 定义每棵树随机选取样本集和特征子集的比例（下同）
#                                    subsample = 0.8,
#                                    colsample_bytree = 0.8), 
#                     d_data, prediction = T, nfold = 5, nrounds = 100) # 十折交叉验证（下同）
# 训练集上的error
print(xgb)
table(round(predict(xgb,d_data)), data$y)

# 变量重要性
importance <- xgb.importance(data_xgb@Dimnames[[2]], model = xgb) 
head(importance)
xgb.ggplot.importance(importance[1:5,])

# cv 验证
accuracy_xgb <- sapply(1: k ,function(i){
  # 分割数据集
  test_x <- data[random_index==i, ]
  train_x <- data[random_index!=i, ]
  
  # 处理matrix
  data_xgb_train <-  sparse.model.matrix(y ~ . , data = train_x) 
  data_xgb_test <-  sparse.model.matrix(y ~ . , data = test_x) 
  d_train <- xgb.DMatrix(data_xgb_train, label = train_x$y)   
  d_test <- xgb.DMatrix(data_xgb_test, label = test_x$y)    
  
  # xgb
  xgb <- xgboost(data = d_train, max_depth = 6, gamma = 0.2, subsample = 0.8, eta = 0.1, booster = "gbtree",
                 objective='binary:logistic', nround = 50)
  # prediction
  table_tmp <- table(test_x$y, round(predict(xgb, d_test)))
  right_ratio <- sum(diag(table_tmp)) / sum(table_tmp)
  right_ratio
}
)
accuracy_xgb

# library(pROC)      # 画ROC曲线
# library(ROCR)      # 算AUC值
# pre1 <- predict(xgb, d_test)
# modelroc1 <- roc(test_x$y, pre1)
# plot(modelroc1, print.auc=TRUE, auc.polygon=TRUE, 
#      auc.polygon.col="#f6f6f6",print.thres=TRUE,main = "LDA模型预测ROC曲线",
#      xlab = "特异度",ylab = "灵敏度")
# table_tmp <- table(test_x$y, round(predict(xgb, d_test)))
# ## 查准率和查全率
# chazhun = table_tmp[2,2] / sum(table_tmp[,2])   # 我们认为可以被录取的，真的被录取了
# chaquan = table_tmp[2,2] / sum(table_tmp[2,])
# F1_score = 2 * chazhun* chaquan/ (chazhun+ chaquan)


#########   6.knn-------------------
library(caret)     # KNN
accuracy_knn <- sapply(1: k ,function(i){
  # 分割数据集
  random_index <- sample(x = 1:k, size = n, replace = T)
  test_x <- data[random_index==i, ]
  train_x <- data[random_index!=i, ]
  
  # knn和预测
  knn_train <- knn3(as.factor(y) ~ . , data = train_x, k = 100)
  y_hat <- predict(knn_train, test_x)[,2]
  y_pred <- ifelse(y_hat > 0.5, 1, 0)
  table_tmp <- table(test_x$y, y_pred)
  right_ratio <- sum(diag(table_tmp)) / sum(table_tmp)
  right_ratio
  }
)
accuracy_knn


#########   7.svm-------------------
library(e1071)         # SVM
fit.svm <- svm(as.factor(y) ~ . , data = data, gamma = 0.1, kernel = "radial")
summary(fit.svm)

accuracy_svm <- sapply(1: k ,function(i){
  # 分割数据集
  test_x <- data[random_index==i, ]
  train_x <- data[random_index!=i, ]
  
  # svm 和预测
  svm_train <- svm(as.factor(y) ~ . , data = train_x, gamma = 0.1, kernel = "radial")
  y_pred <- predict(svm_train, test_x)
  table_tmp <- table(test_x$y, y_pred)
  right_ratio <- sum(diag(table_tmp)) / sum(table_tmp)
  right_ratio
}
)
accuracy_svm


######### 8. kmeans -------------------
library(cluster)
fit.kmeans <- pam(data[, -18], 2)
table(fit.kmeans$clustering, data$y)


######### 9.mlp -------------------
library(tensorflow)
library(keras)
# 数据
test <- data[random_index==1, ]
train <- data[random_index!=1, ]
test_y <- to_categorical(test$y,2)
train_y <- to_categorical(train$y,2)
train_x <- as.matrix(model.matrix(y~., train))[,-1]
test_x <- as.matrix(model.matrix(y~., test))[,-1]

# 模型
model <- keras_model_sequential()
model %>%
     layer_dense(units = dim(train_x)[2], input_shape = dim(train_x)[2]) %>%   
     layer_dropout(rate = 0.4)%>%
     layer_activation(activation = 'relu') %>%
     layer_dense(units = 4) %>%   
     layer_dropout(rate = 0.4)%>%
     layer_activation(activation = 'relu') %>%
     layer_dense(units = 2) %>%
     layer_activation(activation = 'softmax')
model  %>% 
  compile(loss = "categorical_crossentropy",
       optimizer = optimizer_adam(),
       metrics = "accuracy")
# 训练
model %>% 
     fit(train_x, train_y, epochs = 30, batch_size = 64)

# 输出损失函数最终值和最终预测精度
model %>% 
  evaluate(test_x, test_y)
# 模型预测值
mlp_pred <- model %>% 
  predict_classes(test_x)
table(mlp_pred, test$y)



## cv 验证
accuracy_mlp <- sapply(1: k ,function(i){
  # 分割数据集
  test <- data[random_index==i, ]
  train <- data[random_index!=i, ]
  test_y <- to_categorical(test$y,2)
  train_y <- to_categorical(train$y,2)
  train_x <- as.matrix(model.matrix(y~., train))[,-1]
  test_x <- as.matrix(model.matrix(y~., test))[,-1]
  
  # fit
  model <- keras_model_sequential()
  model %>%
    layer_dense(units = dim(train_x)[2], input_shape = dim(train_x)[2]) %>%   
    layer_dropout(rate = 0.4)%>%
    layer_activation(activation = 'relu') %>%
    layer_dense(units = 10) %>%   
    layer_dropout(rate = 0.4)%>%
    layer_activation(activation = 'relu') %>%
    layer_dense(units = 2) %>%
    layer_activation(activation = 'softmax')
  model  %>% 
    compile(loss = "categorical_crossentropy",
            optimizer = optimizer_adam(),
            metrics = "accuracy")
  # 训练
  model %>% 
    fit(train_x, train_y, epochs = 50, batch_size = 128)
  
  # 模型预测值
  mlp_pred <- model %>% 
    predict_classes(test_x)
  table_tmp = table(mlp_pred, test$y)
  right_ratio <- sum(diag(table_tmp)) / sum(table_tmp)
  return(right_ratio)
}
)
accuracy_mlp







