rm(list = ls())
# 1.读入数据 ————————————————————————
descriptive <- read.csv("/Users/mac/Desktop/快乐研一/数据挖掘/留学推荐code&data/留学数据code&data/data/descriptive.csv", header = T, stringsAsFactors = T)   
descriptive$major_apply_new <- relevel(as.factor(descriptive$major_apply_new), ref = "Others")
descriptive$type <- relevel(as.factor(descriptive$type), ref = "混合")
colnames(descriptive)
features <- c("gpa_dis", "toefl_dis", "gre_total_dis", "type", "major_apply_new",
              "season", "cross",  "intern", "research", "rl","sci", "exchange","first",
              "Before_CollegeRank", "Districts", "CollegeRankTop50")

data <- descriptive[, features]  # 提取变量
data$y <- as.factor(descriptive$offertype == "Admitted")  
data$cross <- as.factor(data$cross)
data$intern <- as.factor(data$intern)
data$research <- as.factor(data$research)
data$rl <- as.factor(data$rl)
data$sci <- as.factor(data$sci)
data$exchange <- as.factor(data$exchange)
data$first <- as.factor(data$first)


### 平衡数据试一试：（欠采样）
set.seed(123)
data_1 = data[data$y == T,]
data_0 = data[data$y == F,]
data_11 = data_1[sample(x = 1:dim(data_1)[1], size = dim(data_0)[1],replace = TRUE),]
data = rbind(data_11, data_0)
dim(data)
summary(data)



### 定义绘制ROC曲线的函数
library(pROC)      # 画ROC曲线
library(ROCR)      # 算AUC值
library(showtext)

plotROC <- function(input){  # roc_curve
  yhat = input$prob.TRUE
  y = input$truth
  modelroc <- roc(y,yhat)
  plot(modelroc, print.auc=TRUE,auc.polygon=TRUE, 
       auc.polygon.col="#f6f6f6", print.thres=TRUE,main = "ROC curve")
}
Accuracy <- function(input){  # matrix & accuracy
  tmp = table(input[, c("truth", "response")])  # rf预测结果
  print(tmp)
  return(sum(diag(tmp)) / sum(tmp))
}






############### 2.MLR 包
# 定义分类任务
library(mlr)
library(dplyr)
library(kernlab)
# install.packages("kernlab")  # svm要用
classif.task = makeClassifTask(data = data, target = "y")

# 划分训练和测试
n = getTaskSize(classif.task)
set.seed(123)
random_index <- sample(x = 1:n, size = n * 0.8, replace = F)
train.set = random_index
test.set = (1:n)[-random_index]


##### 1. logistic  ------------------------
logistic_learner = makeLearner("classif.logreg", predict.type = "prob", fix.factors.prediction = TRUE)
lr_model = train(logistic_learner, classif.task, subset = train.set)
lr.pred = predict(lr_model, task = classif.task, subset = test.set)
sort(lr_model$learner.model$coefficients)       # 模型系数
plotROC(lr.pred$data);Accuracy(lr.pred$data)


####### 2. knn  ------------------
# install.packages("kknn")
getLearnerParamSet(makeLearner("classif.kknn"))  # 获取所有超参数
rdesc = makeResampleDesc("CV", iters = 5L)  # 5折交叉验证
discrete_ps = makeParamSet(  # 参数空间
  makeDiscreteParam("k", values = c(1, 3, 5, 10, 20, 30, 50, 80, 100))
  )
ctrl = makeTuneControlGrid()
res_knn = tuneParams("classif.kknn", task = classif.task,
                 resampling = rdesc, par.set = discrete_ps, control = ctrl)
res_knn # 搜索结果
ghpe_knn <- generateHyperParsEffectData(res_knn)   # 搜索过程的数据
plotHyperParsEffect(ghpe_knn, x = "k", y = "mmce.test.mean")
# 获得最优参数
knn_best = setHyperPars(makeLearner("classif.kknn", predict.type = "prob"), k = res_knn$x$k)
# 用最优参数建模
knn_model_best = train(knn_best, classif.task, subset = train.set)
knn.pred = predict(knn_model_best, task = classif.task, subset = test.set)
# 预测结果
plotROC(knn.pred$data);Accuracy(knn.pred$data)


#######  3. SVM  ------------------------
# 调参,cv交叉验证搜索：
rdesc = makeResampleDesc("CV", iters = 5L)  # 5折交叉验证
getLearnerParamSet(makeLearner("classif.ksvm"))  # 获取所有超参数
discrete_ps = makeParamSet(  # 参数空间
  makeDiscreteParam("C", values = c(0.5, 1.0, 1.5, 2.0)),
  makeDiscreteParam("sigma", values = c(0.5, 1.0, 1.5, 2.0))
  #makeDiscreteParam("kernel", values = c("vanilladot","polydot","rbfdot"))  # 线性、多项式、高斯
)
ctrl = makeTuneControlGrid()
res_svm = tuneParams("classif.ksvm", task = classif.task,
                 resampling = rdesc, par.set = discrete_ps, control = ctrl)
res_svm # 搜索结果
ghpe_svm <- generateHyperParsEffectData(res_svm)   # 搜索过程的数据
plotHyperParsEffect(ghpe_svm, x = "C",y = "sigma", z = "mmce.test.mean")
# 获得最优参数 & 用最优参数建模
svm_best = setHyperPars(makeLearner("classif.ksvm", predict.type = "prob"), C = res_svm$x$C, sigma = res_svm$x$sigma)
svm_model_best = train(svm_best, classif.task, subset = train.set)
svm.pred = predict(svm_model_best, task = classif.task, subset = test.set)
# 预测结果
plotROC(svm.pred$data);Accuracy(svm.pred$data)


###### 4. Tree ------------------------
library(rpart)         # 决策树
library(rpart.plot)    # 决策树结构可视化

# 决策树的可视化解释（用中文更清晰）
fit.tree <- rpart(as.factor(y) ~ . , data = data)
rpart.plot(fit.tree, main = "CART", type = 5, extra = 2, tweak = 1.2)  # 可视化

# 搜索参数
getLearnerParamSet(makeLearner("classif.rpart"))  # 获取所有超参数
discrete_ps = makeParamSet(  # 参数空间
  makeDiscreteParam("minsplit", values = c(50, 100, 150, 200, 250)),
  makeDiscreteParam("maxdepth", values = c(5, 6, 7, 8, 9)),
  makeDiscreteParam("cp", values = c(0, 0.001, 0.003, 0.005, 0.01))
)
ctrl = makeTuneControlGrid()
res_tree = tuneParams("classif.rpart", task = classif.task,
                 resampling = rdesc, par.set = discrete_ps, control = ctrl)
res_tree # 搜索结果
# 获得最优参数 & 用最优参数建模
tree_best = setHyperPars(makeLearner("classif.rpart", predict.type = "prob"), minsplit = res_tree$x$minsplit, maxdepth = res_tree$x$maxdepth,  cp = res_tree$x$cp)
tree_model_best = train(tree_best, classif.task, subset = train.set)
tree.pred = predict(tree_model_best, task = classif.task, subset = test.set)
# 预测结果
plotROC(tree.pred$data);Accuracy(tree.pred$data)


##### 5. 随机森林 ------------------------
getLearnerParamSet(makeLearner("classif.randomForest"))  # 获取所有超参数
discrete_ps = makeParamSet(  # 参数空间
  makeDiscreteParam("ntree", values = c(10, 15, 20, 30, 40, 50)),  # 树总数
  makeDiscreteParam("mtry", values = c(5, 8, 10, 12))   # 变量数
)
ctrl = makeTuneControlGrid()
res_rf = tuneParams("classif.randomForest", task = classif.task,
                 resampling = rdesc, par.set = discrete_ps, control = ctrl)
res_rf # 搜索结果
ghpe_rf <- generateHyperParsEffectData(res_rf)   # 搜索过程的数据
plotHyperParsEffect(ghpe_rf, x = "ntree",y = "mtry", z = "mmce.test.mean")
# 获得最优参数 & 用最优参数建模
rf_best = setHyperPars(makeLearner("classif.randomForest", predict.type = "prob"),
                       ntree = res_rf$x$ntree, mtry = res_rf$x$mtry)
rf_model_best = train(rf_best, classif.task, subset = train.set)
forest.pred = predict(rf_model_best, task = classif.task, subset = test.set)
# 预测结果
plotROC(forest.pred$data);Accuracy(forest.pred$data)


##### 6.Neural Network ------------------------
getLearnerParamSet(makeLearner("classif.nnet"))  # 获取所有超参数
discrete_ps = makeParamSet(  # 参数空间
  makeDiscreteParam("size", values = c(2,4,8,10,12,15,20))
)
ctrl = makeTuneControlGrid()
res_nnet = tuneParams("classif.nnet", task = classif.task,
                 resampling = rdesc, par.set = discrete_ps, control = ctrl)
res_nnet # 搜索结果
ghpe_net <- generateHyperParsEffectData(res_nnet)   # 搜索过程的数据
plotHyperParsEffect(ghpe_net, x = "size",y  = "mmce.test.mean")
# 获得最优参数 & 用最优参数建模
net_best = setHyperPars(makeLearner("classif.nnet", predict.type = "prob"), size = res_nnet$x$size)
nnet_model_best = train(net_best, classif.task, subset = train.set)
nnet.pred = predict(nnet_model_best, task = classif.task, subset = test.set)
# 预测
plotROC(nnet.pred$data);Accuracy(nnet.pred$data)



##### 7.LDA  & QDA  ------------------------
# lda
lda_learner = makeLearner("classif.lda", predict.type = "prob", fix.factors.prediction = TRUE)
lda_model = train(lda_learner, classif.task, subset = train.set)
lda.pred = predict(lda_model, task = classif.task, subset = test.set)
plotROC(lda.pred$data);Accuracy(lda.pred$data)

# qda
qda_learner = makeLearner("classif.qda", predict.type = "prob", fix.factors.prediction = TRUE)
qda_model = train(qda_learner, classif.task, subset = train.set)
qda.pred = predict(qda_model, task = classif.task, subset = test.set)
plotROC(qda.pred$data);Accuracy(qda.pred$data)



##### 8.XGBoost ------------------------
library(xgboost)       # xgboost
library(Matrix)        # 数据格式转换
#library(Ckmeans.1d.dp)
# 由于建立xgboost模型需要特殊格式数据，对训练集和预测集进行格式转换
data_xgb <-  sparse.model.matrix(y ~ . , data = data) 
data$y <- as.numeric(data$y) - 1 
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
table(round(predict(xgb,d_data)), data$y)

# 变量重要性
importance <- xgb.importance(data_xgb@Dimnames[[2]], model = xgb) 
head(importance)
xgb.ggplot.importance(importance[1:10,])


# cv 验证 + 调参结果
n <- dim(data)[1]
k <- 5
cv_index <- sample(x = 1: k, size = n, replace = T)
accuracy_xgb <- sapply(1: k ,function(i){
  # 分割数据集
  test_x <- data[cv_index==i, ]
  train_x <- data[cv_index!=i, ]
  
  # 处理matrix
  data_xgb_train <-  sparse.model.matrix(y ~ . , data = train_x) 
  data_xgb_test <-  sparse.model.matrix(y ~ . , data = test_x) 
  d_train <- xgb.DMatrix(data_xgb_train, label = train_x$y)   
  d_test <- xgb.DMatrix(data_xgb_test, label = test_x$y)    
  
  # xgb
  xgb <- xgboost(data = d_train, max_depth = 20, gamma = 0.2, subsample = 0.9, eta = 0.1,
                 booster = "gbtree",objective='binary:logistic', nround = 50)
  # prediction
  table_tmp <- table(test_x$y, round(predict(xgb, d_test)))
  right_ratio <- sum(diag(table_tmp)) / sum(table_tmp)
  right_ratio
}
)
accuracy_xgb


###### 9. kmeans & em 聚类算法
# kmeans
library(cluster)
fit.kmeans <- pam(data[, -18], 2)
table(fit.kmeans$clustering, data$y)
# em
library(mclust)
fit.em = Mclust(data[, -18], 2)
table(fit.em$classification, data$y)



###### 10. MLP 多层神经网络
#install.packages("tensorflow")
library(tensorflow)
library(keras)
## 数据处理
test <- data[ - random_index, ]
train <- data[random_index, ]
test_y <- to_categorical(as.factor(test$y),2)
train_y <- to_categorical(as.factor(train$y),2)
train_x <- as.matrix(model.matrix(y~., train))[,-1]
test_x <- as.matrix(model.matrix(y~., test))[,-1]
## 模型搭建
model <- keras_model_sequential()
model %>% 
  layer_dense(units = dim(train_x)[2], input_shape = dim(train_x)[2]) %>%   
  layer_dropout(rate = 0.4)%>%
  layer_activation(activation = 'relu') %>%
  layer_dense(units = 4) %>%   
  layer_dropout(rate = 0.4)%>%
  layer_activation(activation = 'relu') %>%
  layer_dense(units = 2) %>%
  layer_activation(activation = 'softmax')   ## 各层
model  %>% 
  compile(loss = "categorical_crossentropy",
          optimizer = optimizer_adam(),
          metrics = "accuracy")   ## compile
model %>%   ## 训练
  fit(train_x, train_y, epochs = 40, batch_size = 64)
model %>%   ## 输出最终预测精度
  evaluate(train_x, train_y)
## 模型预测值
mlp_pred <- model %>% 
  predict_classes(test_x)
table(mlp_pred, test$y)

## cv 验证
accuracy_mlp <- sapply(1: k ,function(i){
  # 分割数据集
  test <- data[cv_index==i, ]
  train <- data[cv_index!=i, ]
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
    layer_dense(units = 8) %>%   
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
    fit(train_x, train_y, epochs = 40, batch_size = 128)
  
  # 模型预测值
  mlp_pred <- model %>% 
    predict_classes(test_x)
  table_tmp = table(mlp_pred, test$y)
  right_ratio <- sum(diag(table_tmp)) / sum(table_tmp)
  return(right_ratio)
}
)
accuracy_mlp


