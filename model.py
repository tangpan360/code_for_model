"""
#read_me
回归/分类任务所需要的机器学习模型整合
目前计划：
    给定输入(DF文件)、选择任务类型、选择模型类型、自动对数据进行划分、训练、测试、评价

输入参数：
(model_name="LogisticRegression", task_type="Regression", reduction="mean", train_vail_split_type="T",
                 seed = 20, test_method="normal", visual="True", learning_rate=0.001, max_iter=100, train_vaild_split_rate=0.8, mulit_class=True):
model_name: 模型类型 :"LogisticRegression"
模型类型：
线性回归模型 Regression                          # ##分类模型 Classficiation
              'Ridge'                                'GaussianNB'
              'LinearRegression'                     'LogisticRegression'
              'RandomForestRegressor'                'RandomForestClassifier'
              'GradientBoostingRegressor'            'GradientBoostingClassifier'
              'AdaBoostRegressor'                    'AdaBoostClassifier'
                                                     'SVC'
              'KNeighborsRegressor'                  'KNeighborsClassifier'
              'CatBoostRegressor'                    'CatBoostClassifier'
              'LGBMRegressor'                        'LGBMClassifier'
              'XGBRegressor'                         'XGBClassifier'
task_type            : 任务类型           :  Regression/Classficiation
reduction            : 缺失值填充方式      :  默认均值填充(mean) 可选择:向上(ffill)、向下(bfill)、均值(mean)、中位数(mid)
train_vail_split_type: 数据集划分方式      :  Y(分类任务,按y抽样)/T(回归任务 随机分配)
seed                 : 随机种子数         :  20(默认) 可设自己的幸运数字
test_method          : 测试方法           :  目前未优化  可扩展梯度搜索和几折交叉
visual               : 可视化            :  True/False
learning_rate        : 学习率            : 建议0.3-0.00001 部分模型使用不到
max_iter             : 最大迭代次数        : 100-10000  越大运行越慢  部分模型使用不到
train_vaild_split_rate: 训练集验证集划分    : 0.7 0.8 0.9 都可以
mulit_class          : 是否是多分类问题     : 多分类问题True/二分类 False

任务类型：
    回归任务：Regression
    分类任务：Classficiation :多分类

数据检查：
    如果数据有缺失值 可以选择填充方式reduction (待扩展：当数据有字符类型，自动独热编码): 默认均值mean 可选：向上ffill、bfill向下、均值mean、中位数mid

数据集划分方式：
    回归任务："T"
    分类任务：按y等比抽样或随机抽样 "Y"

随机种子：
    默认为20

测试方法：
    正常
    五折(待扩展)

评价指标：
    loss_cerition: 固定,优化模型时自己调代码换

    评价：
        回归任务：MSE、MAE、RMSE CORR
        分类任务：Accuracy、Precision、F1、Recall

可视化展示："True"（默认）、"False"
    回归任务：预测结果与真实值曲线 guan式图
    分类任务：ROC+AUC图 

"""
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import  roc_curve, auc
from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import pandas as pd
# from sklearn.datasets import load_boston
from bls_addenhencenodes import broadNet
import warnings
warnings.filterwarnings("ignore")

class Machine_learning(object):
    def __init__(self, model_name="LogisticRegression", task_type="Regression", reduction="mean", train_vail_split_type="T", 
                 seed = 20, test_method="normal", visual="True", learning_rate=0.001, max_iter=100, train_vaild_split_rate=0.8, mulit_class=True):
        
        self.multi_class = mulit_class
        self.seed = seed
        self.model = self._build_model(task_type, model_name, self.seed, learning_rate, max_iter)
        self.reduction = reduction
        self.train_vail_split_type = train_vail_split_type
        self.test_method = test_method
        self.visual = visual
        self.task_type = task_type
        self.train_vail_split_rate = train_vaild_split_rate
        
    def choose_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    """
                metric_method
    """
    def metric_method(self, true, pred):
        if self.task_type=="Regression":
            true = np.array(true)
            pred =np.array(pred)
            u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
            d = np.sqrt(((true - true.mean(0)) ** 2).sum(0) * ((pred - pred.mean(0)) ** 2).sum(0))
            M1 = np.mean(((pred - true) ** 2))
            M2 = np.mean(np.abs(pred - true))
            M3 = np.mean(np.sqrt(np.mean(((pred - true) ** 2))))
            M4 = (u / d).mean()
        else:
            M1 = accuracy_score(true, pred)
            M2 = precision_score(true, pred, average="macro")
            M3 = recall_score(true, pred, average="macro")
            M4 = f1_score(true, pred, average="macro")

        return M1, M2, M3, M4

    """
                Train
    """
    def train_vail(self, X, y, Test_x):
        try:
            if self.reduction == "mid":
                X = X.fillna(X.median())
                Test_x = Test_x.fillna(Test_x.median())
            elif self.reduction == "mean":
                X =X.fillna(X.mean())
                Test_x = Test_x.fillna(Test_x.median())
            elif self.reduction != None:
                X= X.fillna(method=self.reduction)
                Test_x = Test_x.fillna(Test_x.median())
            else:
                pass
        except Exception as e:
            ValueError("Reduction fill error")

        if self.train_vail_split_type=="Y":
            train_X, valid_X, train_y, valid_y = train_test_split(X ,
                                                    y,train_size=self.train_vail_split_rate,stratify=y,random_state=self.seed)
        else:
            train_X, valid_X, train_y, valid_y = train_test_split(X ,
                                                    y,train_size=self.train_vail_split_rate, random_state=self.seed)
        
        self.model.fit(train_X, train_y)
        pred = self.model.predict(valid_X)
        M1, M2, M3, M4 = self.metric_method(valid_y,pred)
        Test_y = self.model.predict(Test_x)
        try:
            Test_y.to_csv("./result.csv")
        except:
            Test_y = pd.Series(Test_y.flatten())
            Test_y.to_csv("./result.csv")
        if self.task_type == "Classficiation":
            print(f"Accuracy:{M1}, Precision:{M2}, Recall:{M3}, F1:{M4}")
        else:
            print(f"MSE:{M1}, MAE:{M2}, RMSE:{M3}, CORR:{M4}")

        if self.visual:
            if self.task_type == "Classficiation":
                pred_prob = self.model.predict_proba(valid_X)
                n_classes = pred_prob.shape[-1]
                self.visual_result(valid_y, pred_prob, n_classes=n_classes)
            else:
                self.visual_result(valid_y, pred)

    """
                   Results visualization
    """
    def visual_result(self, true, preds=None,n_classes=2, path='./test.pdf'):
        if self.visual ==True:
            if self.task_type=="Regression":


                true = np.array(true)
                pred =np.array(preds)
                plt.figure()
                # ax = plt.plot(x_scale, model, marker=maker, linestyle='--', clip_on = False)
                # handles_list.append(ax)
                plt.plot(true, label='GroundTruth', linewidth=2,  linestyle='--', marker=">")
                if preds is not None:
                    plt.plot(preds, label='Prediction', linewidth=2,  linestyle='--',marker="o")
                plt.grid(axis='y', linestyle='--')
                plt.grid(axis='x', linestyle='--')
                plt.legend()

                plt.show()

            else:

                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                true = np.array(pd.get_dummies(true))
                for i in range(n_classes):
                    fpr[i], tpr[i], _ = roc_curve(true[:, i], preds[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
                mean_tpr = np.zeros_like(all_fpr)

                for i in range(n_classes):
                    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
                mean_tpr /= n_classes
                fpr["macro"] = all_fpr
                tpr["macro"] = mean_tpr
                roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
                lw=2
                plt.figure()

                plt.plot(fpr["macro"], tpr["macro"],
                label='macro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["macro"]),
                color='navy', linestyle=':', linewidth=4)

                colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])

                for i, color in zip(range(n_classes), colors):
                    plt.plot(fpr[i], tpr[i], color=color, lw=lw,label='ROC curve of class {0} (area = {1:0.2f})'''.format(i, roc_auc[i]))

                plt.plot([0, 1], [0, 1], 'k--', lw=lw)
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.grid(axis='y', linestyle='--')
                plt.grid(axis='x', linestyle='--')
                plt.title('Some extension of Receiver operating characteristic to every class')
                plt.legend(loc="lower right")
                plt.show()

    """
                   Build Model
    """
    def _build_model(self, task_type="Regression", model_name="LogisticRegression", seed=20, learning_rate=0.001, max_iter=100):

        task_type_gpu= "GPU" if torch.cuda.is_available() else "CPU"
        if task_type == "Regression":

                model_dict = {
                    'Ridge': Ridge(alpha=1.0, 
                                    fit_intercept=True, normalize=False,copy_X=True,max_iter=max_iter,
                                    tol=1e-3, solver="auto",random_state=seed),
                    'LinearRegression': LinearRegression(fit_intercept=True,n_jobs=1),
                    'RandomForestRegressor' :RandomForestRegressor(n_estimators=max_iter, criterion="squared_error", random_state=seed ),
                    'GradientBoostingRegressor': GradientBoostingRegressor(loss="ls" , learning_rate=learning_rate, n_estimators=max_iter, subsample=1.0, criterion="squared_error", 
                                                                           min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,max_depth=3,
                                                                           min_impurity_decrease=0.0, init=None,random_state=seed,
                                                                           max_features=None,alpha=0.9,verbose=0,max_leaf_nodes=None,warm_start=False),##待验证
                    'AdaBoostRegressor': AdaBoostRegressor(loss="square", learning_rate=learning_rate, n_estimators=max_iter, random_state=seed,
                                                           ),
                    'KNeighborsRegressor': KNeighborsRegressor(n_neighbors=3, weights='distance', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None),
                    'CatBoostRegressor': CatBoostRegressor(verbose=0,loss_function="RMSE", iterations=max_iter, learning_rate=learning_rate,random_seed=seed, task_type=task_type_gpu),
                    'LGBMRegressor': LGBMRegressor(num_leaves=2**5-1, reg_alpha=0.25,reg_lambda=0.25, objective='regression_l2',max_depth=-1, learning_rate=learning_rate, min_child_samples=5, 
                                                            random_state=seed, n_estimators=max_iter,subsample=0.9, colsample_bytree=0.7, metric="ls", verbose=0),
                    'XGBRegressor': XGBRegressor(reg_alpha=0.25,reg_lambda=0.25, objective='reg:squarederror',max_depth=8, learning_rate=learning_rate, 
                                                            random_state=seed, n_estimators=max_iter,subsample=0.9, colsample_bytree=0.7),                                 
                    
                }
                
                model = model_dict[model_name]

        
        if task_type == "Classficiation":

            if self.multi_class == True:
                    cat_loss = "MultiClass" ##取值RMSE, Logloss, MAE, CrossEntropy, Quantile, LogLinQuantile, Multiclass, MultiClassOneVsAll, MAPE, Poisson。默认Logloss
                    lgbm_obj = "multiclass"
                    lgbm_loss = "multi_logloss"
                    xgb_class = "multiclass"

            else :
                    cat_loss = "CrossEntropy"  ##取值RMSE, Logloss, MAE, CrossEntropy, Quantile, LogLinQuantile, Multiclass, MultiClassOneVsAll, MAPE, Poisson。默认Logloss
                    lgbm_obj = "binary"
                    xgb_class = "binary:logistic"
                    lgbm_loss = "CrossEntropy"


            model_dict = {
                'GaussianNB': GaussianNB(),
                'LogisticRegression': LogisticRegression(fit_intercept=True,max_iter=max_iter, n_jobs=1, penalty="l2"),
                'RandomForestClassifier' :RandomForestClassifier(n_estimators=max_iter, criterion="entropy", random_state=seed ),
                'GradientBoostingClassifier': GradientBoostingClassifier(loss="deviance", learning_rate=learning_rate, n_estimators=max_iter,
                                                                    subsample=1.0, criterion="friedman_mse",
                                                                    min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,max_depth=3,
                                                                    min_impurity_decrease=0.0,init=None,random_state=seed,
                                                                    max_features=None,verbose=0,max_leaf_nodes=None,warm_start=False),##待验证
                'AdaBoostClassifier': AdaBoostClassifier( learning_rate=learning_rate, n_estimators=max_iter,random_state=seed,algorithm="SAMME.R"
                                                    ),
                'SVC': SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True,
                                                        probability=True, tol=0.001, cache_size=200, class_weight=None,
                                                        verbose=False, max_iter=max_iter, decision_function_shape='ovr',
                                                        random_state=seed),
                'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=3, weights='distance', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None),
                'CatBoostClassifier': CatBoostClassifier(verbose=0, depth=6, loss_function=cat_loss, custom_metric="F1",max_ctr_complexity = 4 , eval_metric="Accuracy", iterations=max_iter, learning_rate=learning_rate,random_seed=seed,task_type=task_type_gpu),
                'LGBMClassifier': LGBMClassifier(num_leaves=32, objective=lgbm_obj,max_depth=8, learning_rate=learning_rate, random_state=seed, n_estimators=max_iter,subsample=0.9, silent=1, ),
                'XGBClassifier': XGBClassifier(reg_alpha=0.25,reg_lambda=0.25, objective=xgb_class,max_depth=6, learning_rate=learning_rate,
                                                        random_state=seed, n_estimators=max_iter,subsample=0.9, colsample_bytree=0.7),
                'blsClassifier': broadNet(
                                            map_num = 10,        # 初始时多少组mapping nodes
                                            enhance_num = 10,   # 初始时多少enhancement nodes
                                            EPOCH = 20,         # 训练多少轮
                                            map_function = 'relu',
                                            enhance_function = 'relu',
                                            batchsize = 100,    # 每一组的神经元个数
                                            DESIRED_ACC = 0.96, # 期望达到的准确率
                                            STEP = 5            # 一次增加多少组enhancement nodes
                   )

            }
            model = model_dict[model_name]

            
        return model


if __name__ == "__main__":

    # # 多分类测试
    # train_data = pd.read_csv("./pre_train.csv")
    # test_x = pd.read_csv("./pre_test.csv")

    # ##二分类测试
    # train_data = pd.read_csv("./pre_train_titanic.csv")
    # test_x = pd.read_csv("./pre_test_titanic.csv")

    ##二分类测试
    train_data = pd.read_csv("./boston_house_prices_train.csv")
    test_x = pd.read_csv("./boston_house_prices_test.csv")

    y = train_data.pop(train_data.columns[-1])
    X = train_data

    exp = Machine_learning(model_name="blsClassifier", task_type="Classficiation", reduction="mean",
                           train_vail_split_type="T", seed=20, visual=True, train_vaild_split_rate=0.8, max_iter=100, mulit_class=True )

    exp.train_vail(X=X, y=y,Test_x=test_x)
"""
任务/模型类型选择如下：
线性回归模型 Regression                          # ##分类模型 Classficiation
              'Ridge'                                'GaussianNB'
              'LinearRegression'                     'LogisticRegression'
              'RandomForestRegressor'                'RandomForestClassifier'
              'GradientBoostingRegressor'            'GradientBoostingClassifier'
              'AdaBoostRegressor'                    'AdaBoostClassifier'
                                                     'SVC'
              'KNeighborsRegressor'                  'KNeighborsClassifier'
              'CatBoostRegressor'                    'CatBoostClassifier'
              'LGBMRegressor'                        'LGBMClassifier'
              'XGBRegressor'                         'XGBClassifier'
                                                     'blsClassifier'
"""
