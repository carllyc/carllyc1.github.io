
import pandas as pd

data_train = pd.read_csv('..//data//homework//train.csv')
data_test = pd.read_csv('..//data//homework//test.csv')
df_train = data_train.copy()
df_test = data_test.copy()
df_train.sample(10)
df_test.sample(10)
# %%
# 去除无用特征
df_train.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
df_test.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
df_train.info()
df_test.info()
# %%
# 替换/删除空值，这里是删除
print('Is there any NaN in the dataset: {}'.format(df_train.isnull().values.any()))
df_train.dropna(inplace=True)
print('Is there any NaN in the dataset: {}'.format(df_train.isnull().values.any()))
print('Is there any NaN in the dataset: {}'.format(df_test.isnull().values.any()))
df_test.dropna(inplace=True)
print('Is there any NaN in the dataset: {}'.format(df_test.isnull().values.any()))
# %%
# 把categorical数据通过one-hot变成数值型数据
# 很简单，比如sex=[male, female]，变成两个特征,sex_male和sex_female，用0, 1表示
df_train = pd.get_dummies(df_train)
df_test = pd.get_dummies(df_test)
# %%
# 分离特征与标签
X_train = df_train.iloc[:, :-1]
y_train = df_train.iloc[:, -1]
X_test = df_test.iloc[:, :-1]
y_test = df_test.iloc[:, -1]
# %%
# train-test split
print('X_train: {}'.format(X_train))
print('y_train: {}'.format(y_train))
print('X_test: {}'.format(X_test))
print('y_test: {}'.format(y_test))
# %%
# build model
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

models = dict()

models['SVM'] = SVC(kernel='rbf')  # SVM这里我们搞个最常用的
models['KNeighbor'] = KNeighborsClassifier(n_neighbors=5)  # n_neighbors表示neighbor个数
models['RandomForest'] = RandomForestClassifier(n_estimators=100)  # n_estimators表示树的个数
# %%
# predict and evaluate
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import numpy as np


def plot_cm(model, y_true, y_pred, name=None):
   
    _, ax = plt.subplots()
    if name is not None:
        ax.set_title(name)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(ax=ax)
    plt.show()
    return None


def plot_cm_ratio(model, y_true, y_pred, name=None):
   
    _, ax = plt.subplots()
    if name is not None:
        ax.set_title(name)
    cm = confusion_matrix(y_true, y_pred)
    cm_ratio = np.zeros(cm.shape)
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            cm_ratio[i, j] = cm[i, j] / cm[i].sum()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_ratio, display_labels=model.classes_)
    disp.plot(ax=ax)
    plt.show()
    return None


def model_perf(model, y_true, y_pred, name=None):
    
    if name is not None:
        print('For model {}: \n'.format(name))
    cm = confusion_matrix(y_true, y_pred)
    for i in range(len(model.classes_)):
        # TODO: Add comments
        tp = cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        fn = cm[i, :].sum() - cm[i, i]
        tn = cm.sum() - tp - fp - fn
        tpr = tp / (tp + fn)
        fpr = fp / (tn + fp)
        acc = (tp + tn) / cm.sum()
        print('For class {}: \n TPR is {}; \n FPR is {}; \n ACC is {}. \n'
        .format(model.classes_[i], tpr, fpr, acc))
    return None


def ovo_eval(model, name=None):
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    plot_cm(model, y_test, prediction, name)
    plot_cm_ratio(model, y_test, prediction, name)
    model_perf(model, y_test, prediction, name)
    print('Overall Accuracy: {}'.format(model.score(X_test, y_test)))
# %%
# 评估各模型性能
for name, model in models.items():
    ovo_eval(model, name)