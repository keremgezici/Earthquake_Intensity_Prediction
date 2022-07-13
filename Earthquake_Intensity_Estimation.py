
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, plot_roc_curve
from sklearn.model_selection import train_test_split, cross_validate
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
import joblib
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import warnings
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_validate,RandomizedSearchCV
from datetime import datetime
import datetime

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 100)

df= pd.read_excel("doğu.veri.xlsx")
df1= pd.read_excel("doğu.veri.xlsx")

df.head()
df.isnull().sum()
df.info()
df.shape

#Feature Enginering

df['date1'] = pd.to_datetime(df["date1"])
df['year'] = df.date1.apply(lambda x: x.year)
df['month'] = df.date1.apply(lambda x: x.month)
df['weekday'] = df.date1.apply(lambda x: x.dayofweek)
dataover5 = df[df.xm >= 5]

df.loc[(df["month"]<=2),"Mevsim"]="Kış"
df.loc[(df["month"]>2) & (df["month"]<=5 ),"Mevsim"]="İlkbahar"
df.loc[(df["month"]>5) & (df["month"]<=8 ),"Mevsim"]="Yaz"
df.loc[(df["month"]>8 )& (df["month"]<=11 ),"Mevsim"]="Sonbahar"
df.loc[( df["month"]==12 ),"Mevsim"]="Kış"

df["y"]=df["depth"]*df["lat"]
df["z"]=df["depth"]*df["long"]

df.loc[(df["city"]=="van"),"Toplam_Deprem_Sayı"]=900
df.loc[(df["city"]=="bingol"),"Toplam_Deprem_Sayı"]=346
df.loc[(df["city"]=="elazig"),"Toplam_Deprem_Sayı"]=259
df.loc[(df["city"]=="malatya"),"Toplam_Deprem_Sayı"]=137
df.loc[(df["city"]=="adiyaman"),"Toplam_Deprem_Sayı"]=113
df.loc[(df["city"]=="kahramanmaras"),"Toplam_Deprem_Sayı"]=108
df.loc[(df["city"]=="hatay"),"Toplam_Deprem_Sayı"]=101
df.loc[(df["city"]=="mus"),"Toplam_Deprem_Sayı"]=96
df.loc[(df["city"]=="osmaniye"),"Toplam_Deprem_Sayı"]=95
df.loc[(df["city"]=="kahramtuncelianmaras"),"Toplam_Deprem_Sayı"]=94
df.loc[(df["city"]=="gaziantep"),"Toplam_Deprem_Sayı"]=32
df.loc[(df["city"]=="kilis"),"Toplam_Deprem_Sayı"]=8

df.loc[(df["Toplam_Deprem_Sayı"]<=94),"NEW_CITY_CAT"]="Low_Earthquake"
df.loc[(df["Toplam_Deprem_Sayı"]>94)&(df["Toplam_Deprem_Sayı"]<=138),"NEW_CITY_CAT"]="Mıd_Earthquake"
df.loc[(df["Toplam_Deprem_Sayı"]>138)&(df["Toplam_Deprem_Sayı"]<=900),"NEW_CITY_CAT"]="Hıgh_Earthquake"

df.loc[(df["depth"]>=100),"NEW_DEPTH_CAT"]="Hıgh_Depth"
df.loc[(df["depth"]>=40)&(df["depth"]<100),"NEW_DEPTH_CAT"]="Mıd_Depth"
df.loc[(df["depth"]>=1)&(df["depth"]<40),"NEW_DEPTH_CAT"]="Low_Depth"

df.loc[(df["Toplam_Deprem_Sayı"]<=94)&(df["depth"]>=100),"Risk"]="Low_Risk3"
df.loc[(df["Toplam_Deprem_Sayı"]<=94)&(df["depth"]>=40)&(df["depth"]<100),"Risk"]="Low_Risk2"
df.loc[(df["Toplam_Deprem_Sayı"]<=94)&(df["depth"]>=1)&(df["depth"]<40),"Risk"]="Low_Risk1"
df.loc[(df["Toplam_Deprem_Sayı"]>94)&(df["Toplam_Deprem_Sayı"]<=138)&(df["depth"]>=100),"Risk"]="Mıd_Risk3"
df.loc[(df["Toplam_Deprem_Sayı"]>94)&(df["Toplam_Deprem_Sayı"]<=138)&(df["depth"]>=40)&(df["depth"]<100),"Risk"]="Mıd_Risk2"
df.loc[(df["Toplam_Deprem_Sayı"]>94)&(df["Toplam_Deprem_Sayı"]<=138)&(df["depth"]>=1)&(df["depth"]<40),"Risk"]="Mıd_Risk3"
df.loc[(df["Toplam_Deprem_Sayı"]>138)&(df["Toplam_Deprem_Sayı"]<=900)&(df["depth"]>=100),"Risk"]="Mıd_Risk3"
df.loc[(df["Toplam_Deprem_Sayı"]>138)&(df["Toplam_Deprem_Sayı"]<=900)&(df["depth"]>=40)&(df["depth"]<100),"Risk"]="Mıd_Risk2"
df.loc[(df["Toplam_Deprem_Sayı"]>138)&(df["Toplam_Deprem_Sayı"]<=900)&(df["depth"]>=1)&(df["depth"]<40),"Risk"]="Mıd_Risk3"

df.drop(["date1","date_time","country","area","direction","dist","mw","ms","Toplam_Deprem_Sayı","city","richter","md","mb"],axis=1,inplace=True)
df.head()

## Graphics

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)  # Aylara göre depremlerin yoğunluğu
dataover5.month.value_counts().sort_index().plot.bar()

plt.subplot(1, 2, 2)  # Günlere göre depremlerin yoğunluğu
dataover5.weekday.value_counts().sort_index().plot.bar()

plt.figure(figsize=(15,5))
plt.subplot(1,2,1)                                                # Tüm depremlerin yıllara göre dağılımı
plt.plot(df.year.value_counts().sort_index())
plt.subplot(1,2,2)                                                # 5 ten büyük depremlerin yıllara göre dağılımı
plt.plot(df[df.richter >= 5].year.value_counts().sort_index())

plt.scatter(df.depth, df.xm, color= "brown")
plt.legend()
plt.xlabel("Depth")
plt.ylabel("xm")
plt.show()

df.plot(kind= "scatter", x= "depth", y= "city",color= "purple", grid= True)
plt.xlabel= "Latitude"
plt.ylabel= "md"
plt.legend()

df.plot(kind= "scatter", x= "xm", y= "city",color= "purple", grid= True)
plt.xlabel= "Latitude"
plt.ylabel= "xm"
plt.legend()

#plt.scatter(d.long, d.lat, grid=True, label= "latitude - duration", color="red")
df.plot(kind= "scatter", x= "long", y= "lat", grid=True, label= "long - lat", color="red")
plt.legend()
plt.title("long - lat")

## Outliers

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
  quartile1 = dataframe[col_name].quantile(q1)
  quartile3 = dataframe[col_name].quantile(q3)
  interquantile_range = quartile3 - quartile1
  up_limit = quartile3 + 1.5 * interquantile_range
  low_limit = quartile1 - 1.5 * interquantile_range
  return low_limit, up_limit

def check_outlier(dataframe, col_name):
  low_limit, up_limit = outlier_thresholds(dataframe, col_name)
  if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
   return True
  else:
   return False

k=["depth","y","z"]
for col in k:
    print(col,check_outlier(df,col))

 def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
  low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
  dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
  dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

replace_with_thresholds(df, "depth")
replace_with_thresholds(df, "y")
replace_with_thresholds(df, "z")

## Encoding

def label_encoder(dataframe, binary_col):
  labelencoder = LabelEncoder()
  dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
  return dataframe

 labelencod=["Mevsim"]

for col in labelencod:
    df = label_encoder(df, col)

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

ohe_cols = ["NEW_CITY_CAT","NEW_DEPTH_CAT","Risk"]
ohe_cols

df=one_hot_encoder(df, ohe_cols)
df.head()
df.shape

## Scaling
num_cols=["lat","long","depth","year","month","weekday","Mevsim","y","z"]
X_scaled = StandardScaler().fit_transform(df[num_cols])
df[num_cols] = pd.DataFrame(X_scaled, columns=df[num_cols].columns)

df.head()
df.shape

## Model
df.head()
y = df['xm']
X = df.drop(["xm"], axis=1)
X.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=46)

models = [('LR', LinearRegression()),
          ("Ridge", Ridge()),
          ("Lasso", Lasso()),
          ("ElasticNet", ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          ('SVR', SVR()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor()),
        ("CatBoost", CatBoostRegressor(verbose=False))]

print("##################Base_Models#################################")
for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=7, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

# randomcv RF
rf_model = RandomForestRegressor(random_state=17)

rf_random_params = {"max_depth": np.random.randint(5, 20, 10),
                    "max_features": np.random.randint(2, 17, 11),
                    "min_samples_split": np.random.randint(2, 20, 10),
                    "n_estimators": [int(x) for x in np.linspace(start=300, stop=400, num=80)],
                    "min_samples_leaf" : np.random.randint(2, 50, 20),
                    "min_weight_fraction_leaf" : [0.01,0.1,0.2,0.3,0.02,0.5],
                    "min_impurity_decrease":[0.01,0.1,0.2,0.3,0.02,0.5,0.7,0.9],
                    "max_samples":[0.01,0.1,0.2,0.3,0.02,0.5,0.7,0.9]}


rf_random = RandomizedSearchCV(estimator=rf_model,
                               param_distributions=rf_random_params,
                               n_iter=100,  # denenecek parametre sayısı
                               cv=7,
                               verbose=True,
                               random_state=42,
                               n_jobs=-1)

rf_random.fit(X, y)

# rf best hyperparameter value
rf_random.best_params_

# rf best score
rf_random.best_score_


#randomcv LGBM
lgb_model = LGBMRegressor(random_state=17)

lgb_random_params = {"num_leaves" : np.random.randint(2, 10, 5),
                     "n_estimators": [int(x) for x in np.linspace(start=300, stop=2000, num=50)],
                     "min_child_samples": np.random.randint(5, 20, 10),
                     "reg_alpha": [0.01,0.1,0.2,0.3,0.02,0.5,0.7,0.9],
                     "reg_lambda": [0.01,0.1,0.2,0.3,0.02,0.5,0.7,0.9],
                     "learning_rate": [0.01,0.1,0.2,0.3,0.02,0.5,0.7,0.9,1,3,5,7],
                     "colsample_bytree": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                     "min_child_weight" : [0.001,0.01,0.1,0.2,0.3,0.02,0.5,0.7,0.9]}

lgb_random = RandomizedSearchCV(estimator=lgb_model,param_distributions=lgb_random_params,
                                n_iter=100,  # denenecek parametre sayısı
                                cv=7,
                                verbose=True,
                                random_state=42,
                                n_jobs=-1)

lgb_random.fit(X, y)

# lgbm best hyperparameter value
lgb_random.best_params_

# lgm best score
lgb_random.best_score_


#randomcv GBM

gbm_model=GradientBoostingRegressor(random_state=17)
gbm_random_params ={"learning_rate":[0.01,0.1,0.2,0.3,0.02],
        "subsample":[1,0.8,0.7,0.6, 0.5],
        "min_samples_split": np.random.randint(2, 20, 10),
        "min_samples_leaf": np.random.randint(2, 50, 20),
        "min_weight_fraction_leaf":[0.01,0.1,0.2,0.3,0.02,0.5],
        "max_depth":np.random.randint(5, 20, 10),
        "min_impurity_decrease":[0.01,0.1,0.2,0.3,0.02,0.5,0.7,0.9],
        "max_features":np.random.randint(2, 17, 11),
        'n_estimators': [int(x) for x in np.linspace(start=400, stop=500, num=80)]}

gbm_random = RandomizedSearchCV(estimator=gbm_model,param_distributions=gbm_random_params,
                                n_iter=100,  # denenecek parametre sayısı
                                cv=7,
                                verbose=True,
                                random_state=42,
                                n_jobs=-1,)
gbm_random.fit(X, y)

# gbm best hyperparameter value
gbm_random.best_params_

# gbm best score
gbm_random.best_score_


#XGB Random Search

xgb_model = XGBRegressor(random_state=17)


xgb_random_params = {"max_depth": np.random.randint(2, 20, 20),
                     "n_estimators": [int(x) for x in np.linspace(start=100, stop=1000, num=20)],
                     "min_child_weight": [0.3,0.02,0.5,0.7,0.9],
                     "learning_rate": [0.02,0.5,0.7,0.9],
                     "colsample_bytree": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                     "min_child_weight" : [0.02,0.5,]}


xgb_random = RandomizedSearchCV(estimator=xgb_model,param_distributions=xgb_random_params,
                                n_iter=100,  # denenecek parametre sayısı
                                cv=7,
                                verbose=True,
                                random_state=42,
                                n_jobs=-1,)


xgb_random.fit(X, y)


# xgb best hyperparameter value
xgb_random.best_params_
# xgb best score
xgb_random.best_score_

k1={'n_estimators':[340],
 'min_weight_fraction_leaf':[0.02],
 'min_samples_split':[12],
 'min_samples_leaf':[3],
 'min_impurity_decrease':[0.01],
 'max_samples':[0.1],
 'max_features':[12],
 'max_depth':[7]}

m1={'n_estimators':[668],
 'min_child_weight':[0.02],
 'max_depth':[2],
 'learning_rate':[0.02],
 'colsample_bytree':[0.4]}

l1={'reg_lambda': [0.5],
 'reg_alpha': [0.7],
 'num_leaves': [8],
 'n_estimators': [751],
 'min_child_weight': [0.5],
 'min_child_samples': [9],
 'learning_rate': [0.01],
 'colsample_bytree': [0.6],}

n1={'subsample': [0.6],
 'min_weight_fraction_leaf': [0.02],
 'min_samples_split': [17],
 'min_samples_leaf': [3],
 'min_impurity_decrease': [0.7],
 'max_features': [13],
 'max_depth': [8],
 'learning_rate': [0.01],
  'n_estimators': [437],}

#testing
regressors = [#("CART", DecisionTreeRegressor(), cart_params),
    ("RF", RandomForestRegressor(),k1),
    ('XGBoost', XGBRegressor(objective='reg:squarederror'),m1),
    ('LightGBM', LGBMRegressor(),l1),
    ("GBM",GradientBoostingRegressor(),n1)]

def hyperparameter_optimization(X, y, cv=7, scoring="neg_mean_squared_error"):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, classifier, params in regressors:
        print(f"########## {name} ##########")
        rmse = np.mean(np.sqrt(-cross_val_score(classifier, X, y, cv=cv, scoring=scoring)))
        print(f"RMSE: {round(rmse, 4)} ({name}) ")

        gs_best = GridSearchCV(classifier, params, cv=7, n_jobs=-1 ,verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)

        rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=cv, scoring=scoring)))
        print(f"RMSE (After): {round(rmse, 4)} ({name}) ")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models
best_models = hyperparameter_optimization(X, y)

######################################################
# # Stacking & Ensemble Learning
######################################################

voting_reg = VotingRegressor(estimators=[('XGBoost', best_models["XGBoost"]),
                                         ('LightGBM', best_models["LightGBM"]),
                                         ("GBM",best_models["GBM"])])

voting_reg.fit(X, y)

np.mean(np.sqrt(-cross_val_score(voting_reg, X, y, cv=7,scoring="neg_mean_squared_error")))

######################################################
# 6. Prediction for a New Observation
######################################################
df.head()
X.head()

X.columns
random_user = X.sample(10, random_state=19)

voting_reg.predict(random_user)

k=pd.DataFrame({"Real":df1["xm"].tail(20) ,"Predict":voting_reg.predict(X.tail(20))})
k

joblib.dump(voting_reg, "deprem_tahmin.pkl")  ## Saving Model

new_model = joblib.load("deprem_tahmin.pkl")
new_model.predict(random_user)










