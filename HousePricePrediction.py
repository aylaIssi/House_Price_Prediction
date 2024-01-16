#############################################
# Adım 1: Kütüphaneler ve Ayarlamalar
#############################################

import pandas as pd
import warnings
import numpy as np
import seaborn as sns
import joblib
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.tree import export_graphviz, export_text
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve, cross_val_score, \
    RandomizedSearchCV
from sklearn.metrics import classification_report, roc_auc_score, mean_squared_error, mean_absolute_error
from aed import isqcut_ok
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from lightgbm import LGBMRegressor

import warnings

warnings.filterwarnings('ignore')

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', 500)

#############################################
# Adım 2: Verisetlerini Okuma ve İlk Manipülasyonlar
#############################################

train_df = pd.read_csv("C:/Users/PC/PycharmProjects/DataScienceBootcamp2/HousePricePrediction/datasets/train.csv")
test_df = pd.read_csv("C:/Users/PC/PycharmProjects/DataScienceBootcamp2/HousePricePrediction/datasets/test.csv")

df_ = pd.concat([train_df, test_df], axis=0)
df = df_.copy()
df = df.reset_index().iloc[:, 1:]
df.head()

df.columns = [col.upper() for col in df.columns]
df["ID"] = df["ID"].astype(str)


#############################################
# Adım 3: Veriyi Tanıma
#############################################

def check_data(dataframe, head=5):
    print("### Info ###")
    print(dataframe.info())
    print("### DataShape ###")
    print(dataframe.shape)
    print("### MissingValues ###")
    print(dataframe.isnull().sum())
    print("### Describe ###")
    print(dataframe.describe([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T)


check_data(df)

df.head(10)

#############################################
# Adım 4: Date Değişkenlerini Gruplama
#############################################

date_cols = ["YEARBUILT", "YEARREMODADD"]

for col in date_cols:
    if isqcut_ok(df[col], 5):
        df[col] = pd.qcut(df[col], 5, labels=[1, 2, 3, 4, 5])
    else:
        print("{} değişkenine qcut uygulanamıyor".format(col))

df["GARAGEYRBLT"] = pd.qcut(df["GARAGEYRBLT"], 5, labels=[1, 2, 3, 4, 5])

date_cols.append("GARAGEYRBLT")

for col in date_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
    df[col] = df[col].astype(int)

#############################################
# Adım 5: Ordinal Değişkenleri Numeric Olarak Değiştirme
#############################################

df = df.replace("Ex", 5)
df = df.replace("Gd", 4)
df = df.replace("TA", 3)
df = df.replace("Fa", 2)
df = df.replace("Po", 1)

# poolqc, GARAGECOND, GARAGEQUAL, FIREPLACEQU, KITCHENQUAL, HEATINGQC,
# BSMTCOND, EXTERCOND, EXTERQUAL,  numaralandır
cat_but_ord = ["POOLQC", "GARAGECOND", "GARAGEQUAL", "FIREPLACEQU",
               "KITCHENQUAL", "HEATINGQC", "BSMTCOND", "EXTERCOND",
               "EXTERQUAL", "POOLAREA", "YEARBUILT", "YEARREMODADD",
               "GARAGEYRBLT"]

for col in cat_but_ord:
    df[col] = df[col].fillna(0)
for col in cat_but_ord:
    print(df[col].value_counts())


#############################################
# Adım 6: Değişken Tiplerini Yakalama
#############################################

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]  # 2.tanim

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car, num_but_cat


# MSSubClass 16 nunique ve kategorik Neighborhood 25 nunique ve kategorik
cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)

for i in num_but_cat:
    print(df[i].value_counts())
    print("********************")

for i in num_cols:
    if df[i].nunique() < 25:
        print(df[i].value_counts())
        print(df[i].nunique())
        print("********************")


def grab_col_names(dataframe, cat_th=17, car_th=26):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]  # 2.tanim

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car, num_but_cat


cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)

cat_cols = [col for col in cat_cols if col not in cat_but_ord]
num_cols = num_cols + cat_but_ord


#############################################
# Adım 7: Kategorik / Numerik Değişken Analizi
#############################################

def observe_variable_distribution(dataframe, cat_cols, num_cols):
    for column in num_cols:
        value_counts = dataframe[column].value_counts(normalize=True) * 100
        print(f'Numerik Değişken: {column}')
        print(value_counts)
        print('-' * 30)

    for column in cat_cols:
        value_counts = dataframe[column].value_counts(normalize=True) * 100
        print(f'Kategorik Değişken: {column}')
        print(value_counts)
        print('-' * 30)


def compare_categorical_and_numeric(dataframe, cat_cols, num_cols):
    num_percentage = len(num_cols) / (len(cat_cols) + len(num_cols)) * 100
    cat_percentage = 100 - num_percentage

    print(f"Toplam veri setinde numerik değişkenlerin yüzdesi: {num_percentage:.2f}%")
    print(f"Toplam veri setinde kategorik değişkenlerin yüzdesi: {cat_percentage:.2f}%")
    print('-' * 50)


compare_categorical_and_numeric(df, cat_cols, num_cols)

observe_variable_distribution(df, cat_cols, num_cols)


#############################################
# Adım 8: Numeric Değişken Analizi
#############################################

def num_summary(df, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(df[numerical_col].describe(quantiles).T)

    if plot:
        df[numerical_col].hist(bins=10)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


for col in num_cols:
    num_summary(df, col, plot=False)


#############################################
# Adım 9: Hedef Değişken Analizi
#############################################

def group_sum(df, **kwargs):
    """

    Parameters
    ----------
    df : DataFrame
        Verilerin alinacagi dataframe
    kwargs:
        target_cols: list
            Istenilen sonucta gorunmesi istenen sutun
            x gurubuna gore taget degiskeni
        group_cols: list
            Gruplanmasi istenilen sutunlar
        agg_func:
            yapilmasi istenilen agg fonksiyonlari
    Returns
    -------
        (df.groupby(by=group_cols)[target_cols].agg(kwargs['agg_func']))
        kullanilacak dataframeden  group_cols kırılımında target_cols agg_funclari nedir?
        sorunsuna cevap verir
    """
    target_cols = kwargs.get('target_cols', [])
    group_cols = kwargs.get('group_cols', [])
    agg_func = kwargs.get('agg_func', [])
    return df.groupby(by=group_cols)[target_cols].agg(kwargs['agg_func'])


for col in cat_cols:
    print(group_sum(df, target_cols="SALEPRICE", agg_func="mean", group_cols=col))


#############################################
# Adım 10: Aykırı Değişken Kontrol ve Yakalama
#############################################

def outlier_th(dataframe, col_name, q1=0.01, q3=0.99):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


low_limit, up_limit = outlier_th(df, num_cols)


# Check Outlier Fonksiyonu
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_th(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


for col in num_cols:
    print(col, check_outlier(df, col))


# Outlier kisimlarini, Threshold degerleri ile degistirmek istersek:
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_th(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


# Eşik değerleri ile değiştirmeden önce kontrol edelim:
for col in num_cols:
    print(col, check_outlier(df, col))

# Eşik değerlerini değiştirelim
for col in num_cols:
    replace_with_thresholds(df, col)

# Tekrar kontrol edelim
for col in num_cols:
    print(col, check_outlier(df, col))


#############################################
# Adım 11: Eksik Gözlen Analizi
#############################################
def missing_values(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])  # yeni df birlestiriyoruz
    print(missing_df, end="\n")

    if na_name:
        return na_columns, n_miss, ratio


na_cols, n_miss, ratio = missing_values(df, True)


def missing_vs_target(dataframe, target, na_columns):
    temp_df = df.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


missing_vs_target(df, "SALEPRICE", na_cols)

#############################################
# Adım 12: Eksik Gözlem İşlemleri
#############################################

for i, j in zip(n_miss.index, ratio):
    if j > 90:
        df = df.drop(i, axis=1)
        if i in cat_cols:
            cat_cols.remove(i)
        else:
            num_cols.remove(i)

for i in cat_cols:
    print(df[i].value_counts())
    print("****************************")

for i in cat_cols:
    df[i] = df[i].fillna(df[i].mode()[0])

na_cols, n_miss, ratio = missing_values(df, True)

for i, j in zip(n_miss.index, ratio):
    if j < 10:
        df[i] = df[i].fillna(df[i].mean())

#############################################
# Adım 13: Numeric Eksik Gözlemler İçin KNN Imputer
#############################################

na_cols, n_miss, ratio = missing_values(df, True)

na_cols.remove("SALEPRICE")

dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)

scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)

imputer = KNNImputer(n_neighbors=5)
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)

dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)

for i in na_cols:
    print(df.loc[df[i].isnull(), [i]][:5])
    print("#############################")
    print(dff.loc[df[i].isnull(), [i]][:5])

df["LOTFRONTAGE_IMPUTED"] = dff[["LOTFRONTAGE"]]
# df[["LOTFRONTAGE", "LOTFRONTAGE_IMPUTED"]][1000:1020]
df = df.drop("LOTFRONTAGE_IMPUTED", axis=1)
df["LOTFRONTAGE"] = dff[["LOTFRONTAGE"]]


#############################################
# Adım 14: ENCODING
#############################################

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df


def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


rare_analyser(df, "SALEPRICE", cat_cols)
# df["MSZONING"].value_counts()

new_df = rare_encoder(df, 0.015)

rare_analyser(new_df, "SALEPRICE", cat_cols)
# new_df["MSZONING"].value_counts()

df = new_df.copy()

le = LabelEncoder()

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


for col in binary_cols:
    df = label_encoder(df, col)

df.head()


# One Hot Encoder
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


df = one_hot_encoder(df, cat_cols)
df.head()

num_cols.remove("SALEPRICE")
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df.head()

#############################################
# Adım 15: MODELLEME
#############################################

df.info()

train = df.loc[df["SALEPRICE"].notnull()]
test = df.loc[df["SALEPRICE"].isnull()]

train = train.drop("ID", axis=1)

y = train["SALEPRICE"]
X = train.drop(["SALEPRICE"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

#############################################
# Adım 16: LİNEER REGRESYON
#############################################



reg_model = LinearRegression().fit(X, y)

y_pred = reg_model.predict(X_train)
# Train RMSE
np.sqrt(mean_squared_error(y_train, y_pred))
# TRAIN RKARE
reg_model.score(X_train, y_train)
# Test RMSE
y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
# Test RKARE
reg_model.score(X_test, y_test)

big = 0
small = 0
for i, j in zip(y_test, y_pred):
    print(i, "-----", j, "--------", abs(i - j) / i)
    if abs(i - j) / i > 0.1:
        big += 1
    else:
        small += 1
print("small:{},big:{}".format(small / (small + big), big / (small + big)))

# cross val
np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X,
                                 y,
                                 cv=5,
                                 scoring="neg_mean_squared_error")))

#############################################
# Adım 16: ADABOOST REGRESYON
#############################################

reg_model = AdaBoostRegressor().fit(X, y)
y_pred = reg_model.predict(X_train)

# Train RMSE
np.sqrt(mean_squared_error(y_train, y_pred))
# TRAIN RKARE
reg_model.score(X_train, y_train)
# Test RMSE
y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
# Test RKARE
reg_model.score(X_test, y_test)
big = 0
small = 0
for i, j in zip(y_test, y_pred):
    print(i, "-----", j, "--------", abs(i - j) / i)
    if abs(i - j) / i > 0.1:
        big += 1
    else:
        small += 1
print("small:{},big:{}".format(small / (small + big), big / (small + big)))

np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X,
                                 y,
                                 cv=5,
                                 scoring="neg_mean_squared_error")))


#############################################
# Adım 16: RANDOM FORESTS REGRESYON
#############################################

model = RandomForestRegressor().fit(X, y)
y_pred = reg_model.predict(X_train)

# Train RMSE
np.sqrt(mean_squared_error(y_train, y_pred))
# TRAIN RKARE
model.score(X_train, y_train)
# Test RMSE
y_pred = model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
# Test RKARE
model.score(X_test, y_test)
big = 0
small = 0
for i, j in zip(y_test, y_pred):
    print(i, "-----", j, "--------", abs(i - j) / i)
    if abs(i - j) / i > 0.1:
        big += 1
    else:
        small += 1
print("small:{},big:{}".format(small / (small + big), big / (small + big)))

#############################################
# Adım 16: GRADIENT BOOSTING REGRESYON
#############################################

model = GradientBoostingRegressor().fit(X, y)
y_pred = reg_model.predict(X_train)

# Train RMSE
np.sqrt(mean_squared_error(y_train, y_pred))
# TRAIN RKARE
model.score(X_train, y_train)
# Test RMSE
y_pred = model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
# Test RKARE
model.score(X_test, y_test)
big = 0
small = 0
for i, j in zip(y_test, y_pred):
    print(i, "-----", j, "--------", abs(i - j) / i)
    if abs(i - j) / i > 0.1:
        big += 1
    else:
        small += 1
print("small:{},big:{}".format(small / (small + big), big / (small + big)))

#############################################
# Adım 16: DECISION TREE REGRESYON
#############################################

model = DecisionTreeRegressor().fit(X, y)
y_pred = model.predict(X_train)

# Train RMSE
np.sqrt(mean_squared_error(y_train, y_pred))
# TRAIN RKARE
model.score(X_train, y_train)
# Test RMSE
y_pred = model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
# 1.41

# Test RKARE
model.score(X_test, y_test)
for i, j in zip(y_test, y_pred):
    print(i, "-----", j, "--------", abs(i - j) / i)

np.mean(np.sqrt(-cross_val_score(model,
                                 X,
                                 y,
                                 cv=5,
                                 scoring="neg_mean_squared_error")))

#############################################
# Adım 16: LIGHTGBM
#############################################

model = LGBMRegressor().fit(X, y)
y_pred = model.predict(X_train)

# Train RMSE
np.sqrt(mean_squared_error(y_train, y_pred))
# TRAIN RKARE
model.score(X_train, y_train)
# Test RMSE
y_pred = model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
# TRAIN RKARE
model.score(X_test, y_test)

big = 0
small = 0
for i, j in zip(y_test, y_pred):
    print(i, "-----", j, "--------", abs(i - j) / i)
    if abs(i - j) / i > 0.1:
        big += 1
    else:
        small += 1
print("small:{},big:{}".format(small / (small + big), big / (small + big)))

#############################################
# Adım 16: LIGHTGBM HYPERPARAMETER
#############################################

model = LGBMRegressor().fit(X, y)


model.get_params()
lgbm_params = {"learning_rate": [0.01, 0.015, 0.02, 0.03, 0.1],
               "n_estimators": [300, 350, 400, 450],
               "colsample_bytree": [0.7, 0.8, 0.6, 1]}

lgbm_best_grid = GridSearchCV(model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

lgbm_final = model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)



#TRAIN
y_pred = lgbm_final.predict(X_train)
# Train RMSE
np.sqrt(mean_squared_error(y_train, y_pred))
# TRAIN RKARE
lgbm_final.score(X_train, y_train)


# Test
y_pred = lgbm_final.predict(X_test)
# Test RMSE
np.sqrt(mean_squared_error(y_test, y_pred))
# TRAIN RKARE
lgbm_final.score(X_test, y_test)


big= 0
small = 0
for i, j in zip(y_test, y_pred):
    print(i, "-----", j, "--------", abs(i - j) / i)
    if abs(i - j) / i > 0.1:
        big += 1
    else:
        small += 1
print("small:{},big:{}".format(small / (small + big), big / (small + big)))
