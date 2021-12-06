import pandas
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.pipeline import make_pipeline

# --------------Read & Normalize data----------------
data = pandas.read_csv("cleaned_data_youtube.csv", names=['Duration', 'NumSubscribers', 'ViewCount'])
data = data.sample(frac=1)
scaler = StandardScaler()
duration = data.iloc[:, 0]
numSubscribers = data.iloc[:, 1]
viewCount = data.iloc[:, 2]
x = np.column_stack((duration, numSubscribers))

X_std = scaler.fit_transform(x)
# --------------Read & Normalize data----------------

# --------------Baseline-----------------
model = LinearRegression()
model.fit(X_std, viewCount)
ypred = model.predict(X_std)
print("Baseline MSE: " + str(mean_squared_error(viewCount, ypred)))
print("Baseline R2: " + str(r2_score(viewCount, ypred)))
# --------------Baseline-----------------

# --------------Ridge----------------
kf = KFold(n_splits=5)
plt.rc('font', size=18)
plt.rcParams['figure.constrained_layout.use'] = True
mean_error = []
std_error = []
for q in range(1, 10):
    Xpoly = PolynomialFeatures(q).fit_transform(X_std)
    # model = make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=1/(2*0.05)))
    model = Ridge(alpha=1/(2*0.05))
    temp = []
    plotted = False
    for train, test in kf.split(Xpoly):
        model.fit(Xpoly[train], viewCount[train])
        ypred = model.predict(Xpoly[test])
        temp.append(mean_squared_error(viewCount[test], ypred))
    mean_error.append(np.array(temp).mean())
    std_error.append(np.array(temp).std())
plt.clf()
plt.errorbar(range(1, 10), mean_error, yerr=std_error, linewidth=3)
plt.xlabel('q')
plt.ylabel('Mean square error')
plt.show()

mean_error = []
std_error = []
Ci_range = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 1, 5, 10]
for Ci in Ci_range:
    Xpoly = PolynomialFeatures(2).fit_transform(X_std)
    # model = make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=1/(2*0.05)))
    model = Ridge(alpha=1 / (2 * Ci))
    temp = []
    kf = KFold(n_splits=5)
    for train, test in kf.split(Xpoly):
        model.fit(Xpoly[train], viewCount[train])
        ypred = model.predict(Xpoly[test])
        temp.append(mean_squared_error(viewCount[test], ypred))
    mean_error.append(np.array(temp).std())
    std_error.append(np.array(temp).std())
plt.clf()
plt.errorbar(Ci_range, mean_error, yerr=std_error)
plt.xlabel('C value')
plt.ylabel('mean error')
plt.xlim(0.00005)
plt.show()

Xpoly = PolynomialFeatures(2).fit_transform(X_std)
model = Ridge(1 / (2 * 0.1))
model.fit(X_std, viewCount)
ypred = model.predict(X_std)
print("Ridge MSE: " + str(mean_squared_error(viewCount, ypred)))
print("Ridge R2: " + str(r2_score(viewCount, ypred)))
# --------------Ridge----------------

plt.clf()
# --------------Lasso----------------
plt.rc('font', size=18)
plt.rcParams['figure.constrained_layout.use'] = True
Ci_range = [0.01, 0.1, 1, 2, 5, 10, 50]
mean_error = []
std_error = []
for Ci in Ci_range:
    Xpoly = PolynomialFeatures(2).fit_transform(X_std)
    # model = make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=1/(2*0.05)))
    model = Lasso(alpha=1/(2*Ci))
    temp = []
    plotted = False
    for train, test in kf.split(Xpoly):
        model.fit(Xpoly[train], viewCount[train])
        ypred = model.predict(Xpoly[test])
        temp.append(mean_squared_error(viewCount[test], ypred))
    mean_error.append(np.array(temp).mean())
    std_error.append(np.array(temp).std())
plt.clf()
plt.errorbar(Ci_range, mean_error, yerr=std_error, linewidth=3)
plt.xlabel('Ci')
plt.ylabel('Mean square error')
plt.show()

Xpoly = PolynomialFeatures(2).fit_transform(X_std)
model = Lasso(1 / (2 * 0.1))
model.fit(X_std, viewCount)
ypred = model.predict(X_std)
print("Lasso MSE: " + str(mean_squared_error(viewCount, ypred)))
print("Lasso R2: " + str(r2_score(viewCount, ypred)))
# --------------Lasso----------------

plt.clf()

# --------------Elastic Net----------------
mean_error = []
std_error = []
Ci_range = [0.01, 0.1, 0.5, 1, 5, 10, 50, 100]
for Ci in Ci_range:
    Xpoly = PolynomialFeatures(2).fit_transform(X_std)
    # model = make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=1/(2*0.05)))
    model = ElasticNet(alpha=Ci, l1_ratio=0.5)
    temp = []
    kf = KFold(n_splits=5)
    for train, test in kf.split(Xpoly):
        model.fit(Xpoly[train], viewCount[train])
        ypred = model.predict(Xpoly[test])
        temp.append(mean_squared_error(viewCount[test], ypred))
    mean_error.append(np.array(temp).std())
    std_error.append(np.array(temp).std())
plt.clf()
plt.errorbar(Ci_range, mean_error, yerr=std_error)
plt.xlabel('alpha value')
plt.ylabel('mean error')
plt.xlim(0.00005)
plt.show()

mean_error = []
std_error = []
l1_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for l1 in l1_range:
    Xpoly = PolynomialFeatures(2).fit_transform(X_std)
    # model = make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=1/(2*0.05)))
    model = ElasticNet(alpha=1, l1_ratio=l1)
    temp = []
    kf = KFold(n_splits=5)
    for train, test in kf.split(Xpoly):
        model.fit(Xpoly[train], viewCount[train])
        ypred = model.predict(Xpoly[test])
        temp.append(mean_squared_error(viewCount[test], ypred))
    mean_error.append(np.array(temp).std())
    std_error.append(np.array(temp).std())
plt.clf()
plt.errorbar(l1_range, mean_error, yerr=std_error)
plt.xlabel('l1 ratio')
plt.ylabel('mean error')
plt.xlim(0.00005)
plt.show()

Xpoly = PolynomialFeatures(2).fit_transform(X_std)
model = ElasticNet(alpha=1, l1_ratio=0.5)
model.fit(X_std, viewCount)
ypred = model.predict(X_std)
print("Elastic MSE: " + str(mean_squared_error(viewCount, ypred)))
print("Elastic R2: " + str(r2_score(viewCount, ypred)))
# --------------Elastic Net----------------
