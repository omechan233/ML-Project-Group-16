import numpy as np
import pandas
import numpy
import matplotlib.pyplot as plt


# data is the dataset read by pandas
# cols is the columns you want to remove outliers
# for this project, cols is always ['Duration', 'NumSubscribers', 'ViewCount']
# the return value is cleaned pandas DataFrame
def outlier_removal(data, cols):
    data.boxplot(cols)
    plt.show()
    for i in cols:
        q75, q25 = np.percentile(data.loc[:, i], [75, 25])
        intr_qr = q75 - q25

        _max = q75 + (1.5 * intr_qr)
        _min = q25 - (1.5 * intr_qr)

        data.loc[data[i] < _min, i] = np.nan
        data.loc[data[i] > _max, i] = np.nan

    outliers = data.isnull().sum()
    row = len(data.index)
    print(outliers)
    data = data.dropna(axis=0)
    row -= len(data.index)
    print(str(row) + " outliers dropped.")
    return data
