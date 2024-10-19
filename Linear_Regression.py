# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import r2_score
# from sklearn.metrics import mean_absolute_error
# from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import train_test_split
# from sklearn import linear_model
# import joblib
# # Đọc dữ liệu
# # data = pd.read_csv('./Cellphone.csv')
# data = pd.read_csv('./UorLaptop.csv')

# # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
# dt_Train, dt_Test = train_test_split(data, test_size=0.3, shuffle=False)

# # Xác định biến độc lập (X) và biến phụ thuộc (y)
# X_train = dt_Train.iloc[:, :6]
# y_train = dt_Train.iloc[:, 6]
# X_test = dt_Test.iloc[:, :6]
# y_test = dt_Test.iloc[:, 6]

# # Huấn luyện mô hình hồi quy tuyến tính
# clf = linear_model.LinearRegression().fit(X_train, y_train)
# print('w=', clf.coef_)
# print('w0=', clf.intercept_)

# # Dự đoán dữ liệu kiểm tra
# y_pred = clf.predict(X_test)
# y = np.array(y_test)

# # Hàm tính chỉ số NSE
# def tinh_nse(y_test, y_pred):
#     numerator = np.sum((y_test - y_pred) ** 2)
#     denominator = np.sum((y_pred - np.mean(y_pred)) ** 2)
#     nse = 1 - (numerator / denominator)
#     return nse

# # In ra các chỉ số hiệu suất
# print("R2: %.9f" % r2_score(y_test, y_pred))
# print("NSE: %.9f" % tinh_nse(y_test, y_pred))
# print("MAE: %.9f" % mean_absolute_error(y_test, y_pred))
# print("RMSE: %.9f" % np.sqrt(mean_squared_error(y_test, y_pred)))
# print("Thuc te Du doan Chenh lech")
# for i in range(0, len(y)):
#     print("%.2f" % y[i], "  ", y_pred[i], "  ", abs(y[i]-y_pred[i]))

# # Vẽ đồ thị hồi quy
# plt.figure(figsize=(12, 6))

# # Vẽ phân tán giữa giá trị thực tế và dự đoán
# sns.scatterplot(x=y_test, y=y_pred, color='blue', label='Dự đoán vs Thực tế')

# # Vẽ đường hồi quy
# sns.lineplot(x=y_test, y=y_test, color='red', linestyle='--', label='Đường hồi quy')

# # Thêm nhãn và tiêu đề cho đồ thị
# plt.xlabel('Thực tế')
# plt.ylabel('Dự đoán')
# plt.title('Đồ thị hồi quy')
# plt.legend()
# plt.grid(True)

# # Hiển thị đồ thị
# plt.show()
# clf = linear_model.LinearRegression().fit(X_train, y_train)
# print('w=', clf.coef_)
# print('w0=', clf.intercept_)

# # In hệ số Linear Regression
# print('Hệ số Linear Regression:', clf.coef_)


# joblib.dump(clf,'linear_model.pkl')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import linear_model

# Đọc dữ liệu
# data = pd.read_csv('./Cellphone.csv')
data = pd.read_csv('./UorLaptop.csv')

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
dt_Train, dt_Test = train_test_split(data, test_size=0.3, shuffle=False)

# Xác định biến độc lập (X) và biến phụ thuộc (y)
X_train = dt_Train.iloc[:, :6]
y_train = dt_Train.iloc[:, 6]
X_test = dt_Test.iloc[:, :6]
y_test = dt_Test.iloc[:, 6]

# Huấn luyện mô hình hồi quy tuyến tính
clf = linear_model.LinearRegression().fit(X_train, y_train)
print('w=', clf.coef_)
print('w0=', clf.intercept_)

# Dự đoán dữ liệu kiểm tra
y_pred = clf.predict(X_test)
y = np.array(y_test)

# Hàm tính chỉ số NSE
def tinh_nse(y_test, y_pred):
    numerator = np.sum((y_test - y_pred) ** 2)
    denominator = np.sum((y_pred - np.mean(y_pred)) ** 2)
    nse = 1 - (numerator / denominator)
    return nse

# In ra các chỉ số hiệu suất
print("R2: %.9f" % r2_score(y_test, y_pred))
print("NSE: %.9f" % tinh_nse(y_test, y_pred))
print("MAE: %.9f" % mean_absolute_error(y_test, y_pred))
print("RMSE: %.9f" % np.sqrt(mean_squared_error(y_test, y_pred)))
print("Thuc te Du doan Chenh lech")
for i in range(0, len(y)):
    print("%.2f" % y[i], "  ", y_pred[i], "  ", abs(y[i]-y_pred[i]))

# Vẽ đồ thị hồi quy
plt.figure(figsize=(12, 6))

# Vẽ phân tán giữa giá trị thực tế và dự đoán
sns.scatterplot(x=y_test, y=y_pred, color='blue', label='Dự đoán vs Thực tế')

# Vẽ đường hồi quy
sns.lineplot(x=y_test, y=y_test, color='red', linestyle='--', label='Đường hồi quy')

# Thêm nhãn và tiêu đề cho đồ thị
plt.xlabel('Thực tế')
plt.ylabel('Dự đoán')
plt.title('Đồ thị hồi quy')
plt.legend()
plt.grid(True)

# Hiển thị đồ thị
plt.show()
clf = linear_model.LinearRegression().fit(X_train, y_train)
print('w=', clf.coef_)
print('w0=', clf.intercept_)

# In hệ số Linear Regression
print('Hệ số Linear Regression:', clf.coef_)

clf = linear_model.LinearRegression().fit(X_train, y_train)
# Lưu mô hình Linear Regression
with open('linear_model.pkl', 'wb') as file:
    pickle.dump(clf, file)
