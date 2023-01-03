import numpy as np
import pandas as pd

# 下面的csv文件可在上面的推荐链接下载
airline_data = pd.read_csv(r"E:\课程\SecondYear\004_Yang\001_Python\航空公司客户价值分析\air_data.csv",
                           encoding="gb18030")  # 导入航空数据
print('原始数据的形状为：', airline_data.shape)

# 去除票价为空的记录
exp1 = airline_data["SUM_YR_1"].notnull()
exp2 = airline_data["SUM_YR_2"].notnull()
exp = exp1 & exp2
airline_notnull = airline_data.loc[exp, :]
print('删除缺失记录后数据的形状为：', airline_notnull.shape)

# 只保留票价非零的，或者平均折扣率不为0且总飞行公里数大于0的记录。
index1 = airline_notnull['SUM_YR_1'] != 0
index2 = airline_notnull['SUM_YR_2'] != 0
index3 = (airline_notnull['SEG_KM_SUM'] > 0) & (airline_notnull['avg_discount'] != 0)
airline = airline_notnull[(index1 | index2) & index3]
print('删除异常记录后数据的形状为：', airline.shape)

# 选取需求特征
airline_selection = airline[["FFP_DATE", "LOAD_TIME", "FLIGHT_COUNT", "LAST_TO_END", "avg_discount", "SEG_KM_SUM"]]

# 构建L特征
L = pd.to_datetime(airline_selection["LOAD_TIME"]) - pd.to_datetime(airline_selection["FFP_DATE"])

# 提取数字，由于模型中L单位为：月，所以需要除以30
# L = L.astype("str").str.split(' ').str[0]
# L = L.astype("int")/30

# 对L这一列应用lambda函数，对L中的每一个x都执行函数操作
L = L.apply(lambda x: round(int(str(x).split(' ')[0]) / 30, 2))

# 合并特征
airline_features = pd.concat([L, airline_selection.iloc[:, 2:]], axis=1)
print('构建的LRFMC特征前5行为：\n', airline_features.head())

# 标准差标准化: 使用sklearn 中preprocessing 模块的StandardScaler 函数;
# 也可以使用自定义的方法（数据分析中标准化方法,因为此处不需要对训练集与测试集用同一套规则）
from sklearn.preprocessing import StandardScaler  ##标准差标准化

data = StandardScaler().fit_transform(airline_features)
np.savez('airline_scale.npz', data)
print('标准化后LRFMC五个特征为：\n', data[:5, :])
