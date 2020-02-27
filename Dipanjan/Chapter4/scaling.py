from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from pandas import DataFrame

views = DataFrame([1295, 12, 19000, 5, 1, 300], columns = ['views'])

ss = StandardScaler()

views['zscore'] = ss.fit_transform(views[['views']])

minmax = MinMaxScaler()

views['minmax'] = minmax.fit_transform(views[['views']])

rs = RobustScaler()

views['robust'] = rs.fit_transform(views[['views']])

print(views)