from numpy import NAN
from numpy import array
from numpy import sum
from numpy import mean
from numpy import count_nonzero
from pandas import isnull
from MySQLdb import connect
from getpass import getpass
from pandas import read_sql_query
from pandas import to_datetime
from sklearn.preprocessing import MinMaxScaler

db = connect(host = 'localhost', user = 'root', passwd = 'P@ul0M3l0!@#', database = 'dataScienceDB')

sqlCommand = """SELECT * FROM undData"""

df = read_sql_query(sqlCommand, db)

print("Number of rows ", df.shape[0])
print("Number of columns ", df.shape[1])
print("Colum Names", df.columns.values.tolist())
print("Columns Data Type", df.dtypes)

print("Columns with null values ", df.columns[df.isnull().any()].tolist())

print("Lines with null values ", len(isnull(df).any(1).nonzero()[0].tolist()))

print("Sample indices with missing data ", isnull(df).any(1).nonzero()[0].tolist()[0:5])

#replace columns names

#funcion that replaces column's name 
def clean_column_names(df, rename_dict={}, do_inplace = True):

	if not rename_dict:
		return df.rename(columns = {col: col.lower().replace(' ', '_') for col in df.columns.values.tolist()}, inplace = do_inplace)

	else:
		return df.rename(columns = rename_dict, inplace = do_inplace)

#rename columns
thisDict = {'UserType':'UserClass'}

clean_column_names(df, thisDict)

print("Colum Names", df.columns.values.tolist())

df['Date'] = to_datetime(df.Date)

userclass_map = {'a':0, 'b':1, 'c':2, 'd':3, 's': 4, 'v': 5, 'x': 6, 'aa':7, 'ab':8, NAN: -1}

df_normalized = df.dropna().copy()
min_max_scaler = MinMaxScaler()
norm_price = array(df_normalized['Price'])
np_scaled = min_max_scaler.fit_transform(norm_price.reshape(-1, 1))
df_normalized['normalized_price'] = np_scaled.reshape(-1, 1)

print(df.groupby(['UserClass'])['Quantity'].agg([sum, mean, count_nonzero]))