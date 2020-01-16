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

sqlCommand = """SELECT * FROM pokemon"""

pokemonDF = read_sql_query(sqlCommand, db)

print(pokemonDF[pokemonDF['speed']==max(pokemonDF['speed'])])