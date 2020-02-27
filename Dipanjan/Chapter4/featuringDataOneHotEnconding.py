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
from pandas import cut
from pandas import get_dummies
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

db = connect(host = 'localhost', user = 'root', passwd = 'P@ul0M3l0!@#', database = 'dataScienceDB')

sqlCommand = """SELECT * FROM pokemon"""

pokemonDF = read_sql_query(sqlCommand, db)

n_bins = 10
interval = (max(pokemonDF['speed']) - min(pokemonDF['speed']))/n_bins
bin_range = []

bin_names = [i for i in range (0, n_bins)]

for i in range (0, int(max(pokemonDF['speed'])), int(interval)):
	bin_range.append(i)

pokemonDF['speed_bin_ranges'] = cut(array(pokemonDF['speed']), bins = bin_range)

pokemonDF['speed_bin_level'] = cut(array(pokemonDF['speed']), bins = bin_range, labels = bin_names)

gle = LabelEncoder()
gen_labels = gle.fit_transform(pokemonDF['leg'])

pokemonDF['codedLeg'] = gen_labels

gen_onehot_feature = get_dummies(pokemonDF['gen'])
print(concat([pokemonDF[['name', 'gen']], gen_onehot_feature], axis = 1))