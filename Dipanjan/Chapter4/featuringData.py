from numpy import NAN
from numpy import array
from numpy import sum
from numpy import mean
from numpy import count_nonzero
from numpy import unique
from pandas import isnull
from MySQLdb import connect
from getpass import getpass
from pandas import read_sql_query
from pandas import to_datetime
from pandas import cut
from pandas import get_dummies
from pandas import concat
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_selection import VarianceThreshold

db = connect(host = 'localhost', user = 'root', passwd = 'P@ul0M3l0!@#', database = 'dataScienceDB')

sqlCommand = """SELECT * FROM pokemon"""

pokemonDF = read_sql_query(sqlCommand, db)

unique_type1 = unique(pokemonDF['type1'])

fh = FeatureHasher(n_features = 2, input_type = 'string')
hash_feat = fh.fit_transform(pokemonDF['type1'])
hash_feat = hash_feat.toarray()

pokegen = get_dummies(pokemonDF['gen'])

vt = VarianceThreshold(threshold = 0.161)

vt.fit(pokegen)

poke_gen_subset = pokegen.iloc[:,vt.get_support()]

print(poke_gen_subset)