from numpy import array
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
	
corpus = ['The sky is blue and beautiful.','Love this blue and beautiful sky!', 'The quick brown fox jumps over the lazy dog.','The brown fox is quick and the blue dog is lazy!', 'The sky is very blue and the sky is very beautiful today', 'The dog is lazy but the brown fox is quick!']

labels = ['weather', 'weather', 'animals', 'animals', 'weather', 'animals']

corpus_df = DataFrame({"Document":corpus,"Category":labels})

cv = CountVectorizer(ngram_range=(2,2))
cv_matrix = cv.fit_transform(corpus)
cv_matrix = cv_matrix.toarray()

tv = TfidfVectorizer(min_df = 0, max_df = 5, use_idf = True)
tv_matrix = tv.fit_transform(corpus)

similarity_matrix = cosine_similarity(tv_matrix.toarray())
print(similarity_matrix)