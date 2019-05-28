from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer

corpus=[
    'UNC played Duke in basketball',
    'Duke lokst the basketball game',
    'I ate a sandwich',
    'I am gathering ingredients for the sandwich',
    'There were many wizards at the gathering'
]

vectorizer = TfidfVectorizer(stop_words='english')
print(vectorizer.fit_transform(corpus).todense())
print(vectorizer.vocabulary_)