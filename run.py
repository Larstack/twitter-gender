import re
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

# Columnas que utilizaremos:
# gender: one of male, female, or brand
# description: the user's profile description
# text: text of a random one of the user's tweets
# gender:confidence: a float representing confidence in the provided gender

# Leemos el csv
file = './gender-classifier-DFE-791531.csv'
data = pd.read_csv(file, encoding='latin1')

def normalize(text):
  text = str(text)
  text = text.lower()
  # Remove non-ASCII chars
  text = re.sub('[^\x00-\x7F]+', ' ', text)
  # Remove URLs
  text = re.sub('https?:\/\/.*[\r\n]*', ' ', text)
  # Remove special chars
  text = re.sub('[?!+%{}:;.,"\'()\[\]_]', '', text)
  # Remove double spaces
  text = re.sub('\s+', ' ', text)
  return text

# normalizamos el texto
data['norm_text'] = [normalize(s) for s in data['text']]
data['norm_descr'] = [normalize(s) for s in data['description']]

# concatenamos la descripción del perfil con el texto del tweet
data['text_description'] = data['norm_text'].str.cat(data['norm_descr'], sep=' ')

# borramos los registros con gender nulos
data = data.dropna(subset=['gender'],how ='any')

# borramos los registros que tienen una confianza que no sea 1
confident_data = data[data['gender:confidence']==1]

# generamos la matriz con los tweets
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(confident_data['text_description'])

# codificamos los labels del atributo categórica gender en valores numéricos
encoder = LabelEncoder()
y = encoder.fit_transform(confident_data['gender'])

# obtenemos los sets de train y test
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4)

print("\nProcessing")
print ("-----------------------\n")
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
print ("KNeighborsClassifier")
knn.fit(X_train, y_train)
print ("Test set accuracy: {:.2f}".format(knn.score(X_test,y_test)))
print ("-----------------------\n")

from sklearn.ensemble import RandomForestClassifier
randomForest = RandomForestClassifier()
print ("RandomForestClassifier")
randomForest.fit(X_train, y_train)
print ("Test set accuracy: {:.2f}".format(randomForest.score(X_test,y_test)))
print ("-----------------------\n")

from sklearn.ensemble import GradientBoostingClassifier
gradientBoosting = GradientBoostingClassifier()
print ("GradientBoostingClassifier")
gradientBoosting.fit(X_train, y_train)
print ("Test set accuracy: {:.2f}".format(gradientBoosting.score(X_test,y_test)))
print ("-----------------------\n")

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(max_iter=1000)
print ("MLPClassifier")
mlp.fit(X_train, y_train)
print ("Test set accuracy: {:.2f}".format(mlp.score(X_test,y_test)))
print ("-----------------------\n")

from sklearn.linear_model import LogisticRegression
logistic = LogisticRegression(max_iter=1000)
print ("LogisticRegression")
logistic.fit(X_train, y_train)
print ("Test set accuracy: {:.2f}".format(logistic.score(X_test,y_test)))
print ("-----------------------\n")

from sklearn.svm import LinearSVC
svc = LinearSVC(max_iter=1000)
print ("LinearSVC")
svc.fit(X_train, y_train)
print ("Test set accuracy: {:.2f}".format(svc.score(X_test,y_test)))
print ("-----------------------\n")
