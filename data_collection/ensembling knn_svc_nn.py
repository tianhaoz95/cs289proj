import pandas as pd
from sklearn.externals import joblib
import json
from random import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler2 = StandardScaler()

try:
    data = joblib.load('data.pkl')
    train = joblib.load('train.pkl')
    test = joblib.load('test.pkl')

except:
    songs = pd.read_csv('songs.csv', encoding='ISO-8859-1')
    data = songs[['danceability','energy','loudness','mode','speechiness',
                  'instrumentalness','liveness','valence','tempo','genre_1','tempo_1','mood_1']]
    data = data.copy()
    data['genre_1'].fillna(0, inplace=True)
    genres = {}
    tempos = {}
    moods = {}
    indmood = 1
    indgenres = -4
    indtempos = -1
    for index, row in data[['genre_1','tempo_1','mood_1']].iterrows():
        if row['genre_1'] not in genres:
            genres[row['genre_1']] = indgenres
            indgenres+=1
        if row['tempo_1'] not in tempos:
            tempos[row['tempo_1']] = indtempos
            indtempos+=1
        if row['mood_1'] not in moods:
            moods[row['mood_1']] = indmood
            indmood+=1
        data['genre_1'].replace([row['genre_1']], genres[row['genre_1']],inplace=True)
        data['tempo_1'].replace([row['tempo_1']], tempos[row['tempo_1']],inplace=True)
        data['mood_1'].replace([row['mood_1']], moods[row['mood_1']],inplace=True)
    # Normalizing column values
    data[['tempo']] = data[['tempo']].apply(lambda x: x / x.max())
    scaler.fit(data['loudness'].reshape(-1, 1))
    data['loudness'] = scaler.transform(data['loudness'].reshape(-1, 1))
    scaler2.fit(data['instrumentalness'].reshape(-1,1))
    data['instrumentalness'] = scaler2.transform(data['instrumentalness'].reshape(-1, 1))

    joblib.dump(data,'data.pkl')
    train, test = train_test_split(data, test_size=0.15)
    joblib.dump(train, 'train.pkl')
    joblib.dump(test, 'test.pkl')

print('Data loaded and preprocessed',data.columns)



# Analysis section ------------------------------------------------------
#
# To see number of songs per artist
# grouped = songs.groupby('artist')['id'].nunique()
# # maximum songs by any artist
# # checking type of grouped via type(grouped) = pandas.core.series.Series
# # top 20 frequencies
# top20 = grouped.nlargest(20)
# # Number of occurences frequencies
# freq = grouped.value_counts(1)
# Let's remove the artists since the frequencies look like this --
# 1     0.750250
# 2     0.165388
# 3     0.040013
# 75% of the artists occur only once in the data
# Analysis section ------------------------------------------------------






y = train['mood_1']
X = train.drop('mood_1', axis=1)

ytest = test['mood_1']
Xtest = test.drop('mood_1',axis=1)

# Neural network
from sklearn.neural_network import MLPClassifier
try:
    mlp = joblib.load('mlp.pkl')
except:
    mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10),
                        solver='lbfgs', activation='relu',
                        learning_rate_init=0.5, max_iter=10000)
    mlp.fit(X, y)
    joblib.dump(mlp, 'mlp.pkl')

print("Only Neural Network score=",mlp.score(Xtest,ytest))

from sklearn import svm
try:
    svmclf = joblib.load('svmclf.pkl')
except:
    svmclf = svm.SVC()
    svmclf.fit(X,y)
    joblib.dump(svmclf,'svmclf.pkl')

print("Only SVC Score=",svmclf.score(Xtest,ytest))

from sklearn.neighbors import KNeighborsClassifier
try:
    knn = joblib.load('knn.pkl')
except:
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X, y)
    joblib.dump(knn, 'knn.pkl')

print("Only KNN Score=",knn.score(Xtest,ytest))


# Ensembling now
Xtestensemble = Xtest.copy()
Xtestensemble['mlp'] = mlp.predict(Xtest)
Xtestensemble['svc'] = svmclf.predict(Xtest)
Xtestensemble['knn'] = knn.predict(Xtest)


Xensemble = X.copy()
Xensemble['mlp'] = mlp.predict(X)
Xensemble['svc'] = svmclf.predict(X)
Xensemble['knn'] = svmclf.predict(X)
print(X.shape,Xensemble.shape,Xtest.shape,Xtestensemble.shape)
try:
    mlpensemble = joblib.load('mlpensemble.pkl')
except:
    mlpensemble = MLPClassifier(hidden_layer_sizes=(100, 100, 100,100),
                        solver='lbfgs', activation='relu',
                        learning_rate_init=0.5, max_iter=1000)
    mlpensemble.fit(Xensemble, y)
    joblib.dump(mlpensemble, 'mlpensemble.pkl')
print('Ensembling using MLP score=',mlpensemble.score(Xtestensemble,ytest))



try:
    svmensemble = joblib.load('svmensemble.pkl')
except:
    svmensemble = svm.SVC()
    svmensemble.fit(Xensemble,y)
    joblib.dump(svmensemble,'svmensemble.pkl')
print('Ensembling using SVC score=',svmensemble.score(Xtestensemble,ytest))


try:
    knnensemble = joblib.load('knnensemble.pkl')
except:
    knnensemble = KNeighborsClassifier(n_neighbors=5)
    knnensemble.fit(Xensemble, y)
    joblib.dump(knnensemble, 'knnensemble.pkl')

print('Ensembling using KNN score=',knnensemble.score(Xtestensemble,ytest))