import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

root_piezo = './latent_feature/{}_piezo.npy'
root_audio = './latent_feature/{}_audio.npy'

labels = []
piezos = []
audios = []
piezos_audios = []
for i in range(10):
    piezo_features = np.load(root_piezo.format(i))
    audio_features = np.load(root_audio.format(i))
    piezo_audio_features = np.concatenate((piezo_features, audio_features), axis=1)

    piezo_list = piezo_features.tolist()
    audio_list = audio_features.tolist()
    piezo_audio_list = piezo_audio_features.tolist()

    piezos = piezos + piezo_list
    audios = audios + audio_list
    piezos_audios = piezos_audios + piezo_audio_list

    for j in range(len(piezo_list)):
        labels.append(i)

kmeans = KMeans(n_clusters=10, random_state=0)
kmeans.fit(audios)
pre_labels = kmeans.labels_
score = metrics.adjusted_rand_score(labels, pre_labels)
print("Kmeans with only audios Score:" + str(score))

kmeans = KMeans(n_clusters=10, random_state=0)
kmeans.fit(piezos)
pre_labels = kmeans.labels_
score = metrics.adjusted_rand_score(labels, pre_labels)
print("Kmeans with only piezos Score:" + str(score))

kmeans = KMeans(n_clusters=10, random_state=0)
kmeans.fit(piezos_audios)
pre_labels = kmeans.labels_
score = metrics.adjusted_rand_score(labels, pre_labels)
print("Kmeans with piezo and audio(Simple concat) Score:" + str(score))

X_train, X_test, y_train, y_test = train_test_split(audios, labels, test_size=0.2, random_state=10, shuffle=True)
tree_clf = DecisionTreeClassifier(random_state=10)
tree_clf.fit(X_train, y_train)
predictions = tree_clf.predict(X_test)
print("Decision tree for only audios")
print(classification_report(y_test, predictions, digits=4))

X_train, X_test, y_train, y_test = train_test_split(piezos, labels, test_size=0.2, random_state=10, shuffle=True)
tree_clf = DecisionTreeClassifier(random_state=10)
tree_clf.fit(X_train, y_train)
predictions = tree_clf.predict(X_test)
print("Decision tree for only piezos")
print(classification_report(y_test, predictions, digits=4))

X_train, X_test, y_train, y_test = train_test_split(piezos_audios, labels, test_size=0.2, random_state=10, shuffle=True)
tree_clf = DecisionTreeClassifier(random_state=10)
tree_clf.fit(X_train, y_train)
predictions = tree_clf.predict(X_test)
print("Decision tree for audio and piezo (Simple Concat)")
print(classification_report(y_test, predictions, digits=4))
