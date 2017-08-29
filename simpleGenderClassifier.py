#Importar a biblioteca
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Criar uma variavel com o classificador
clf_decTree = tree.DecisionTreeClassifier()
clf_KNN = KNeighborsClassifier()

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

# Realizando a classificacao
clf_decTree = clf_decTree.fit(X, Y)
clf_KNN.fit(X, Y)

# Medindo a acuracia
pred_tree = clf_decTree.predict(X)
acc_tree = accuracy_score(Y, pred_tree) * 100
print('Accuracy for DecisionTree: {}'.format(acc_tree))

pred_KNN = clf_KNN.predict(X)
acc_KNN = accuracy_score(Y, pred_KNN) * 100
print('Accuracy for KNN: {}'.format(acc_KNN))

# testando com novos dados
predicTest = [[190, 70, 43] , [150, 50, 38]]
print("prediction Test Data = " + str(predicTest))

prediction = clf_decTree.predict(predicTest)
print("Decision Tree Prediction = " + str(prediction))

prediction = clf_KNN.predict(predicTest)
print("KNN Prediction = " + str(prediction))


