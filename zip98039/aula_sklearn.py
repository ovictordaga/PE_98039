import pandas as pd
import numpy as np
from sklearn import tree

train = pd.read_csv('media_data_98039.csv')    
y_train = train['CARO']
x_train = train.drop(['CARO'], axis=1).values 
decision_tree = tree.DecisionTreeClassifier(max_depth = 20)
decision_tree.fit(x_train, y_train)

with open("arvore98039.dot", 'w') as f:
                              f = tree.export_graphviz(decision_tree,
                              out_file=f,
                              max_depth = 20,
                              impurity = True,
                              feature_names = list(train.drop(['CARO'], axis=1)),
                              class_names = ['ABAIXO', 'MEDIA','ACIMA'],
                              rounded = True,
                              filled= True )
        