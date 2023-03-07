from flask import request

import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder


class Crop_Predict(object):

    def __init__(self):
        self.data = pd.read_csv('Crop1.csv')
        self.city = pd.read_csv('Ploted_6001.csv')


    def crop(self):
        self.data = shuffle(self.data)

        y = self.data.loc[:,'Crop']
        labelEncoder_y = LabelEncoder()
        y = labelEncoder_y.fit_transform(y)

        self.data['crop_num'] = y
        X = self.data.loc[:,['N','P','K','pH','temp','climate']].astype(float)
        y = self.data.loc[:,'crop_num']


        # Training Model
        from sklearn.neighbors import KNeighborsClassifier
        clf = KNeighborsClassifier(n_jobs=3, n_neighbors=20,weights='distance')
        clf.fit(X,y)

        if request.method == 'POST':

            N = request.form['Nitrogen']
            P = request.form['Phosphorous']
            K = request.form['Potassium']
            pH = request.form['pH']
            temp = request.form['temp']
            climate = request.form['climate']
            
            
            temp = 20
            temp1 = temp + 10
            temp2 = temp - 7
            if climate == 'summer':
                climate = 1
            if climate == 'winter':
                climate = 2
            if climate == 'rainy':
                 climate = 3
            columns = ['N','P','K','pH','temp','climate']
            values = np.array([N,P,K,pH,temp,climate])
            pred = pd.DataFrame(values.reshape(-1, len(values)),columns=columns)
            prediction = clf.predict(pred)
               
            real_pred = clf.predict_proba(pred)
           
            lst = []
            for i in range(101):
                if real_pred[0][i] != 0.0:
                    lst.append(i)

            lt= []
            for i in range(10):
                load_data = self.data[self.data.index == lst[i]]
                for index, row in load_data.iterrows():
                    if row['temp'] >= temp2 and row['temp'] <= temp1:
                         if row['climate'] == climate:
                            lt.append(row['Crop'])
                
        return lt