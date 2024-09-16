#eisagwgh twn aparaithtwn vivliothikwn
#import the necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

#eisagwgh kai anagnwsh tou arxeioy
#import and read the data.csv
data = pd.read_csv('data.csv')

#ektypwsh twn diastasewn tou dataset (plithos eggrafwn kai sthles)
#print the dimensions of the dataset (the number of lines and columns)
print("Dataset Dimensions : ", data.shape)

#elegxos gia mhdenikes times sto dataset
#check for missing values in the dataset
print(data.isnull().sum())

#ektypwse poses times exei i kathe mia etiketa
#count and print the number of values from every label in the dataset
print(data['label'].value_counts())

#ektypwsi gia kathe xaraktiristiko ksexwrista tin mikroteri , mesh kai megalyteri timi tous antistoixa
#print for every column the minimum value, the mean value and the maximum value
print("Nitrogen - Statistics")
print("Minimum Nitrogen  : ", data['N'].min())
print("Average Nitrogen  : ", data['N'].mean())
print("Maximum Nitrogen  : ", data['N'].max())
print("-----------------------------------------")
print("Phosphorus - Statistics")
print("Minimum Phosphorus  : ", data['P'].min())
print("Average Phosphorus  : ", data['P'].mean())
print("Maximum Phosphorus  : ", data['P'].max())
print("-----------------------------------------")
print("Potassium - Statistics")
print("Minimum Potassium  : ", data['K'].min())
print("Average Potassium  : ", data['K'].mean())
print("Maximum Potassium  : ", data['K'].max())
print("-----------------------------------------")
print("Temperature - Statistics")
print("Minimum Temperature  : {0:.3f}".format(data['temperature'].min())) #0.3f gia ta dekadika akribeias na einai 3
print("Average Temperature  : {0:.3f}".format(data['temperature'].mean()))
print("Maximum Temperature  : {0:.3f}".format(data['temperature'].max())) #0.3f for three decimals 
print("-----------------------------------------")
print("Humidity - Statistics")
print("Minimum Humidity  : {0:.3f}".format(data['humidity'].min()))
print("Average Humidity  : {0:.3f}".format(data['humidity'].mean()))
print("Maximum Humidity  : {0:.3f}".format(data['humidity'].max()))
print("-----------------------------------------")
print("ph - Statistics")
print("Minimum ph  : {0:.3f}".format(data['ph'].min()))
print("Average ph  : {0:.3f}".format(data['ph'].mean()))
print("Maximum ph  : {0:.3f}".format(data['ph'].max()))
print("-----------------------------------------")
print("Rainfall - Statistics")
print("Minimum Rainfall  : {0:.3f}".format(data['rainfall'].min()))
print("Average Rainfall  : {0:.3f}".format(data['rainfall'].mean()))
print("Maximum Rainfall  : {0:.3f}".format(data['rainfall'].max()))



#dhmiourgia plot gia kathe ena xarakthristiko ksexwrista
#create plots for every column separately
plt.subplot(5,5,1)
sns.distplot(data['N'],color ='blue')
plt.xlabel('Ratio of Nitrogen',fontsize=12)
plt.ylabel('Density',fontsize=12)
plt.grid()

plt.subplot(5,5,3)
sns.distplot(data['P'],color ='pink')
plt.xlabel('Ratio of Phosphorus',fontsize=12)
plt.ylabel('Density',fontsize=12)
plt.grid()

plt.subplot(5,5,5)
sns.distplot(data['K'],color ='darkblue')
plt.xlabel('Ratio of Potassium',fontsize=12)
plt.ylabel('Density',fontsize=12)
plt.grid()

plt.subplot(5,5,11)
sns.distplot(data['temperature'],color ='black')
plt.xlabel('Temperature',fontsize=12)
plt.ylabel('Density',fontsize=12)
plt.grid()

plt.subplot(5,5,13)
sns.distplot(data['rainfall'],color ='grey')
plt.xlabel('Rainfall',fontsize=12)
plt.ylabel('Density',fontsize=12)
plt.grid()

plt.subplot(5,5,15)
sns.distplot(data['humidity'],color ='lightgreen')
plt.xlabel('Humidity',fontsize=12)
plt.ylabel('Density',fontsize=12)
plt.grid()

plt.subplot(5,5,21)
sns.distplot(data['ph'],color ='darkgreen')
plt.xlabel('ph level',fontsize=12)
plt.ylabel('Density',fontsize=12)
plt.grid()

plt.suptitle('Distribution for Agricultural conditions', fontsize =20)
plt.show()


#ektypwsi twn etiketwn pou xreiazontai megaluterh timh natriou apo to 120
#print that labels needing nitrogen>120
print("Crops needing very high ratio of Nitrogen :", data[data['N']>120]['label'].unique())
print("Crops needing very high ratio of Phosphorus :", data[data['P']>100]['label'].unique())
print("Crops needing very high ratio of Potassium :", data[data['K']>200]['label'].unique())
print("Crops needing very high Rainfall : ", data[data['rainfall']>200]['label'].unique())
print("Crops needing very low temperature : ", data[data['temperature']<10]['label'].unique())
print("Crops needing very high temperature : ", data[data['temperature']>40]['label'].unique())
print("Crops needing very low humidity : ", data[data['humidity']<20]['label'].unique())
print("Crops needing very low ph :", data[data['ph']<4]['label'].unique())
print("Crops needing very high ph : ", data[data['ph']>9]['label'].unique())

print("Crops suitable for Summmer")
print(data[(data['temperature']>30) & (data['humidity'] > 50)]['label'].unique())
print("-----------------------------------")
print("Crops suitable for Winter")
print(data[(data['temperature']<20) & (data['humidity'] > 30)]['label'].unique())
print("-----------------------------------")
print("Crops suitable for Rain")
print(data[(data['rainfall']>200) & (data['humidity'] > 30)]['label'].unique())


#orizoume ws y thn 1h sthlh
#set y the first column (label)
y=data['label']

#diagrafoume thn 1h sthlh pou einai oi etiketes
#drop the first column (label)
x=data.drop(['label'],axis=1)

#to train kai to test ta xwrizoume se analogia 80% kai 20% (doulevei me ta idia akribws apotelesmata kai sto 75% kai 25% enw pio katw xanei ligo sthn akribeia)
#split train and test to 80% and 20% respectively (same results with split 75%-25%, but not the same with less of this)
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#get the current time
#pairnoume thn trexwn wra
start_time = time.time()

#logistic regression
model=LogisticRegression()
model.fit(x_train,y_train)
y_pred =model.predict(x_test)
plt.rcParams['figure.figsize']=(10,10)
cm=confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot= True, cmap='Wistia')
plt.title('Confusion Matrix for Logistic Rregression', fontsize = 15)
plt.show()
cr=classification_report(y_test,y_pred)
print(cr)

#get the current time again
#pairnoume thn trexwn wra ksana
end_time = time.time()

#calculate the elapsed time for logistic regression
#ypologizoume thn diafora ths telikhs me thn arxikh wra pou phrame
elapsed_time_LogReg = end_time - start_time

#print the elapsed time
#ektypwse thn wra pou perase gia na ektelestei o algorithmos
print(f"Elapsed time for Logistic Regression: {elapsed_time_LogReg} seconds")

#get the current time
#pairnoume thn trexwn wra
start_time = time.time()

#knn
model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train,y_train)
y_pred =model.predict(x_test)
plt.rcParams['figure.figsize']=(10,10)
cm=confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot= True, cmap='Wistia')
plt.title('Confusion Matrix for KNN', fontsize = 15)
plt.show()
cr=classification_report(y_test,y_pred)
print(cr)

#get the current time again
#pairnoume thn trexwn wra ksana
end_time = time.time()

#calculate the elapsed time for KNN
#ypologizoume thn diafora ths telikhs me thn arxikh wra pou phrame
elapsed_time_KNN = end_time - start_time

#print the elapsed time
#ektypwse thn wra pou perase gia na ektelestei o algorithmos
print(f"Elapsed time for KNN: {elapsed_time_KNN} seconds")

#get the current time
#pairnoume thn trexwn wra
start_time = time.time()

#Naive Bayes
model = GaussianNB()
model.fit(x_train,y_train)
y_pred =model.predict(x_test)
plt.rcParams['figure.figsize']=(10,10)
cm=confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot= True, cmap='Wistia')
plt.title('Confusion Matrix for Gaussian NB', fontsize = 15)
plt.show()
cr=classification_report(y_test,y_pred)
print(cr)

#get the current time again
#pairnoume thn trexwn wra ksana
end_time = time.time()

#calculate the elapsed time for Naive Bayes
#ypologizoume thn diafora ths telikhs me thn arxikh wra pou phrame
elapsed_time_NaiveBayes = end_time - start_time

#print the elapsed time
#ektypwse thn wra pou perase gia na ektelestei o algorithmos
print(f"Elapsed time for NaiveBayes: {elapsed_time_NaiveBayes} seconds")