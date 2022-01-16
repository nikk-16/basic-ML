import pandas as pd
#pandas is library used for data extraction and manipulation.
import numpy as  np 
#numpy module is use for perform numerical task or operations on data.
import matplotlib.pyplot as plt
#matplot is used for data visualization and graphical plotting library for creating static, animated, and interactive visualizations .
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import LabelEncoder 
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix,accuracy_score
#####################
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tkinter as tk
######################



df=pd.read_csv(r'C:\Users\RohanRVC\Documents\Kidney_disease_predicition/kidney_disease.csv')


df.head()


columns=pd.read_csv('C:/Users/RohanRVC/Documents/Kidney_disease_predicition/data_description.txt',sep='-')
columns=columns.reset_index()
                            
columns.columns=['cols','abb_col_names']
columns

df.head()

df.columns=columns['abb_col_names'].values
df.head()

df.dtypes

def convert_dtype(df,feature):
    df[feature]=pd.to_numeric(df[feature],errors='coerce')

features=['packed cell volume','white blood cell count','red blood cell count']

for feature in features:
    convert_dtype(df,feature)

df.dtypes

df.drop('id',axis=1,inplace=True)

def extract_cat_num(df):
    cat_col=[col for col in df.columns if df[col].dtype=='object']
    num_col=[col for col in df.columns if df[col].dtype!='object']
    return cat_col,num_col

extract_cat_num(df)

cat_col,num_col=extract_cat_num(df)

cat_col

num_col

for col in cat_col:
    print('{} has {} values '.format(col,df[col].unique()))
    print('\n')

df['diabetes mellitus'].replace(to_replace={'\tno':'no','\tyes':'yes'},inplace=True)
df['coronary artery disease']=df['coronary artery disease'].replace(to_replace='\tno',value='no')
df['class']=df['class'].replace(to_replace='ckd\t',value='ckd')

for col in cat_col:
    print('{} has {} values '.format(col,df[col].unique()))
    print('\n')

#analising distribution of data
len(num_col)

plt.figure(figsize=(30,20))
for i, feature in enumerate(num_col):
    plt.subplot(5,3,i+1)
    df[feature].hist()
    plt.title(feature)

##ckd, not ckd

len(cat_col)

plt.figure(figsize=(20,20))
for i,feature in enumerate(cat_col):
    plt.subplot(4,3,i+1)
    sns.countplot(df[feature])

sns.countplot(df['class'])

#heat Map #co relation
plt.figure(figsize=(10,8))
df.corr()
sns.heatmap(df.corr(),annot=True)

#stats
df.groupby(['red blood cells','class'])['red blood cell count'].agg(['count','mean','median','min','max'])

px.violin(df,y='red blood cell count',x='class',color='class') #max and Min 

df.columns


px.scatter(df,x='haemoglobin',y='packed cell volume')


grid=sns.FacetGrid(df,hue='class',aspect=2)
grid.map(sns.kdeplot,'red blood cell count')
grid.add_legend()

#automate analysis
def violin(col):
    fig=px.violin(df,y=col,x='color',color='class',box=True)
    return fig.show()

def scatters(col1,col2):
    fig=px.scatter(df,x=col1,y=col2,color='class')
    return fig.show()

#from this function we can plot any colums line gragh within single line
def kde_plot(feature):
    grid=sns.FacetGrid(df,hue='class',aspect=2)
    grid.map(sns.kdeplot,feature)
    grid.add_legend()

kde_plot('red blood cell count')

scatters('packed cell volume','haemoglobin') #less then 13 positive


px.violin(df,y='packed cell volume',x='class',color='class')

scatters('red blood cell count','albumin')

df.isna().sum().sort_values(ascending=False) #mising values


##normal distribution
##fill mising value with mean, median , std dev , 

sns.countplot(df['red blood cells'])

data=df.copy()

data.head()

data['red blood cells'].dropna().sample() #random value

data['red blood cells'].isnull().sum() #missing values

random_sample=data['red blood cells'].dropna().sample(152) #random selcet value
random_sample

data[data['red blood cells'].isnull()].index

random_sample.index

random_sample.index=data[data['red blood cells'].isnull()].index

random_sample

data.loc[data['red blood cells'].isnull(),'red blood cells']=random_sample

data.head()

data['red blood cells'].isnull().sum()

sns.countplot(data['red blood cells'])

def Random_value_Imputation(feature): #function for cleaning data msiisng
    random_sample=data[feature].dropna().sample(data[feature].isnull().sum())
    random_sample.index=data[data[feature].isnull()].index
    data.loc[data[feature].isnull(),feature]=random_sample

data[num_col].isnull().sum()

for col in num_col:
    Random_value_Imputation(col)

data[num_col].isnull().sum()

data[cat_col].isnull().sum()

Random_value_Imputation(' pus cell')

data['pus cell clumps'].mode()[0]

def impute_mode(feature):
    mode=data[feature].mode()[0]
    data[feature]=data[feature].fillna(mode)

for col in cat_col:
    impute_mode(col)

data[cat_col].isnull().sum()

data.head()

#cat to num 

for col in cat_col:
    print('{} has {} categories'.format(col,data[col].nunique()))

##label encoding
##normal -0
##abnormal - 1
##use case --100 

le=LabelEncoder()

for col in cat_col:
    data[col]=le.fit_transform(data[col])

data.head()

ind_col=[col for col in data.columns if col!='class']
dep_col='class'

X=data[ind_col]
y=data[dep_col]

X.head()

y

ordered_rank_features=SelectKBest(score_func=chi2,k=20)
ordered_feature=ordered_rank_features.fit(X,y)

ordered_feature

ordered_feature.scores_

datascores=pd.DataFrame(ordered_feature.scores_,columns=['Score'])

datascores

X.columns

dfcols=pd.DataFrame(X.columns)
dfcols

features_rank=pd.concat([dfcols,datascores],axis=1)
features_rank

features_rank.columns=['features','Score']
features_rank

features_rank.nlargest(10,'Score')

selected_columns=features_rank.nlargest(10,'Score')['features'].values

selected_columns

X_new=data[selected_columns]

X_new.head()

len(X_new)

X_new.shape

X_train, X_test, y_train, y_test=train_test_split(X_new,y,random_state=0,test_size=0.25)

print(X_train.shape)

print(X_test.shape)

y_train.value_counts()

XGBClassifier()

params={
    'learning_rate':[0.05,0.20,0.25],
    'max_depth':[5,8,10],
    'min_child_weight':[1,3,5,7],
    'gamma':[0.0,0.1,0.2,0.4],
    'colsample_bytree':[0.3,0.4,0.7]
    
}


classifier=XGBClassifier()

random_search=RandomizedSearchCV(classifier,param_distributions=params,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)

random_search.fit(X_train,y_train)

random_search.best_estimator_

random_search.best_params_

classifier=XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.7, gamma=0.0, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.05, max_delta_step=0, max_depth=8,
              min_child_weight=3, monotone_constraints='()',
              n_estimators=100, n_jobs=12, num_parallel_tree=1, random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)


classifier.fit(X_train,y_train)


y_pred=classifier.predict(X_test)

y_pred

confusion_matrix(y_test,y_pred)

accuracy_score(y_test,y_pred)

print("Accuracy is ",int(accuracy_score(y_test,y_pred)*100),"%")



class Predictor:

    def has_disease(self, row):
        self.train(self)
        return True if self.predict(self, row) == 1 else False

    @staticmethod
    def train(self):
        df = pd.read_csv(r'C:\Users\RohanRVC\Documents\heart attack predictor/dataset.csv')
        dataset = df
        self.standardScaler = StandardScaler()
        columns_to_scale = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak',
                            'slope', 'ca', 'thal']
        dataset[columns_to_scale] = self.standardScaler.fit_transform(dataset[columns_to_scale])
        y = dataset['target']
        X = dataset.drop(['target'], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
        self.knn_classifier = KNeighborsClassifier(n_neighbors=8)
        self.knn_classifier.fit(X, y)
        score = self.knn_classifier.score(X_test, y_test)
        print('--Training Complete--')
        print('Score: ' + str(score))

    @staticmethod
    def predict(self, row):
        user_df = np.array(row).reshape(1, 13)
        user_df = self.standardScaler.transform(user_df)
        predicted = self.knn_classifier.predict(user_df)
        print("Predicted: " + str(predicted[0]))
        return predicted[0]


#row = [[37, 1, 2, 130, 250, 0, 1, 187, 0, 3.5, 0, 0, 2]]
# row = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
# col = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
# for i in range(0, 13):
#     row[0][i] = input(f"Enter {col[i]} : ")  # OverWriting the List


la=str()
def onClick():
    row=[[age.get(),gender.get(),cp.get(),tbps.get(),chol.get(),fbs.get(),restecg.get(),thalach.get(),exang.get(),oldpeak.get(),slope.get(),ca.get(),thal.get()]]
    print(row)
    predictor = Predictor()
    o = predictor.has_disease(row)
    root2 = tk.Tk()
    root2.title("Prediction Window")
    if (o == True):
        print("Person Have Chronic Kidney Disease")
        la="Person Have Chronic Kidney Disease"
        tk.Label(root2, text=la, font=("times new roman", 20), fg="white", bg="maroon", height=2).grid(row=15, column=1)
    else:
        print("Person Is Healthy")
        la="Person Is Healthy"
        tk.Label(root2, text=la, font=("times new roman", 20), fg="white", bg="green", height=2).grid(row=15, column=1)

    return True
root = tk.Tk()
root.title("Heart Disease Predictor")
tk.Label(root,text="""Fill your Details""",font=("times new roman", 12)).grid(row=0)

#tk.Label(root,text='Patience Name',padx=20, font=("times new roman", 12)).grid(row=1,column=0)
# #patience_name = tk.IntVar()
# #tk.Entry(root,textvariable=patience_name).grid(row=1,column=1)
# tk.Label(root,text='Patience Name-:',padx=20, font=("times new roman", 12)).grid(row=3,column=0)
# cp = tk.IntVar()
# tk.Entry(root,textvariable=cp).grid(row=3,column=1)

tk.Label(root,text='Age',padx=20, font=("times new roman", 12)).grid(row=1,column=0)
age = tk.IntVar()
tk.Entry(root,textvariable=age).grid(row=1,column=1)



tk.Label(root,text="""Sex""",padx=20, font=("times new roman", 12)).grid(row=2,column=0)
gender = tk.IntVar()
tk.Radiobutton(root,text="Male (1)",padx=20,variable=gender,value=1).grid(row=2,column=1)
tk.Radiobutton(root,text="Female (0)",padx=20,variable=gender,value=0).grid(row=2,column=2)

tk.Label(root,text='White blood cell count', font=("times new roman", 12)).grid(row=3,column=0)
cp = tk.IntVar()
tk.Entry(root,textvariable=cp).grid(row=3,column=1)

tk.Label(root,text='blood Urea', font=("times new roman", 12)).grid(row=4,column=0)
tbps = tk.IntVar()
tk.Entry(root,textvariable=tbps).grid(row=4,column=1)

tk.Label(root,text='blood glucose random', font=("times new roman", 12)).grid(row=5,column=0)
chol = tk.IntVar()
tk.Entry(root,textvariable=chol).grid(row=5,column=1)

tk.Label(root,text="""fbs:serum creatinine	""",padx=20, font=("times new roman", 12)).grid(row=6,column=0)
fbs=tk.IntVar()
tk.Radiobutton(root,text="True (1)",padx=20,variable=fbs,value=1).grid(row=6,column=1)
tk.Radiobutton(root,text="False (0)",padx=20,variable=fbs,value=0).grid(row=6,column=2)

tk.Label(root,text="""restecg: packed cell volume""",padx=20, font=("times new roman", 12)).grid(row=7,column=0)
restecg=tk.IntVar()
tk.Radiobutton(root,text="0",padx=20,variable=restecg,value=0).grid(row=7,column=1)
tk.Radiobutton(root,text="1",padx=20,variable=restecg,value=1).grid(row=7,column=2)
tk.Radiobutton(root,text="2",padx=20,variable=restecg,value=2).grid(row=7,column=3)

tk.Label(root,text='albumin', font=("times new roman", 12)).grid(row=8,column=0)
thalach = tk.IntVar()
tk.Entry(root,textvariable=thalach).grid(row=8,column=1)

tk.Label(root,text="""haemoglobin """,padx=20, font=("times new roman", 12)).grid(row=9,column=0)
exang=tk.IntVar()
tk.Radiobutton(root,text="Yes (1)",padx=20,variable=exang,value=1).grid(row=9,column=1)
tk.Radiobutton(root,text="No (0)",padx=20,variable=exang,value=0).grid(row=9,column=2)

tk.Label(root,text='oldpeak : ST depression induced by exercise relative to rest', font=("times new roman", 12)).grid(row=10,column=0)
oldpeak = tk.DoubleVar()
tk.Entry(root,textvariable=oldpeak).grid(row=10,column=1)

tk.Label(root,text="""slope: the slope of the peak exercise ST segment""",padx=20, font=("times new roman", 12)).grid(row=11,column=0)
slope=tk.IntVar()
tk.Radiobutton(root,text="upsloping (0)",padx=20,variable=slope,value=0).grid(row=11,column=1)
tk.Radiobutton(root,text="flat (1)",padx=20,variable=slope,value=1).grid(row=11,column=2)
tk.Radiobutton(root,text="downsloping (2)",padx=20,variable=slope,value=2).grid(row=11,column=3)

tk.Label(root,text="""ca: number of major vessels (0-4) colored by flourosop""",padx=20, font=("times new roman", 12)).grid(row=12,column=0)
ca=tk.IntVar()
tk.Radiobutton(root,text="0",padx=20,variable=ca,value=0).grid(row=12,column=1)
tk.Radiobutton(root,text="1",padx=20,variable=ca,value=1).grid(row=12,column=2)
tk.Radiobutton(root,text="2",padx=20,variable=ca,value=2).grid(row=12,column=3)
tk.Radiobutton(root,text="3",padx=20,variable=ca,value=3).grid(row=12,column=4)
tk.Radiobutton(root,text="4",padx=20,variable=ca,value=4).grid(row=12,column=5)

tk.Label(root,text="""thal""",padx=20, font=("times new roman", 12)).grid(row=13,column=0)
thal=tk.IntVar()
tk.Radiobutton(root,text="0",padx=20,variable=thal,value=0).grid(row=13,column=1)
tk.Radiobutton(root,text="1",padx=20,variable=thal,value=1).grid(row=13,column=2)
tk.Radiobutton(root,text="2",padx=20,variable=thal,value=2).grid(row=13,column=3)
tk.Radiobutton(root,text="3",padx=20,variable=thal,value=3).grid(row=13,column=4)

tk.Button(root, text='Predict', command=onClick).grid(row=14, column=2, sticky=tk.W, pady=4)

root.mainloop()
