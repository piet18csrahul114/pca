import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
st.set_option('deprecation.showfileUploaderEncoding', False)
# Load the pickled model
model = pickle.load(open('/content/drive/My Drive/pca.pkl','rb'))

dataset = pd.read_csv('Classification Dataset1.csv')
dataset=dataset.drop(['Surname'], axis = 1)

X = dataset.iloc[:, :-1].values

# Taking care of missing data
#handling missing data (Replacing missing data with the mean value)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.NAN, strategy= 'constant', fill_value='Male', verbose=1, copy=True)
#Fitting imputer object to the independent variables x.
imputer = imputer.fit(X[:, [2]])
#Replacing missing data with the calculated mean value
X[:, [2]]= imputer.transform(X[:, [2]])

from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 2] = labelencoder_X.fit_transform(X[:, 2])

from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])

# Taking care of missing data
#handling missing data (Replacing missing data with the mean value)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.NAN, strategy= 'mean', fill_value=None, verbose=1, copy=True)
#Fitting imputer object to the independent variables x.
imputer = imputer.fit(X[:, :])
#Replacing missing data with the calculated mean value
X[:, :]= imputer.transform(X[:,:])

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

def predict_note_authentication(CreditScore1,Geography1,Gender1,Age1,Tenure1,Balance1,HasCrCard1,IsActiveMember1,EstimatedSalary1,
                                CreditScore2,Geography2,Gender2,Age2,Tenure2,Balance2,HasCrCard2,IsActiveMember2,EstimatedSalary2,
                                CreditScore3,Geography3,Gender3,Age3,Tenure3,Balance3,HasCrCard3,IsActiveMember3,EstimatedSalary3,
                                CreditScore4,Geography4,Gender4,Age4,Tenure4,Balance4,HasCrCard4,IsActiveMember4,EstimatedSalary4,
                                CreditScore5,Geography5,Gender5,Age5,Tenure5,Balance5,HasCrCard5,IsActiveMember5,EstimatedSalary5,
                                CreditScore6,Geography6,Gender6,Age6,Tenure6,Balance6,HasCrCard6,IsActiveMember6,EstimatedSalary6):
  X1=sc.fit_transform([[CreditScore1,Geography1,Gender1,Age1,Tenure1,Balance1,HasCrCard1,IsActiveMember1,EstimatedSalary1,],
                      [CreditScore2,Geography2,Gender2,Age2,Tenure2,Balance2,HasCrCard2,IsActiveMember2,EstimatedSalary2],
                      [CreditScore3,Geography3,Gender3,Age3,Tenure3,Balance3,HasCrCard3,IsActiveMember3,EstimatedSalary3],
                      [CreditScore4,Geography4,Gender4,Age4,Tenure4,Balance4,HasCrCard4,IsActiveMember4,EstimatedSalary4],
                      [CreditScore5,Geography5,Gender5,Age5,Tenure5,Balance5,HasCrCard5,IsActiveMember5,EstimatedSalary5],
                      [CreditScore6,Geography6,Gender6,Age6,Tenure6,Balance6,HasCrCard6,IsActiveMember6,EstimatedSalary6]])
  # Applying PCA
  from sklearn.decomposition import PCA
  pca = PCA(n_components = 6)
  X1 = pca.fit_transform(X1)

  output = model.predict(X1)
  res=[]
  for i in output:
    if i==[0]:
      res.append("Customer will not Leave")
    else:
      res.append("Customer will Leave")
  #print(prediction)
  return res
def main():
    
    html_temp = """
   <div class="" style="background-color:orange;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
   <center><p style="font-size:30px;color:white;margin-top:10px;">Department of Computer Engineering</p></center> 
   <center><p style="font-size:25px;color:white;margin-top:10px;">ML Lab Experiment-11</p></center> 
   </div>
   </div>
   </div>
   """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.header("Customer Prediction")
    Age1 = st.number_input('Insert a Age',18,60)
    CreditScore1= st.number_input('Insert a CreditScore',400,1000)
    HasCrCard1 = st.number_input('Insert a HasCrCard 0 For No 1 For Yes',0,1)
    Tenure1 = st.number_input('Insert a Tenure',0,20)
    Balance1 = st.number_input('Insert a Balance',0)
    Gender1 = st.number_input('Insert 0 For Male 1 For Female ',0,1)
    Geography1= st.number_input('Insert Geography 0 For France 1 For Spain',0,1)
    IsActiveMember1= st.number_input('Insert a IsActiveMember 0 For No 1 For Yes',0,1)
    EstimatedSalary1= st.number_input('Insert a EstimatedSalary',0)

    Age2 = st.number_input('Insert a Age2',18,60)
    CreditScore2= st.number_input('Insert a CreditScore2',400,1000)
    HasCrCard2 = st.number_input('Insert a HasCrCard2 0 For No 1 For Yes',0,1)
    Tenure2 = st.number_input('Insert a Tenure2',0,20)
    Balance2 = st.number_input('Insert a Balance2',0)
    Gender2 = st.number_input('Insert 0 For Male2 1 For Female ',0,1)
    Geography2= st.number_input('Insert Geography2 0 For France 1 For Spain',0,1)
    IsActiveMember2= st.number_input('Insert a IsActiveMember2 0 For No 1 For Yes',0,1)
    EstimatedSalary2= st.number_input('Insert a EstimatedSalary2',0)

    Age3 = st.number_input('Insert a Age3',18,60)
    CreditScore3= st.number_input('Insert a CreditScore3',400,1000)
    HasCrCard3 = st.number_input('Insert a HasCrCard3 0 For No 1 For Yes',0,1)
    Tenure3 = st.number_input('Insert a Tenure3',0,20)
    Balance3 = st.number_input('Insert a Balance3',0)
    Gender3 = st.number_input('Insert 0 For Male3 1 For Female ',0,1)
    Geography3= st.number_input('Insert Geography3 0 For France 1 For Spain',0,1)
    IsActiveMember3= st.number_input('Insert a IsActiveMember3 0 For No 1 For Yes',0,1)
    EstimatedSalary3= st.number_input('Insert a EstimatedSalary3',0)

    Age4 = st.number_input('Insert a Age4',18,60)
    CreditScore4= st.number_input('Insert a CreditScore4',400,1000)
    HasCrCard4 = st.number_input('Insert a HasCrCard4 0 For No 1 For Yes',0,1)
    Tenure4 = st.number_input('Insert a Tenure4',0,20)
    Balance4 = st.number_input('Insert a Balance4',0)
    Gender4 = st.number_input('Insert 0 For Male4 1 For Female4 ',0,1)
    Geography4= st.number_input('Insert Geography4 0 For France 1 For Spain',0,1)
    IsActiveMember4= st.number_input('Insert a IsActiveMember4 0 For No 1 For Yes',0,1)
    EstimatedSalary4= st.number_input('Insert a EstimatedSalary4',0)

    Age5 = st.number_input('Insert a Age5',18,60)
    CreditScore5= st.number_input('Insert a CreditScore5',400,1000)
    HasCrCard5 = st.number_input('Insert a HasCrCard5 0 For No 1 For Yes',0,1)
    Tenure5 = st.number_input('Insert a Tenure5',0,20)
    Balance5 = st.number_input('Insert a Balance5',0)
    Gender5 = st.number_input('Insert 0 For Male5 1 For Female5 ',0,1)
    Geography5= st.number_input('Insert Geography5 0 For France 1 For Spain',0,1)
    IsActiveMember5= st.number_input('Insert a IsActiveMember5 0 For No 1 For Yes',0,1)
    EstimatedSalary5= st.number_input('Insert a EstimatedSalary5',0)

    Age6 = st.number_input('Insert a Age6',18,60)
    CreditScore6= st.number_input('Insert a CreditScore6',400,1000)
    HasCrCard6 = st.number_input('Insert a HasCrCard6 0 For No 1 For Yes',0,1)
    Tenure6 = st.number_input('Insert a Tenure6',0,20)
    Balance6 = st.number_input('Insert a Balance6',0)
    Gender6 = st.number_input('Insert 0 For Male6 1 For Female6 ',0,1)
    Geography6= st.number_input('Insert Geography6 0 For France 1 For Spain',0,1)
    IsActiveMember6= st.number_input('Insert a IsActiveMember6 0 For No 1 For Yes',0,1)
    EstimatedSalary6= st.number_input('Insert a EstimatedSalary6',0)
    
    
    resul=""
    if st.button("Predict"):
      result=predict_note_authentication(CreditScore1,Geography1,Gender1,Age1,Tenure1,Balance1,HasCrCard1,IsActiveMember1,EstimatedSalary1,
                                         CreditScore2,Geography2,Gender2,Age2,Tenure2,Balance2,HasCrCard2,IsActiveMember2,EstimatedSalary2,
                                         CreditScore3,Geography3,Gender3,Age3,Tenure3,Balance3,HasCrCard3,IsActiveMember3,EstimatedSalary3,
                                         CreditScore4,Geography4,Gender4,Age4,Tenure4,Balance4,HasCrCard4,IsActiveMember4,EstimatedSalary4,
                                         CreditScore5,Geography5,Gender5,Age5,Tenure5,Balance5,HasCrCard5,IsActiveMember5,EstimatedSalary5,
                                         CreditScore6,Geography6,Gender6,Age6,Tenure6,Balance6,HasCrCard6,IsActiveMember6,EstimatedSalary6)
      st.success('Model has predicted {}'.format(result))
    if st.button("About"):
      st.subheader("Developed by RAHUL CHHABLANI")
      st.subheader("C-Section,PIET")

if __name__=='__main__':
  main()