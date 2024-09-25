# Import Required Packages
import pandas as pd
import streamlit as st
import numpy as np
import pypickle as py
#import seaborn as sns
#import matplotlib.pyplot as plt
#import graphviz
#importing the libarries
#from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LinearRegression
#from sklearn.metrics import mean_squared_error
#from sklearn import metrics
#pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
#from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler,LabelEncoder
#preprocessing

loaded_model = py.load("model.pkl")

#create a function for the prediction taking in the data(the columns to be entered by the user)

def prediction(data):
     

#create a dataframe for the data 
     df = pd.DataFrame(data)
 #convert the row to numerical form
     label = LabelEncoder()
 #Create a list for the categorical columns    
     rowitems = [0,6,13,14,15,16,17,18,20]

     for i in rowitems:
          df.iloc[i] = label.fit_transform(df.iloc[i])
#craete a variable that will convert the data to a numpy array and reshape it to a one dimensional
     num_data = df.values.reshape(1,-1)  
     #prediction the model
     pred = loaded_model.predict(num_data)     

     if  pred[0] == 0:
          return "The student that dropped out"
     elif pred[0] == 1:
          return "The student that Enrolled to Graduate School" 
     else:
           return "The stdent that graduated"   
     
def main ():
     st.title("The student Dropout Prediction Model")
     Martial_status = st.selectbox("Please select your martial status",("married","single","seperated",
                                                                    "widowed","divorced","dont know"))
     Application_mode = st.number_input("Please select your Application mode:")
     Application_order = st.number_input("Please enter application number:")
     course = st.selectbox("please select your course",("171","9254","9070","9773","8014","9991","9500",
                                                    "9238","9670","9853","9085",
     "9130","9556","9147","9003","33","9119"))
     previous_qualification= st.selectbox("please select your previous qualification",("1","19","42","39","10",
                                                                                   "3","40","2","4","12","43",
                                                                                   "15","6","9","38","5","14"))
     Nationality= st.selectbox("Please enter your Nacionality",("1","62","6","41","26",
                                                            "103","13","25","21","101","11","22",
                                                            "32""100","24","109","2","108", "105","14","17"))
     #Previous_qualification_grade = st.number_input("please enter your prvious qualification (grade):")
     Daytime_evening_attendance= st.selectbox("Enter your attendance", ("day","evening"))
     Mothers_qualification= st.selectbox("Enter your mother's qualification:", ("19","1","37","38","3",
                                                                                "4","42","2","34","12",
                                                                                " 40","9","5","39","11",
                                                                                "41","30","14","35","36",
                                                                                "6","10","29","43","18",
                                                                                "22","27","26","44"))
     Fathers_qualification= st.selectbox("Enter your father's qualification:", ('12','3','37','38','1','19',
                                                                                '5','4','34','2','39','11','9',
                                                                                '36','26','40','14','20','35',
                                                                                '41','22','13','29','43','18',
                                                                                '42','10','6','30','25','44',
                                                                                '33','27','31'))
     Mothers_occupation = st.number_input("Please enter the mother's occupation?:")
     Fathers_occupation =st.number_input("Please enter father's occupation:")
     previous_qualification_grade = st.number_input("please enter your prvious qualification (grade):")
     Admission_grade = st.number_input("Enter your admission grade: ")
     Displaced=st.selectbox(" select number of displaced:",('yes','No'))
     Educational_special_needs=st.radio("Enter number of educational  needs:",("yes","N0"))
     Debtor=st.radio("Enter number of debtor:",("yes","No"))
     Tuition_fees_up_to_date=st.radio("Enter number with paid tuition fees up to date:",("yes","No"))
     Gender=st.selectbox("select the gender:",('male','female'))
     Scholarship_holder=st.selectbox("select the number of scholarship holder:",('No','yes'))
     Age_at_enrollment=st.number_input("Please enter enrollment age:")
     International=st.selectbox("select number of international:",('No','yes'))
     Curricular_units_1st_sem_credited=st.number_input("Enter the number of credited cirricular units ist sem:")
     Curricular_units_1st_sem_enrolled=st.number_input("""Enter the number of enrolled cirricular
                                                        units ist sem:""")
     Curricular_units_1st_sem_evaluations=st.number_input("""Enter the number of evaluated curricular
                                                           units ist sem:""")
     Curricular_units_1st_sem_approved=st.number_input("Enter the number of approved units ist sem:")
     Curricular_units_1st_sem_grade=st.number_input("Enter the grade of curricular units ist sem:")
     Curricular_units_1st_sem_without_evaluations=st.number_input("""Enter the number without 
                                                                  evaluations curricular units first sem:""")
     Curricular_units_2nd_sem_credited=st.number_input("Enter the number of credited cirricular units second sem:")
     Curricular_units_2nd_sem_enrolled=st.number_input("""Enter the number of enrolled 
                                                       cirricular units 2nd sem:""")
     Curricular_units_2nd_sem_evaluations=st.number_input("""Enter the number of evaluated 
                                                          curricular units 2nd sem:""")
     Curricular_units_2nd_sem_approved=st.number_input("Enter the number of approved units 2nd sem:")
     Curricular_units_2nd_sem_grade=st.number_input("Enter the grade of curricular units 2nd sem:")
     Curricular_units_2nd_sem_without_evaluations=st.number_input("""Enter the number without 
                                                                  evaluations curricular units 2nd sem:""")
     Unemployment_rate=st.number_input("Enter unemployment rate:")
     Inflation_rate=st.number_input("Enter inflation rate:")
     GDP=st.number_input("Enter GDP")
     #Target=st.selectbox("select the Target:,"('Dropout','Graduate','Enrolled'))

     Dropout= ""

     if st.button("Result"):
          Dropout= prediction(
               [Martial_status,Application_mode,Application_order,course,previous_qualification,Nationality,
          Daytime_evening_attendance,Mothers_qualification,Fathers_qualification,Mothers_occupation,Fathers_occupation,
          previous_qualification_grade,Admission_grade,Displaced,Educational_special_needs,Debtor,Tuition_fees_up_to_date,
          Gender,Scholarship_holder,Age_at_enrollment,International,Curricular_units_1st_sem_credited,
          Curricular_units_1st_sem_enrolled,Curricular_units_1st_sem_evaluations,
          Curricular_units_1st_sem_approved,Curricular_units_1st_sem_grade,Curricular_units_1st_sem_without_evaluations,
          Curricular_units_2nd_sem_credited,Curricular_units_2nd_sem_enrolled,Curricular_units_2nd_sem_evaluations,
          Curricular_units_2nd_sem_approved,Curricular_units_2nd_sem_grade,Curricular_units_2nd_sem_without_evaluations,Unemployment_rate,Inflation_rate,
          GDP])

          st.success(Dropout)

if  __name__ == "__main__":
          main()

        

     

     


