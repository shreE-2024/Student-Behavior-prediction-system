import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns

# Disable the warning about pyplot global use
st.set_option('deprecation.showPyplotGlobalUse', False)

# Load the data
@st.cache_data()
def load_data():
    return pd.read_csv("students_data.csv")

df = load_data()

# Define default action
action = "Search Student"

# Title at the top of the page
st.title("Student Behavior Prediction")

# Main area option for user to choose action
action = st.radio("Select Action", ("Search Student", "Add New Student Entry", "Survey"))

# Sidebar - About
st.sidebar.title("About")
st.sidebar.info(
    "This web app is designed to predict student behavior based on various factors. "
    "The models included are Random Forest, Logistic Regression, KNN, Decision Tree, and Gradient Boosting."
)

# Sidebar - Connect with Us
st.sidebar.title("Connect with Us")
st.sidebar.info(
    "For any queries or support, feel free to reach out to us at [email protected]"
)

# Define categories
def get_category(CGPA, semester_marks_avg, attendance, study_time):
    if study_time >= 2:        
        if attendance >= 90:            
            if semester_marks_avg >= 90:                
                if CGPA * 10 >= 90:                       
                    return "Excellence"
                elif 80 <= CGPA * 10 < 90:
                    return "Good"
                elif 60 <= CGPA * 10 < 80:        
                    return "Good"
                elif 50 <= CGPA * 10 < 60:        
                    return "Better"    
                elif 40 <= CGPA * 10 < 50:        
                    return "Low"    
                elif 30 <= CGPA * 10 < 40:        
                    return "Bad"    
                elif CGPA * 10 < 30:        
                    return "Very Bad"    
                else:        
                    return "Undefined"
            elif 80 <= semester_marks_avg < 90:
                if CGPA * 10 >= 90:                       
                    return "Excellence"
                elif 80 <= CGPA * 10 < 90:
                    return "Good"
                elif 60 <= CGPA * 10 < 80:        
                    return "Good"
                elif 50 <= CGPA * 10 < 60:        
                    return "Better"    
                elif 40 <= CGPA * 10 < 50:        
                    return "Average"    
                elif 30 <= CGPA * 10 < 40:        
                    return "Low"    
                elif CGPA * 10 < 30:        
                    return "Bad"    
                else:        
                    return "Undefined"        
                    
            elif 60 <= semester_marks_avg < 80:  
                if CGPA * 10 >= 90:                       
                    return "Good"
                elif 80 <= CGPA * 10 < 90:
                    return "Good"
                elif 60 <= CGPA * 10 < 80:        
                    return "Good"
                elif 50 <= CGPA * 10 < 60:        
                    return "Better"    
                elif 40 <= CGPA * 10 < 50:        
                    return "Average"    
                elif 30 <= CGPA * 10 < 40:        
                    return "Low"    
                elif CGPA * 10 < 30:        
                    return " Bad"    
                else:        
                    return "Undefined"    
            elif 50 <= semester_marks_avg < 60: 
                if CGPA * 10 >= 90:                                           
                    return "Good"
                elif 80 <= CGPA * 10 < 90:
                    return "Better"
                elif 60 <= CGPA * 10 < 80:        
                    return "Better"
                elif 50 <= CGPA * 10 < 60:        
                    return "Average"    
                elif 40 <= CGPA * 10 < 50:        
                    return "Low"    
                elif 30 <= CGPA * 10 < 40:        
                    return "Low"    
                elif CGPA * 10 < 30:        
                    return "Bad"    
                else:        
                    return "Undefined"
            elif 40 <= semester_marks_avg < 50:
                if CGPA * 10 >= 90:                       
                    return "Good"
                elif 80 <= CGPA * 10 < 90:
                    return "Good"
                elif 60 <= CGPA * 10 < 80:        
                    return "Better"
                elif 50 <= CGPA * 10 < 60:        
                    return "Average"    
                elif 40 <= CGPA * 10 < 50:        
                    return "Low"    
                elif 30 <= CGPA * 10 < 40:        
                    return "Bad"    
                elif CGPA * 10 < 30:        
                    return "Bad"    
                else:        
                    return "Undefined"    
            elif 30 <= semester_marks_avg < 40:
                if CGPA * 10 >= 90:                       
                    return "Better"
                elif 80 <= CGPA * 10 < 90:
                    return "Better"
                elif 60 <= CGPA * 10 < 80:        
                    return "Average"
                elif 50 <= CGPA * 10 < 60:        
                    return "Low"    
                elif 40 <= CGPA * 10 < 50:        
                    return "Low"    
                elif 30 <= CGPA * 10 < 40:        
                    return "Bad"    
                elif CGPA * 10 < 30:        
                    return "Very Bad"    
                else:        
                    return "Undefined"
                   
            elif semester_marks_avg < 30:
                if CGPA * 10 >= 90:                       
                    return "Good"
                elif 80 <= CGPA * 10 < 90:
                    return "Better"
                elif 60 <= CGPA * 10 < 80:        
                    return "Average"
                elif 50 <= CGPA * 10 < 60:        
                    return "Low"    
                elif 40 <= CGPA * 10 < 50:        
                    return "Low"    
                elif 30 <= CGPA * 10 < 40:        
                    return "Bad"    
                elif CGPA * 10 < 30:        
                    return "Very Bad"    
                else:        
                    return "Undefined"    
            else:        
                return "Undefined"
        elif 80 <= attendance < 90:
            if semester_marks_avg >= 90:                
                if CGPA * 10 >= 90:                       
                    return "Excellence"
                elif 80 <= CGPA * 10 < 90:
                    return "Good"
                elif 60 <= CGPA * 10 < 80:        
                    return "Better"
                elif 50 <= CGPA * 10 < 60:        
                    return "Better"    
                elif 40 <= CGPA * 10 < 50:        
                    return "Average"    
                elif 30 <= CGPA * 10 < 40:        
                    return "Low"    
                elif CGPA * 10 < 30:        
                    return "Bad"    
                else:        
                    return "Undefined"
            elif 80 <= semester_marks_avg < 90:
                if CGPA * 10 >= 90:                       
                    return "Excellence"
                elif 80 <= CGPA * 10 < 90:
                    return "Good"
                elif 60 <= CGPA * 10 < 80:        
                    return "Good"
                elif 50 <= CGPA * 10 < 60:        
                    return "Better"    
                elif 40 <= CGPA * 10 < 50:        
                    return "Average"    
                elif 30 <= CGPA * 10 < 40:        
                    return "Low"    
                elif CGPA * 10 < 30:        
                    return " Bad"    
                else:        
                    return "Undefined"        
                    
            elif 60 <= semester_marks_avg < 80:  
                if CGPA * 10 >= 90:                       
                    return "Good"
                elif 80 <= CGPA * 10 < 90:
                    return "Good"
                elif 60 <= CGPA * 10 < 80:        
                    return "Good"
                elif 50 <= CGPA * 10 < 60:        
                    return "Better"    
                elif 40 <= CGPA * 10 < 50:        
                    return "Low"    
                elif 30 <= CGPA * 10 < 40:        
                    return "Bad"    
                elif CGPA * 10 < 30:        
                    return "Very Bad"    
                else:        
                    return "Undefined"    
            elif 50 <= semester_marks_avg < 60: 
                if CGPA * 10 >= 90:                                           
                    return "Good"
                elif 80 <= CGPA * 10 < 90:
                    return "Bettr"
                elif 60 <= CGPA * 10 < 80:        
                    return "Better"
                elif 50 <= CGPA * 10 < 60:        
                    return "Average"    
                elif 40 <= CGPA * 10 < 50:        
                    return "Low"    
                elif 30 <= CGPA * 10 < 40:        
                    return "Bad"    
                elif CGPA * 10 < 30:        
                    return "Very Bad"    
                else:        
                    return "Undefined"
            elif 40 <= semester_marks_avg < 50:
                if CGPA * 10 >= 90:                       
                    return "better"
                elif 80 <= CGPA * 10 < 90:
                    return "Average"
                elif 60 <= CGPA * 10 < 80:        
                    return "Average"
                elif 50 <= CGPA * 10 < 60:        
                    return "Average"    
                elif 40 <= CGPA * 10 < 50:        
                    return "Low"    
                elif 30 <= CGPA * 10 < 40:        
                    return "Bad"    
                elif CGPA * 10 < 30:        
                    return "Very Bad"    
                else:        
                    return "Undefined"    
            elif 30 <= semester_marks_avg < 40:
                if CGPA * 10 >= 90:                       
                    return "Better"
                elif 80 <= CGPA * 10 < 90:
                    return "Low"
                elif 60 <= CGPA * 10 < 80:        
                    return "Low"
                elif 50 <= CGPA * 10 < 60:        
                    return "Low"    
                elif 40 <= CGPA * 10 < 50:        
                    return "Low"    
                elif 30 <= CGPA * 10 < 40:        
                    return "Bad"    
                elif CGPA * 10 < 30:        
                    return "Very Bad"    
                else:        
                    return "Undefined"
                   
            elif semester_marks_avg < 30:
                if CGPA * 10 >= 90:                       
                    return "Low"
                elif 80 <= CGPA * 10 < 90:
                    return "Low"
                elif 60 <= CGPA * 10 < 80:        
                    return "Low"
                elif 50 <= CGPA * 10 < 60:        
                    return "Low"    
                elif 40 <= CGPA * 10 < 50:        
                    return "Low"    
                elif 30 <= CGPA * 10 < 40:        
                    return "Bad"    
                elif CGPA * 10 < 30:        
                    return "Very Bad"    
                else:        
                    return "Undefined"    
            else:        
                return "Undefined"   
        elif  60 <= attendance < 80:  
            if semester_marks_avg >= 90:                
                if CGPA * 10 >= 90:                       
                    return "Excellence"
                elif 80 <= CGPA * 10 < 90:
                    return "Good"
                elif 60 <= CGPA * 10 < 80:        
                    return "Better"
                elif 50 <= CGPA * 10 < 60:        
                    return "Average"    
                elif 40 <= CGPA * 10 < 50:        
                    return "Low"    
                elif 30 <= CGPA * 10 < 40:        
                    return "Bad"    
                elif CGPA * 10 < 30:        
                    return "Very Bad"    
                else:        
                    return "Undefined"
            elif 80 <= semester_marks_avg < 90:
                if CGPA * 10 >= 90:                       
                    return "Excellence"
                elif 80 <= CGPA * 10 < 90:
                    return "Good"
                elif 60 <= CGPA * 10 < 80:        
                    return "Better"
                elif 50 <= CGPA * 10 < 60:        
                    return "Average"    
                elif 40 <= CGPA * 10 < 50:        
                    return "Low"    
                elif 30 <= CGPA * 10 < 40:        
                    return "Bad"    
                elif CGPA * 10 < 30:        
                    return "Very Bad"    
                else:        
                    return "Undefined"        
                    
            elif 60 <= semester_marks_avg < 80:  
                if CGPA * 10 >= 90:                       
                    return "Better"
                elif 80 <= CGPA * 10 < 90:
                    return "Better"
                elif 60 <= CGPA * 10 < 80:        
                    return "Better"
                elif 50 <= CGPA * 10 < 60:        
                    return "Average"    
                elif 40 <= CGPA * 10 < 50:        
                    return "Low"    
                elif 30 <= CGPA * 10 < 40:        
                    return "Bad"    
                elif CGPA * 10 < 30:        
                    return "Very Bad"    
                else:        
                    return "Undefined"    
            elif 50 <= semester_marks_avg < 60: 
                if CGPA * 10 >= 90:                                           
                    return "Better"
                elif 80 <= CGPA * 10 < 90:
                    return "Bettr"
                elif 60 <= CGPA * 10 < 80:        
                    return "Better"
                elif 50 <= CGPA * 10 < 60:        
                    return "Average"    
                elif 40 <= CGPA * 10 < 50:        
                    return "Low"    
                elif 30 <= CGPA * 10 < 40:        
                    return "Bad"    
                elif CGPA * 10 < 30:        
                    return "Very Bad"    
                else:        
                    return "Undefined"
            elif 40 <= semester_marks_avg < 50:
                if CGPA * 10 >= 90:                       
                    return "Avrage"
                elif 80 <= CGPA * 10 < 90:
                    return "Average"
                elif 60 <= CGPA * 10 < 80:        
                    return "Average"
                elif 50 <= CGPA * 10 < 60:        
                    return "Average"    
                elif 40 <= CGPA * 10 < 50:        
                    return "Low"    
                elif 30 <= CGPA * 10 < 40:        
                    return "Bad"    
                elif CGPA * 10 < 30:        
                    return "Very Bad"    
                else:        
                    return "Undefined"    
            elif 30 <= semester_marks_avg < 40:
                if CGPA * 10 >= 90:                       
                    return "Low"
                elif 80 <= CGPA * 10 < 90:
                    return "Low"
                elif 60 <= CGPA * 10 < 80:        
                    return "Low"
                elif 50 <= CGPA * 10 < 60:        
                    return "Low"    
                elif 40 <= CGPA * 10 < 50:        
                    return "Low"    
                elif 30 <= CGPA * 10 < 40:        
                    return "Bad"    
                elif CGPA * 10 < 30:        
                    return "Very Bad"    
                else:        
                    return "Undefined"
                   
            elif semester_marks_avg < 30:
                if CGPA * 10 >= 90:                       
                    return "Low"
                elif 80 <= CGPA * 10 < 90:
                    return "Low"
                elif 60 <= CGPA * 10 < 80:        
                    return "Low"
                elif 50 <= CGPA * 10 < 60:        
                    return "Low"    
                elif 40 <= CGPA * 10 < 50:        
                    return "Low"    
                elif 30 <= CGPA * 10 < 40:        
                    return "Bad"    
                elif CGPA * 10 < 30:        
                    return "Very Bad"    
                else:        
                    return "Undefined"    
            else:        
                return "Undefined"    
        elif  50 <= attendance < 60:
            if semester_marks_avg >= 90:                
                if CGPA * 10 >= 90:                       
                    return "Excellence"
                elif 80 <= CGPA * 10 < 90:
                    return "Good"
                elif 60 <= CGPA * 10 < 80:        
                    return "Better"
                elif 50 <= CGPA * 10 < 60:        
                    return "Average"    
                elif 40 <= CGPA * 10 < 50:        
                    return "Low"    
                elif 30 <= CGPA * 10 < 40:        
                    return "Bad"    
                elif CGPA * 10 < 30:        
                    return "Very Bad"    
                else:        
                    return "Undefined"
            elif 80 <= semester_marks_avg < 90:
                if CGPA * 10 >= 90:                       
                    return "Excellence"
                elif 80 <= CGPA * 10 < 90:
                    return "Good"
                elif 60 <= CGPA * 10 < 80:        
                    return "Better"
                elif 50 <= CGPA * 10 < 60:        
                    return "Average"    
                elif 40 <= CGPA * 10 < 50:        
                    return "Low"    
                elif 30 <= CGPA * 10 < 40:        
                    return "Bad"    
                elif CGPA * 10 < 30:        
                    return "Very Bad"    
                else:        
                    return "Undefined"        
                    
            elif 60 <= semester_marks_avg < 80:  
                if CGPA * 10 >= 90:                       
                    return "Better"
                elif 80 <= CGPA * 10 < 90:
                    return "Better"
                elif 60 <= CGPA * 10 < 80:        
                    return "Better"
                elif 50 <= CGPA * 10 < 60:        
                    return "Average"    
                elif 40 <= CGPA * 10 < 50:        
                    return "Low"    
                elif 30 <= CGPA * 10 < 40:        
                    return "Bad"    
                elif CGPA * 10 < 30:        
                    return "Very Bad"    
                else:        
                    return "Undefined"    
            elif 50 <= semester_marks_avg < 60: 
                if CGPA * 10 >= 90:                                           
                    return "Better"
                elif 80 <= CGPA * 10 < 90:
                    return "Bettr"
                elif 60 <= CGPA * 10 < 80:        
                    return "Better"
                elif 50 <= CGPA * 10 < 60:        
                    return "Average"    
                elif 40 <= CGPA * 10 < 50:        
                    return "Low"    
                elif 30 <= CGPA * 10 < 40:        
                    return "Bad"    
                elif CGPA * 10 < 30:        
                    return "Very Bad"    
                else:        
                    return "Undefined"
            elif 40 <= semester_marks_avg < 50:
                if CGPA * 10 >= 90:                       
                    return "Avrage"
                elif 80 <= CGPA * 10 < 90:
                    return "Average"
                elif 60 <= CGPA * 10 < 80:        
                    return "Average"
                elif 50 <= CGPA * 10 < 60:        
                    return "Average"    
                elif 40 <= CGPA * 10 < 50:        
                    return "Low"    
                elif 30 <= CGPA * 10 < 40:        
                    return "Bad"    
                elif CGPA * 10 < 30:        
                    return "Very Bad"    
                else:        
                    return "Undefined"    
            elif 30 <= semester_marks_avg < 40:
                if CGPA * 10 >= 90:                       
                    return "Low"
                elif 80 <= CGPA * 10 < 90:
                    return "Low"
                elif 60 <= CGPA * 10 < 80:        
                    return "Low"
                elif 50 <= CGPA * 10 < 60:        
                    return "Low"    
                elif 40 <= CGPA * 10 < 50:        
                    return "Low"    
                elif 30 <= CGPA * 10 < 40:        
                    return "Bad"    
                elif CGPA * 10 < 30:        
                    return "Very Bad"    
                else:        
                    return "Undefined"
                   
            elif semester_marks_avg < 30:
                if CGPA * 10 >= 90:                       
                    return "Low"
                elif 80 <= CGPA * 10 < 90:
                    return "Low"
                elif 60 <= CGPA * 10 < 80:        
                    return "Low"
                elif 50 <= CGPA * 10 < 60:        
                    return "Low"    
                elif 40 <= CGPA * 10 < 50:        
                    return "Low"    
                elif 30 <= CGPA * 10 < 40:        
                    return "Bad"    
                elif CGPA * 10 < 30:        
                    return "Very Bad"    
                else:        
                    return "Undefined"    
            else:        
                return "Undefined"  
        elif  40 <= attendance < 50:
            if semester_marks_avg >= 90:                
                if CGPA * 10 >= 90:                       
                    return "Excellence"
                elif 80 <= CGPA * 10 < 90:
                    return "Good"
                elif 60 <= CGPA * 10 < 80:        
                    return "Better"
                elif 50 <= CGPA * 10 < 60:        
                    return "Average"    
                elif 40 <= CGPA * 10 < 50:        
                    return "Low"    
                elif 30 <= CGPA * 10 < 40:        
                    return "Bad"    
                elif CGPA * 10 < 30:        
                    return "Very Bad"    
                else:        
                    return "Undefined"
            elif 80 <= semester_marks_avg < 90:
                if CGPA * 10 >= 90:                       
                    return "Excellence"
                elif 80 <= CGPA * 10 < 90:
                    return "Good"
                elif 60 <= CGPA * 10 < 80:        
                    return "Better"
                elif 50 <= CGPA * 10 < 60:        
                    return "Average"    
                elif 40 <= CGPA * 10 < 50:        
                    return "Low"    
                elif 30 <= CGPA * 10 < 40:        
                    return "Bad"    
                elif CGPA * 10 < 30:        
                    return "Very Bad"    
                else:        
                    return "Undefined"        
                    
            elif 60 <= semester_marks_avg < 80:  
                if CGPA * 10 >= 90:                       
                    return "Better"
                elif 80 <= CGPA * 10 < 90:
                    return "Better"
                elif 60 <= CGPA * 10 < 80:        
                    return "Better"
                elif 50 <= CGPA * 10 < 60:        
                    return "Average"    
                elif 40 <= CGPA * 10 < 50:        
                    return "Low"    
                elif 30 <= CGPA * 10 < 40:        
                    return "Bad"    
                elif CGPA * 10 < 30:        
                    return "Very Bad"    
                else:        
                    return "Undefined"    
            elif 50 <= semester_marks_avg < 60: 
                if CGPA * 10 >= 90:                                           
                    return "Better"
                elif 80 <= CGPA * 10 < 90:
                    return "Bettr"
                elif 60 <= CGPA * 10 < 80:        
                    return "Better"
                elif 50 <= CGPA * 10 < 60:        
                    return "Average"    
                elif 40 <= CGPA * 10 < 50:        
                    return "Low"    
                elif 30 <= CGPA * 10 < 40:        
                    return "Bad"    
                elif CGPA * 10 < 30:        
                    return "Very Bad"    
                else:        
                    return "Undefined"
            elif 40 <= semester_marks_avg < 50:
                if CGPA * 10 >= 90:                       
                    return "Avrage"
                elif 80 <= CGPA * 10 < 90:
                    return "Average"
                elif 60 <= CGPA * 10 < 80:        
                    return "Average"
                elif 50 <= CGPA * 10 < 60:        
                    return "Average"    
                elif 40 <= CGPA * 10 < 50:        
                    return "Low"    
                elif 30 <= CGPA * 10 < 40:        
                    return "Bad"    
                elif CGPA * 10 < 30:        
                    return "Very Bad"    
                else:        
                    return "Undefined"    
            elif 30 <= semester_marks_avg < 40:
                if CGPA * 10 >= 90:                       
                    return "Low"
                elif 80 <= CGPA * 10 < 90:
                    return "Low"
                elif 60 <= CGPA * 10 < 80:        
                    return "Low"
                elif 50 <= CGPA * 10 < 60:        
                    return "Low"    
                elif 40 <= CGPA * 10 < 50:        
                    return "Low"    
                elif 30 <= CGPA * 10 < 40:        
                    return "Bad"    
                elif CGPA * 10 < 30:        
                    return "Very Bad"    
                else:        
                    return "Undefined"
                   
            elif semester_marks_avg < 30:
                if CGPA * 10 >= 90:                       
                    return "Low"
                elif 80 <= CGPA * 10 < 90:
                    return "Low"
                elif 60 <= CGPA * 10 < 80:        
                    return "Low"
                elif 50 <= CGPA * 10 < 60:        
                    return "Low"    
                elif 40 <= CGPA * 10 < 50:        
                    return "Low"    
                elif 30 <= CGPA * 10 < 40:        
                    return "Bad"    
                elif CGPA * 10 < 30:        
                    return "Very Bad"    
                else:        
                    return "Undefined"    
            else:        
                return "Undefined"
        elif  30 <= attendance < 40: 
            if semester_marks_avg >= 90:                
                if CGPA * 10 >= 90:                       
                    return "Excellence"
                elif 80 <= CGPA * 10 < 90:
                    return "Good"
                elif 60 <= CGPA * 10 < 80:        
                    return "Better"
                elif 50 <= CGPA * 10 < 60:        
                    return "Average"    
                elif 40 <= CGPA * 10 < 50:        
                    return "Low"    
                elif 30 <= CGPA * 10 < 40:        
                    return "Bad"    
                elif CGPA * 10 < 30:        
                    return "Very Bad"    
                else:        
                    return "Undefined"
            elif 80 <= semester_marks_avg < 90:
                if CGPA * 10 >= 90:                       
                    return "Excellence"
                elif 80 <= CGPA * 10 < 90:
                    return "Good"
                elif 60 <= CGPA * 10 < 80:        
                    return "Better"
                elif 50 <= CGPA * 10 < 60:        
                    return "Average"    
                elif 40 <= CGPA * 10 < 50:        
                    return "Low"    
                elif 30 <= CGPA * 10 < 40:        
                    return "Bad"    
                elif CGPA * 10 < 30:        
                    return "Very Bad"    
                else:        
                    return "Undefined"        
                    
            elif 60 <= semester_marks_avg < 80:  
                if CGPA * 10 >= 90:                       
                    return "Better"
                elif 80 <= CGPA * 10 < 90:
                    return "Better"
                elif 60 <= CGPA * 10 < 80:        
                    return "Better"
                elif 50 <= CGPA * 10 < 60:        
                    return "Average"    
                elif 40 <= CGPA * 10 < 50:        
                    return "Low"    
                elif 30 <= CGPA * 10 < 40:        
                    return "Bad"    
                elif CGPA * 10 < 30:        
                    return "Very Bad"    
                else:        
                    return "Undefined"    
            elif 50 <= semester_marks_avg < 60: 
                if CGPA * 10 >= 90:                                           
                    return "Better"
                elif 80 <= CGPA * 10 < 90:
                    return "Bettr"
                elif 60 <= CGPA * 10 < 80:        
                    return "Better"
                elif 50 <= CGPA * 10 < 60:        
                    return "Average"    
                elif 40 <= CGPA * 10 < 50:        
                    return "Low"    
                elif 30 <= CGPA * 10 < 40:        
                    return "Bad"    
                elif CGPA * 10 < 30:        
                    return "Very Bad"    
                else:        
                    return "Undefined"
            elif 40 <= semester_marks_avg < 50:
                if CGPA * 10 >= 90:                       
                    return "Avrage"
                elif 80 <= CGPA * 10 < 90:
                    return "Average"
                elif 60 <= CGPA * 10 < 80:        
                    return "Average"
                elif 50 <= CGPA * 10 < 60:        
                    return "Average"    
                elif 40 <= CGPA * 10 < 50:        
                    return "Low"    
                elif 30 <= CGPA * 10 < 40:        
                    return "Bad"    
                elif CGPA * 10 < 30:        
                    return "Very Bad"    
                else:        
                    return "Undefined"    
            elif 30 <= semester_marks_avg < 40:
                if CGPA * 10 >= 90:                       
                    return "Low"
                elif 80 <= CGPA * 10 < 90:
                    return "Low"
                elif 60 <= CGPA * 10 < 80:        
                    return "Low"
                elif 50 <= CGPA * 10 < 60:        
                    return "Low"    
                elif 40 <= CGPA * 10 < 50:        
                    return "Low"    
                elif 30 <= CGPA * 10 < 40:        
                    return "Bad"    
                elif CGPA * 10 < 30:        
                    return "Very Bad"    
                else:        
                    return "Undefined"
                   
            elif semester_marks_avg < 30:
                if CGPA * 10 >= 90:                       
                    return "Low"
                elif 80 <= CGPA * 10 < 90:
                    return "Low"
                elif 60 <= CGPA * 10 < 80:        
                    return "Low"
                elif 50 <= CGPA * 10 < 60:        
                    return "Low"    
                elif 40 <= CGPA * 10 < 50:        
                    return "Low"    
                elif 30 <= CGPA * 10 < 40:        
                    return "Bad"    
                elif CGPA * 10 < 30:        
                    return "Very Bad"    
                else:        
                    return "Undefined"    
            else:        
                return "Undefined"
        elif attendance < 30:
            if semester_marks_avg >= 90:                
                if CGPA * 10 >= 90:                       
                    return "Excellence"
                elif 80 <= CGPA * 10 < 90:
                    return "Good"
                elif 60 <= CGPA * 10 < 80:        
                    return "Better"
                elif 50 <= CGPA * 10 < 60:        
                    return "Average"    
                elif 40 <= CGPA * 10 < 50:        
                    return "Low"    
                elif 30 <= CGPA * 10 < 40:        
                    return "Bad"    
                elif CGPA * 10 < 30:        
                    return "Very Bad"    
                else:        
                    return "Undefined"
            elif 80 <= semester_marks_avg < 90:
                if CGPA * 10 >= 90:                       
                    return "Excellence"
                elif 80 <= CGPA * 10 < 90:
                    return "Good"
                elif 60 <= CGPA * 10 < 80:        
                    return "Better"
                elif 50 <= CGPA * 10 < 60:        
                    return "Average"    
                elif 40 <= CGPA * 10 < 50:        
                    return "Low"    
                elif 30 <= CGPA * 10 < 40:        
                    return "Bad"    
                elif CGPA * 10 < 30:        
                    return "Very Bad"    
                else:        
                    return "Undefined"        
                    
            elif 60 <= semester_marks_avg < 80:  
                if CGPA * 10 >= 90:                       
                    return "Better"
                elif 80 <= CGPA * 10 < 90:
                    return "Better"
                elif 60 <= CGPA * 10 < 80:        
                    return "Better"
                elif 50 <= CGPA * 10 < 60:        
                    return "Average"    
                elif 40 <= CGPA * 10 < 50:        
                    return "Low"    
                elif 30 <= CGPA * 10 < 40:        
                    return "Bad"    
                elif CGPA * 10 < 30:        
                    return "Very Bad"    
                else:        
                    return "Undefined"    
            elif 50 <= semester_marks_avg < 60: 
                if CGPA * 10 >= 90:                                           
                    return "Better"
                elif 80 <= CGPA * 10 < 90:
                    return "Bettr"
                elif 60 <= CGPA * 10 < 80:        
                    return "Better"
                elif 50 <= CGPA * 10 < 60:        
                    return "Average"    
                elif 40 <= CGPA * 10 < 50:        
                    return "Low"    
                elif 30 <= CGPA * 10 < 40:        
                    return "Bad"    
                elif CGPA * 10 < 30:        
                    return "Very Bad"    
                else:        
                    return "Undefined"
            elif 40 <= semester_marks_avg < 50:
                if CGPA * 10 >= 90:                       
                    return "Avrage"
                elif 80 <= CGPA * 10 < 90:
                    return "Average"
                elif 60 <= CGPA * 10 < 80:        
                    return "Average"
                elif 50 <= CGPA * 10 < 60:        
                    return "Average"    
                elif 40 <= CGPA * 10 < 50:        
                    return "Low"    
                elif 30 <= CGPA * 10 < 40:        
                    return "Bad"    
                elif CGPA * 10 < 30:        
                    return "Very Bad"    
                else:        
                    return "Undefined"    
            elif 30 <= semester_marks_avg < 40:
                if CGPA * 10 >= 90:                       
                    return "Low"
                elif 80 <= CGPA * 10 < 90:
                    return "Low"
                elif 60 <= CGPA * 10 < 80:        
                    return "Low"
                elif 50 <= CGPA * 10 < 60:        
                    return "Low"    
                elif 40 <= CGPA * 10 < 50:        
                    return "Low"    
                elif 30 <= CGPA * 10 < 40:        
                    return "Bad"    
                elif CGPA * 10 < 30:        
                    return "Very Bad"    
                else:        
                    return "Undefined"
                   
            elif semester_marks_avg < 30:
                if CGPA * 10 >= 90:                       
                    return "Low"
                elif 80 <= CGPA * 10 < 90:
                    return "Low"
                elif 60 <= CGPA * 10 < 80:        
                    return "Low"
                elif 50 <= CGPA * 10 < 60:        
                    return "Low"    
                elif 40 <= CGPA * 10 < 50:        
                    return "Low"    
                elif 30 <= CGPA * 10 < 40:        
                    return "Bad"    
                elif CGPA * 10 < 30:        
                    return "Very Bad"    
                else:        
                    return "Undefined"    
            else:        
                return "Undefined"
        else:
            return "Incrase the Study Time "


                
    else:
        return "Undefined"

# Define description messages based on predicted behavior category
behavior_descriptions = {
    "Excellence": "This student exhibits excellent behavior. They consistently achieve high grades, maintain excellent attendance, and devote significant time to studying.",
    "Good": "This student demonstrates good behavior. They perform well academically and maintain satisfactory attendance and study habits.",
    "Better": "This student's behavior is better than average. They show improvement in their academic performance and maintain decent attendance and study habits.",
    "Average": "This student's behavior is average. They achieve moderate grades, attend classes regularly, and spend a reasonable amount of time studying.",
    "Low": "This student's behavior is below average. They struggle with academic performance, attendance, and study habits.",
    "Bad": "This student's behavior is concerning. They consistently perform poorly academically, have low attendance, and spend minimal time studying.",
    "Very Bad": "This student's behavior is very poor. They have significant academic challenges, extremely low attendance, and minimal study habits.",
    "Undefined": "The behavior of this student cannot be categorized based on available data."
}

# Define messages based on the relation between students and teachers
teacher_relation_messages = {
    "Excellent": "This student has an excellent relationship with their teachers. They actively engage with teachers and seek guidance.",
    "Good": "This student has a good relationship with their teachers. They participate in classroom activities and interact positively with teachers.",
    "Average": "This student has an average relationship with their teachers. They maintain a neutral stance in classroom interactions.",
    "Poor": "This student has a poor relationship with their teachers. They may show disinterest or lack of engagement in classroom activities.",
}

# Implement machine learning models
features = ['CGPA', 'First_Semester_marks', 'Second_Semester_marks', 'Third_Semester_marks', 
            'Attendance_Percentage', 'Study_time_at_home']
X = df[['CGPA', 'First_Semester_marks', 'Second_Semester_marks', 'Third_Semester_marks', 
            'Attendance_Percentage', 'Study_time_at_home']]
y = df['Failed_in_subject/course_before']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[['CGPA', 'Attendance_Percentage', 'First_Semester_marks', 'Second_Semester_marks', 'Third_Semester_marks', 'Study_time_at_home']])
X_test_scaled = scaler.transform(X_test[['CGPA', 'Attendance_Percentage', 'First_Semester_marks', 'Second_Semester_marks', 'Third_Semester_marks', 'Study_time_at_home']])

models = {
    "KNN": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)  # Use both X_train_scaled and y_train for classification models

# Use SVM to select the most accurate category
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_scaled, y_train)  # Use both X_train_scaled and y_train for SVM

# Add new student entry section in main area
if action == "Add New Student Entry":
    st.title("Add New Student Entry")

    # Input fields for new student entry
    new_student_name = st.text_input("Student Name")
    new_gender = st.radio("Gender", ["Male", "Female"])
    new_cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, step=0.01)
    new_attendance_percentage = st.number_input("Attendance Percentage", min_value=0, max_value=100, step=1)
    new_parent_education_level = st.selectbox("Parent/Guardian Education Level", ["Primary", "Secondary", "Higher Secondary", "Graduate", "Post Graduate"])
    new_travel_time_to_college = st.number_input("Travel Time to College (minutes)", min_value=0, step=1)
    new_relation_between_teachers = st.radio("Relation Between Teachers", ["Excellent", "Good", "Average", "Poor"])
    new_first_semester_marks = st.number_input("First Semester Marks", min_value=0, step=1)
    new_second_semester_marks = st.number_input("Second Semester Marks", min_value=0, step=1)
    new_third_semester_marks = st.number_input("Third Semester Marks", min_value=0, step=1)
    new_failed_in_subject_before = st.radio("Failed in Subject/Course Before", ["Yes", "No"])
    new_average_weekly_attendance = st.number_input("Average Weekly Attendance", min_value=0, step=1)
    new_study_time_at_home = st.number_input("Study Time at Home (hours)", min_value=0.0, step=0.5)
    new_enjoy_free_time_with = st.radio("Enjoy Free Time With", ["Friends", "Family", "Alone"])

    # Button to add the new student entry
    if st.button("Add New Student"):
        # Create a DataFrame with the new student entry data
        new_entry = pd.DataFrame({
            "Student_Name": [new_student_name],
            "Gender": [new_gender],
            "CGPA": [new_cgpa],
            "Attendance_Percentage": [new_attendance_percentage],
            "Parent/Guardian_Education_level": [new_parent_education_level],
            "Travel_time_to_college": [new_travel_time_to_college],
            "Relation_between_teachers": [new_relation_between_teachers],
            "First_Semester_marks": [new_first_semester_marks],
            "Second_Semester_marks": [new_second_semester_marks],
            "Third_Semester_marks": [new_third_semester_marks],
            "Failed_in_subject/course_before": [new_failed_in_subject_before],
            "Average_weekly_attendance": [new_average_weekly_attendance],
            "Study_time_at_home": [new_study_time_at_home],
            "Enjoy_free_time_with": [new_enjoy_free_time_with]
        })
        
        # Concatenate the new DataFrame with the existing DataFrame and save it to the CSV file
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv("students_data.csv", index=False)
        st.success("New student entry added successfully.")

# Survey section in main area
elif action == "Survey":
    st.title("Survey Section")
    st.write("Select a column to visualize its data:")
    column_name = st.selectbox("Select Column", df.columns.tolist())
    if column_name:
        st.subheader(f"Visualization of {column_name}")
        if df[column_name].dtype == "object":
            st.write("### Value Counts Bar Chart")
            st.bar_chart(df[column_name].value_counts())
            st.write("### Pie Chart")
            fig, ax = plt.subplots()
            ax.pie(df[column_name].value_counts(), labels=df[column_name].value_counts().index, autopct='%1.1f%%', startangle=140)
            ax.axis('equal')
            st.pyplot(fig)
            st.write("### Count Plot")
            sns.countplot(data=df, x=column_name)
            st.pyplot()
        elif df[column_name].dtype in ["int64", "float64"]:
            st.write("### Histogram")
            plt.hist(df[column_name], bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
            plt.xlabel(column_name)
            plt.ylabel('Frequency')
            plt.title(f'Histogram of {column_name}')
            st.pyplot()
            st.write("### Box Plot")
            sns.boxplot(data=df, x=column_name)
            st.pyplot()
        else:
            st.write("Selected column type not supported for visualization.")

else:
    st.title("Search Student")
    student_name = st.text_input("Enter Student Name:")
    if student_name:
        student_data = df[df['Student_Name'] == student_name]
        if not student_data.empty:
            CGPA = student_data['CGPA'].values[0]
            semester_marks_avg = (student_data['First_Semester_marks'].values[0] + 
                                  student_data['Second_Semester_marks'].values[0] + 
                                  student_data['Third_Semester_marks'].values[0]) / 3
            attendance = student_data['Attendance_Percentage'].values[0]
            study_time = student_data['Study_time_at_home'].values[0]
            relation_with_teachers = student_data['Relation_between_teachers'].values[0]

            behavior_category = get_category(CGPA, semester_marks_avg, attendance, study_time)
            svm_prediction = svm_model.predict(scaler.transform([[CGPA, attendance, 
                                                                  student_data['First_Semester_marks'].values[0], 
                                                                  student_data['Second_Semester_marks'].values[0], 
                                                                  student_data['Third_Semester_marks'].values[0], 
                                                                  study_time]]))[0]

            behavior_description = behavior_descriptions.get(behavior_category, "Description not available")
            teacher_relation_message = teacher_relation_messages.get(relation_with_teachers, "Teacher relation message not available")

            st.write(f"Predicted Behavior Category for {student_name}: {behavior_category}")
            st.write(behavior_description)
            st.write(teacher_relation_message)

            st.subheader("Information of Selected Student")
            st.write(student_data)  # Display all information of the selected student

            st.subheader("Graphical Representation of Student Information")
            # Plotting student information
            student_data_numeric = student_data.drop(columns=['Student_Name', 'Gender', 'Parent/Guardian_Education_level', 'Relation_between_teachers', 'Failed_in_subject/course_before', 'Enjoy_free_time_with'])
            student_data_numeric = student_data_numeric.transpose()  # Transpose to have features on x-axis
            fig, ax = plt.subplots(figsize=(10, 6))
            student_data_numeric.plot(kind='bar', legend=None, ax=ax, cmap='tab20')
            plt.title(f"Student Information for {student_name}")
            plt.xlabel("Features")
            plt.ylabel("Values")
            plt.xticks(rotation=45, ha='right')
            for i in range(len(student_data_numeric)): 
                plt.text(x=i, y=student_data_numeric.iloc[i, 0], s=str(round(student_data_numeric.iloc[i, 0], 2)), ha='center')
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.write("Student not found in the database.")
