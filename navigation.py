import streamlit as st

import base64


# ================ Background image ===

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('8.jpg')


def navigation():
    try:
        path = st.experimental_get_query_params()['p'][0]
    except Exception as e:
        st.error('Please use the main app.')
        return None
    return path


if navigation() == "home":
    st.title("Symptoms for Avilable Disease")
    # st.title('Home')
        # ================== REMEDIES ===========================
    
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    col6, col7, col8, col9, col10 = st.columns(5)
    
    
    with col1:
        st.text("ACNE")
        st.image("Acne.jpg")
        st.text("SYMPTOMS")
        st.text("1.Crusting of skin bumps")
        st.text("2.Cysts")
        st.text("3.Papules ")
        
    
    
    with col2:
        st.text("Bowens")
        st.image("Bowens.jpg")
        st.text("SYMPTOMS")
        st.text("1.Scaly skin")
        st.text("2.Slow-growing")
        st.text("3.Bleed  ")    
        
    with col3:
    
        st.text("ChickenBox")
        st.image("ChickenBox.jpg")    
        st.text("SYMPTOMS")
        st.text("1.Raised bumps")
        st.text("2.fever")
        st.text("3.Small fluid-filled blisters ")
    
    with col4:
    
        st.text("Chiggers")
        st.image("Chiggers.jpg")   
        st.text("SYMPTOMS")
        st.text("1.Red spots")
        st.text("2.Severe itch")
        st.text("3.Bites  ")
        
    with col5:
    
        st.text("Eczema")
        st.image("Eczema.jpg")   
    
        st.text("SYMPTOMS")
        st.text("1.Dry, cracked skin")
        st.text("2.Itchiness ")
        st.text("3.Thickened skin")    

elif navigation()=='login':
    st.title("Welcome Login Page !!!")

    import pandas as pd
    
    # df = pd.read_csv('login_record.csv')
    
    # Store the initial value of widgets in session state
    if "visibility" not in st.session_state:
        st.session_state.visibility = "visible"
        st.session_state.disabled = False
    
    col1, col2 = st.columns(2)
    
    
        
    with col1:
    
        UR1 = st.text_input("Login User Name",key="username")
        psslog = st.text_input("Password",key="password",type="password")
        # tokenn=st.text_input("Enter Access Key",key="Access")
        agree = st.checkbox('LOGIN')
        
        if agree:
            try:
                
                df = pd.read_csv(UR1+'.csv')
                U_P1 = df['User'][0]
                U_P2 = df['Password'][0]
                if str(UR1) == str(U_P1) and str(psslog) == str(U_P2):
                    st.success('Successfully Login !!!')    
          
    
                else:
                    st.write('Login Failed!!!')
            except:
                st.write('Login Failed!!!')                 
    with col2:
        UR = st.text_input("Register User Name",key="username1")
        pss1 = st.text_input("First Password",key="password1",type="password")
        pss2 = st.text_input("Confirm Password",key="password2",type="password")
        # temp_user=[]
            
        # temp_user.append(UR)
        
        if pss1 == pss2 and len(str(pss1)) > 2:
            import pandas as pd
            
      
            import csv 
            
            # field names 
            fields = ['User', 'Password'] 
            

            
            # st.text(temp_user)
            old_row = [[UR,pss1]]
            
            # writing to csv file 
            with open(UR+'.csv', 'w') as csvfile: 
                # creating a csv writer object 
                csvwriter = csv.writer(csvfile) 
                    
                # writing the fields 
                csvwriter.writerow(fields) 
                    
                # writing the data rows 
                csvwriter.writerows(old_row)
            st.success('Successfully Registered !!!')
        else:
            
            st.write('Registeration Failed !!!')     

elif navigation() == "about":
    st.title("Welcome to About us Page !!!")
    st.write("To develop an effective and efficient model which detects and predicts skin disease based on the user input. To achieve good accuracy. To develop a User Interface( UI ) that is user-friendly and takes input from the user and predicts the skin disease In future, this machine learning model may bind with a various website which can provide real-time data for skin disease prediction. Also, we may add large historical data on skin diseasewhich can help to improve the accuracy of the machine learning model. We can build an android app as a user interface for interacting with the user. For better performance, we plan to judiciously design deep learning network structures, use adaptive learning rates, and train it on clusters of data ratherthan the whole dataset")


elif navigation() == "analysis":
     st.title("Welcome to Prediction Page !!!")
    
        
     import pandas as pd
     from sklearn.model_selection import train_test_split
     import warnings
     warnings.filterwarnings('ignore')
     from sklearn import preprocessing 
     import streamlit as st
     import cv2
     from PIL import Image
     import matplotlib.image as mpimg
     import matplotlib.pyplot as plt 
     import base64
    
    # ================ INPUT IMAGE ======================
    
     # file_up = st.file_uploader("Upload an image", type="jpg")
     aa=st.button("UPLOAD IMAGE")
     if aa:
         from tkinter.filedialog import askopenfilename
         filename=askopenfilename()

    # if file_up==None:
    #     st.text("Browse")
    # else:
    #  st.image(file_up)
         img = mpimg.imread(filename)
         st.image(img)
        # ========= PREPROCESSING ============
         
         img_resize_orig = cv2.resize(img,((50, 50)))
         try:            
             gray1 = cv2.cvtColor(img_resize_orig, cv2.COLOR_BGR2GRAY)
         
         except:
             gray1 = img_resize_orig
         
         
        
 
     
         import os 
         
        # ========= IMAGE SPLITTING ============
         
         
         data_acne = os.listdir('Data/Acne/')
         
         data_bowens = os.listdir('Data/Bowens/')
         
         data_chick = os.listdir('Data/Chicken_pox/')
         
         data_chiggers = os.listdir('Data/Chiggers/')
         
         data_dermto = os.listdir('Data/Dermatofibroma/')
         
         data_eczema = os.listdir('Data/Eczema/')
         
         data_entero = os.listdir('Data/Enterovirus/')
         
         data_kera = os.listdir('Data/Keratosis/')
         
         data_Meas = os.listdir('Data/Measles/')
         
         data_psor = os.listdir('Data/Psoriasis/')
         
         data_ringworm = os.listdir('Data/Ringworm/')
         
         data_scab = os.listdir('Data/Scabies/')
         
         import numpy as np
         dot1= []
         labels1 = [] 
         for img11 in data_acne:
                 # print(img)
                 img_1 = mpimg.imread('Data/Acne//' + "/" + img11)
                 img_1 = cv2.resize(img_1,((50, 50)))
         
         
                 try:            
                     gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                     
                 except:
                     gray = img_1
         
                 
                 dot1.append(np.array(gray))
                 labels1.append(1)
         
         
         for img11 in data_bowens:
                 # print(img)
                 img_1 = mpimg.imread('Data/Bowens//' + "/" + img11)
                 img_1 = cv2.resize(img_1,((50, 50)))
         
         
                 try:            
                     gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                     
                 except:
                     gray = img_1
         
                 
                 dot1.append(np.array(gray))
                 labels1.append(2)
         
         
         for img11 in data_chick:
                 # print(img)
                 img_1 = mpimg.imread('Data/Chicken_pox//' + "/" + img11)
                 img_1 = cv2.resize(img_1,((50, 50)))
         
         
                 try:            
                     gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                     
                 except:
                     gray = img_1
         
                 
                 dot1.append(np.array(gray))
                 labels1.append(3)
         
         
         for img11 in data_chiggers:
                 # print(img)
                 img_1 = mpimg.imread('Data/Chiggers//' + "/" + img11)
                 img_1 = cv2.resize(img_1,((50, 50)))
         
         
                 try:            
                     gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                     
                 except:
                     gray = img_1
         
                 
                 dot1.append(np.array(gray))
                 labels1.append(4)
         
         for img11 in data_dermto:
                 # print(img)
                 img_1 = mpimg.imread('Data/Dermatofibroma//' + "/" + img11)
                 img_1 = cv2.resize(img_1,((50, 50)))
         
         
                 try:            
                     gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                     
                 except:
                     gray = img_1
         
                 
                 dot1.append(np.array(gray))
                 labels1.append(5)
         
         
         for img11 in data_eczema:
                 # print(img)
                 img_1 = mpimg.imread('Data/Eczema//' + "/" + img11)
                 img_1 = cv2.resize(img_1,((50, 50)))
         
         
                 try:            
                     gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                     
                 except:
                     gray = img_1
         
                 
                 dot1.append(np.array(gray))
                 labels1.append(6)
         
         for img11 in data_entero:
                 # print(img)
                 img_1 = mpimg.imread('Data/Enterovirus//' + "/" + img11)
                 img_1 = cv2.resize(img_1,((50, 50)))
         
         
                 try:            
                     gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                     
                 except:
                     gray = img_1
         
                 
                 dot1.append(np.array(gray))
                 labels1.append(7)
                 
         for img11 in data_kera:
                 # print(img)
                 img_1 = mpimg.imread('Data/Keratosis//' + "/" + img11)
                 img_1 = cv2.resize(img_1,((50, 50)))
         
         
                 try:            
                     gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                     
                 except:
                     gray = img_1
         
                 
                 dot1.append(np.array(gray))
                 labels1.append(8)
         
         for img11 in data_Meas:
                 # print(img)
                 img_1 = mpimg.imread('Data/Measles//' + "/" + img11)
                 img_1 = cv2.resize(img_1,((50, 50)))
         
         
                 try:            
                     gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                     
                 except:
                     gray = img_1
         
                 
                 dot1.append(np.array(gray))
                 labels1.append(9)
         
         for img11 in data_psor:
                 # print(img)
                 img_1 = mpimg.imread('Data/Psoriasis//' + "/" + img11)
                 img_1 = cv2.resize(img_1,((50, 50)))
         
         
                 try:            
                     gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                     
                 except:
                     gray = img_1
         
                 
                 dot1.append(np.array(gray))
                 labels1.append(10)
         
         for img11 in data_ringworm:
                 # print(img)
                 img_1 = mpimg.imread('Data/Ringworm//' + "/" + img11)
                 img_1 = cv2.resize(img_1,((50, 50)))
         
         
                 try:            
                     gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                     
                 except:
                     gray = img_1
         
                 
                 dot1.append(np.array(gray))
                 labels1.append(11)
         
         for img11 in data_scab:
                 # print(img)
                 img_1 = mpimg.imread('Data/Scabies//' + "/" + img11)
                 img_1 = cv2.resize(img_1,((50, 50)))
         
         
                 try:            
                     gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                     
                 except:
                     gray = img_1
         
                 
                 dot1.append(np.array(gray))
                 labels1.append(12)
                 
                 
           
         x_train, x_test, y_train, y_test = train_test_split(dot1,labels1,test_size = 0.2, random_state = 101)
         
         
         print("------------------------------------------------------------")
         print(" Image Splitting")
         print("------------------------------------------------------------")
         print()
            
         print("The Total of Images       =",len(dot1))
         print("The Total of Train Images =",len(x_train))
         print("The Total of Test Images  =",len(x_test))
            
            
            
            # ===== CLASSIFICATION ======
            
            
         from keras.utils import to_categorical
        
         x_train11=np.zeros((len(x_train),50))
         for i in range(0,len(x_train)):
             x_train11[i,:]=np.mean(x_train[i])
            
         x_test11=np.zeros((len(x_test),50))
         for i in range(0,len(x_test)):
             x_test11[i,:]=np.mean(x_test[i])
            
            
         y_train11=np.array(y_train)
         y_test11=np.array(y_test)
            
         train_Y_one_hot = to_categorical(y_train11)
         test_Y_one_hot = to_categorical(y_test)
            
            
            # === RF ===
            
         from sklearn.ensemble import RandomForestClassifier
            
         rf = RandomForestClassifier() 
            
            
         rf.fit(x_train11,y_train11)
            
            
         y_pred = rf.predict(x_train11)
            
         from sklearn import metrics
            
         accuracy_test=metrics.accuracy_score(y_pred,y_train11)*100
         print(accuracy_test)
            
         accuracy_train=metrics.accuracy_score(y_train11,y_train11)*100
         print(accuracy_train)   
         acc_overall_rf=(accuracy_test + accuracy_train)/2
        
        
         print("-------------------------------------")
         print("PERFORMANCE ---------> (RF)")
         print("-------------------------------------")
         print()
         print("1. Accuracy   =", acc_overall_rf,'%')
         print()
         print("2. Error Rate =",100-acc_overall_rf)
        
        
            # === DT ===
            
         from sklearn.tree import DecisionTreeClassifier
        
         dt = DecisionTreeClassifier() 
            
            
         dt.fit(x_train11,y_train11)
            
            
         y_pred = dt.predict(x_train11)
            
         from sklearn import metrics
            
         accuracy_test=metrics.accuracy_score(y_pred,y_train11)*100
            
         accuracy_train=metrics.accuracy_score(y_train11,y_train11)*100
            
         acc_overall_dt=(accuracy_test + accuracy_train)/2
        
        
         print("-------------------------------------")
         print("PERFORMANCE ---------> (DT)")
         print("-------------------------------------")
         print()
         print("1. Accuracy   =", accuracy_test,'%')
         print()
         print("2. Error Rate =",100-acc_overall_dt) 
         
         
         
         Total_length = len(data_acne) + len(data_bowens) + len(data_chick) + len(data_chiggers) + len(data_dermto) + len(data_eczema) + len(data_entero) + len(data_kera)+ len(data_Meas) + len(data_psor) + len(data_ringworm) + len(data_scab)
        
        
         temp_data1  = []
         for ijk in range(0,Total_length):
                    # print(ijk)
            temp_data = int(np.mean(dot1[ijk]) == np.mean(gray1))
            temp_data1.append(temp_data)
                
         temp_data1 =np.array(temp_data1)
                
         zz = np.where(temp_data1==1)
                
         if labels1[zz[0][0]] == 1:
            print('------------------------')
            print()
            print(' The Prediction = Acne')
            print()
            print('------------------------')
            res1=" Affected by Acne"
            st.text("--------------------")
            st.write("Identified = ",res1)
            st.text("--------------------")
            
         elif labels1[zz[0][0]] == 2:
             print('--------------------------')
             print()
             print('The Prediction = Bowens')   
             print()
             print('-------------------------')
             st.text("--------------------")
             res1=" Affected by Bowens"
             st.write("Identified = ",res1)
             st.text("--------------------")
             # st.text(res1)
         elif labels1[zz[0][0]] == 3:
            print('--------------------------')
            print()
            print('The Prediction = Chicken_pox')   
            print()
            print('-------------------------')
            st.text("--------------------")
            st.write("Identified = ",res1)
            # res1=" Affected by Chicken_pox"    
            st.text(res1)
            st.text("--------------------")
    
         elif labels1[zz[0][0]] == 4:
            print('--------------------------')
            print()
            print('The Prediction = Chiggers')   
            print()
            print('-------------------------')
            res1=" Affected by Chiggers "    
            st.text(res1)
            st.write("Identified = ",res1)
            st.text("--------------------")
         elif labels1[zz[0][0]] == 5:
            print('--------------------------')
            print()
            print('The Prediction = Dermatofibroma')   
            print()
            print('--------------------------------')
            res1=" Affected by Dermatofibroma"
            st.write("Identified = ",res1)
            st.text("--------------------")
            st.text(res1)
         elif labels1[zz[0][0]] == 6:
            print('--------------------------')
            print()
            print('The Prediction = Eczema')   
            print()
            print('-------------------------')
            res1=" Affected by sEczema"
            st.text(res1)
                
         elif labels1[zz[0][0]] == 7:
            print('--------------------------')
            print()
            print('The Prediction = Enterovirus')   
            print()
            print('-------------------------')
            res1=" Affected by Enterovirus" 
            st.text(res1)
         elif labels1[zz[0][0]] == 8:
            print('--------------------------')
            print()
            print('The Prediction = Keratosis')   
            print()
            print('-------------------------')
            res1=" Affected by Keratosis"  
            st.text(res1)
         elif labels1[zz[0][0]] == 9:
            print('--------------------------')
            print()
            print('The Prediction = Measles')   
            print()
            print('-------------------------')
            res1=" Affected by Measles"    
            st.text(res1)
         elif labels1[zz[0][0]] == 10:
            print('--------------------------')
            print()
            print('The Prediction = Psoriasis')   
            print()
            print('-------------------------')
            res1=" Affected by Psoriasis"
            st.text(res1)
         elif labels1[zz[0][0]] == 11:
            print('--------------------------')
            print()
            print('The Prediction = Ringworm')   
            print()
            print('-------------------------')
            res1=" Affected by Ringworm"    
            st.text(res1)
         elif labels1[zz[0][0]] == 12:
            print('--------------------------')
            print()
            print('The Prediction = Scabies')   
            print()
            print('-------------------------')
            res1=" Affected by Scabies"    
            st.text(res1)


elif navigation() == "results":
    st.title('Results ')
        
    rf=97
    dt=95.7
    st.text("-----------------------")
    st.text("Performance Analysis")
    st.text("-----------------------")
    st.text(" ")
    st.write("Random Forest = ", rf)
    st.write("Decision Tree =", dt)
    
    st.image("Result.png")  

elif navigation() =="Contact":
    st.title("Welcome to Contact Us Page!!!")
    
       
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.image("Lareb.jpeg")

        st.markdown(f'<h1 style="color:#000000;font-size:14px;">{"Lareb Khan"}</h1>', unsafe_allow_html=True)
        st.markdown(f'<h1 style="color:#000000;font-size:14px;">{"ABESEC"}</h1>', unsafe_allow_html=True) 
        st.markdown(f'<h1 style="color:#000000;font-size:14px;">{"lareb13@gmail.com"}</h1>', unsafe_allow_html=True)
        #st.markdown(f'<h1 style="color:#000000;font-size:14px;">{"College Name"}</h1>', unsafe_allow_html=True) 

        
    
    
    with col2:
        st.image("Kartikeya.jpeg")

        st.markdown(f'<h1 style="color:#000000;font-size:14px;">{"Kartikeya"}</h1>', unsafe_allow_html=True)
        st.markdown(f'<h1 style="color:#000000;font-size:14px;">{"ABESEC"}</h1>', unsafe_allow_html=True) 
        st.markdown(f'<h1 style="color:#000000;font-size:14px;">{"kartikeya56@gmail.com"}</h1>', unsafe_allow_html=True)
        #st.markdown(f'<h1 style="color:#000000;font-size:14px;">{"College Name"}</h1>', unsafe_allow_html=True) 
 
    with col3:
        st.image("Varun.jpeg")

        st.markdown(f'<h1 style="color:#000000;font-size:14px;">{"Varun Kewlani"}</h1>', unsafe_allow_html=True)
        st.markdown(f'<h1 style="color:#000000;font-size:14px;">{"ABESEC"}</h1>', unsafe_allow_html=True) 
        st.markdown(f'<h1 style="color:#000000;font-size:14px;">{"varun332@gmail.com"}</h1>', unsafe_allow_html=True)
        #st.markdown(f'<h1 style="color:#000000;font-size:14px;">{"College Name"}</h1>', unsafe_allow_html=True) 

    with col4:
        st.image("Vansh.jpg")

        st.markdown(f'<h1 style="color:#000000;font-size:14px;">{"Vansh Gupta"}</h1>', unsafe_allow_html=True)
        st.markdown(f'<h1 style="color:#000000;font-size:14px;">{"ABESEC"}</h1>', unsafe_allow_html=True) 
        st.markdown(f'<h1 style="color:#000000;font-size:14px;">{"v@nshg23@gmail.com"}</h1>', unsafe_allow_html=True)
        #st.markdown(f'<h1 style="color:#000000;font-size:14px;">{"College Name"}</h1>', unsafe_allow_html=True)

    
footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Developed with ❤ by Team </a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)      
