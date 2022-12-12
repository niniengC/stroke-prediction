import streamlit as st
st.set_page_config(
    page_title="Stroke Prediction"
)
st.set_option('deprecation.showPyplotGlobalUse', False)
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score ,precision_score,recall_score,f1_score, classification_report

#Load Dataset
def load_dataset():
   dataset="https://raw.githubusercontent.com/niniengC/dataset/main/healthcare-dataset-stroke-data.csv"
   df = pd.read_csv(dataset,  header='infer', index_col=False)
   return df
    
def normalisasi(df_new,nama_normalisasi):
    num_cols=['age', 'bmi', 'avg_glucose_level']
    if nama_normalisasi=="Standard Scaler":
        jenis_normalisasi=StandardScaler()
        df_new[num_cols] = jenis_normalisasi.fit_transform(df_new[num_cols])
    elif nama_normalisasi=="MinMax Scaler":
        jenis_normalisasi=MinMaxScaler()
        df_new[num_cols] = jenis_normalisasi.fit_transform(df_new[num_cols])
    return df_new

def tambah_parameter(nama_model):
    params=dict()
    if nama_model=='K-Nearest Neighbor':
        K= st.slider("K",1,15)
        params["K"] = K
    elif nama_model=="Decission Tree":
        criterion = st.selectbox("Pilih Criterion",("entropy","gini"))
        params["criterion"] = criterion
    elif nama_model=="Random Forest":
        n_estimators=st.slider("n_estimators", 2, 15)
        params["n_estimators"] = n_estimators
    return params

def pilih_klasifikasi(nama_model,params):
    model=None
    if nama_model=="K-Nearest Neighbor":
        model = KNeighborsClassifier(n_neighbors=params["K"])
    elif nama_model=="Decission Tree":
        model = DecisionTreeClassifier(criterion=params["criterion"],max_depth=params["max_depth"])
    elif nama_model=="Naive-Bayes GaussianNB":
        model = GaussianNB()
    elif nama_model=="Random Forest":
        model = RandomForestClassifier(n_estimators=params["n_estimators"])
    return model

        
st.title("Stroke Prediction")
st.caption("""
    Ninieng Choirunnisa' | 200411100094 | Penambangan Data - B

    Teknik Informatika - Universitas Trunojoyo Madura""")
tab1, tab2, tab3, tab4 = st.tabs(["Home", "Preprocessing", "Normalization dan Modelling", "Implementation"])

with tab1:
   st.header("About Stroke")
   st.write("""
    Stroke adalah kondisi ketika pasokan darah ke otak terganggu karena penyumbatan (stroke iskemik) atau pecahnya pembuluh darah (stroke hemoragik). Kondisi ini menyebabkan area tertentu pada otak tidak mendapat suplai oksigen dan nutrisi sehingga terjadi kematian sel-sel otak.

    Stroke merupakan keadaan darurat medis, karena tanpa suplai oksigen dan nutrisi, sel-sel pada bagian otak yang terdampak bisa mati hanya dalam hitungan menit. Akibatnya, bagian tubuh yang dikendalikan oleh area otak tersebut tidak bisa berfungsi dengan baik.

    A. Gejala dan Penyebab Stroke

    Gejala stroke umumnya terjadi di bagian tubuh yang dikendalikan oleh area otak yang rusak. Gejala yang dialami penderita stroke bisa meliputi:

    - Lemah pada otot-otot wajah yang membuat satu sisi wajah turun

    - Kesulitan mengangkat kedua lengan akibat lemas atau mati rasa

    - Kesulitan berbicara

    - Disartria

    - Kesemutan

    - Kesulitan mengenal wajah (prosopagnosia)

    Penyebab stroke secara umum terbagi menjadi dua, yaitu adanya gumpalan darah pada pembuluh darah di otak dan pecahnya pembuluh darah di otak.

    Penyempitan atau pecahnya pembuluh darah tersebut dapat terjadi akibat beberapa faktor, seperti tekanan darah tinggi, penggunaan obat pengencer darah, aneurisma otak, dan trauma otak.

    B. Pengobatan dan Pencegahan Stroke

    Penanganan stroke tergantung pada jenis stroke yang dialami pasien. Tindakan yang dapat dilakukan bisa berupa pemberian obat-obatan atau operasi. Selain itu, untuk mendukung proses pemulihan, penderita akan disarankan untuk menjalani fisioterapi dan terapi psikologis.

    Pada umumnya, pencegahan stroke hampir sama dengan cara mencegah penyakit jantung, yaitu dengan menerapkan pola hidup sehat, seperti:

    - Menjaga tekanan darah agar tetap normal

    - Tidak merokok dan tidak mengonsumsi minuman beralkohol

    - Menjaga berat badan ideal

    - Berolahraga secara rutin

    - Menjalani pemeriksaan rutin untuk kondisi medis yang diderita, misalnya diabetes dan hipertensi
    """)

   st.header("About Dataset")
   st.write("""
   Dataset ini digunakan untuk memprediksi apakah pasien kemungkinan terkena stroke berdasarkan parameter input seperti jenis kelamin, usia, berbagai penyakit, dan status merokok. Setiap baris dalam data memberikan informasi yang relevan tentang pasien.
   
   **Informasi Atribut**
   1. id: pengidentifikasi unik
   2. gender: "Pria", "Wanita" atau "Lainnya"
   3. age : umur pasien
   4. Hypertension : 0 bila pasien tidak hipertensi, 1 bila pasien hipertensi
   5. heart_disease: 0 jika pasien tidak memiliki penyakit jantung, 1 jika pasien memiliki penyakit jantung
   6. ever_married: "Tidak" atau "Ya"
   7. work_type: "anak-anak", "Pekerja Pemerintah", "Never_worked", "Swasta" atau "Wiraswasta"
   8. Residence_type: "Pedesaan" atau "Perkotaan"
   9. avg_glucose_level : kadar glukosa rata-rata dalam darah
   10.bmi : indeks massa tubuh
   11. smoking_status: "sebelumnya merokok", "tidak pernah merokok", "merokok" atau "Tidak diketahui"*
   12. stroke: 1 jika pasien mengalami stroke atau 0 jika tidak
   
   *Catatan* : "Tidak diketahui" dalam smoking_status berarti informasi tidak tersedia untuk pasien ini
   
   Link Source Dataset : https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset""")
   df= load_dataset()
   df

   #VISUALISASI DATA
   #HEATMAP CORRELATION
   st.header("Visualisasi Data")
   st.subheader("A. Heatmap Correlation")
   # Compute the correlation matrix
   corr = df.corr()

   # Generate a mask for the upper triangle
   mask = np.triu(np.ones_like(corr, dtype=bool))

   # Set up the matplotlib figure
   f, ax = plt.subplots(figsize=(11, 9))

   # Generate a custom diverging colormap
   cmap = sns.diverging_palette(230, 20, as_cmap=True)

   # Draw the heatmap with the mask and correct aspect ratio
   sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
   st.pyplot(f)

   #COUNT PLOT
   st.subheader("B. Count Plot")
   sns.set_theme(style="darkgrid")

   #GENDER
   st.write("1. Gender")
   f, ax =plt.subplots()
   ax = sns.countplot(data=df, x="gender")
   st.pyplot(f)
   st.write("Di atas, Anda dapat dilihat bahwa jumalah Perempuan yang ada di dataset lebih tinggi daripada laki-laki.")


   #HYPERTENSION
   st.write("2. Hypertension")
   f, ax =plt.subplots()
   ax = sns.countplot(data=df, x="hypertension")
   st.pyplot(f)
   st.write("Dari atas terlihat bahwa semakin sedikit orang yang menderita hipertensi.")

   #STATUS MENIKAH
   st.write("3. Status Menikah")
   f, ax =plt.subplots()
   ax = sns.countplot(data=df, x="ever_married")
   st.pyplot(f)
   st.write("Rasio yang bisa dilihat dari atas adalah sekitar 2:1 untuk pernah menikah.")

   #WORK TYPE
   st.write("4. Work Type")
   f, ax =plt.subplots()
   ax = sns.countplot(data=df, x="work_type")
   st.pyplot(f)
   st.write("Banyak orang bekerja di sektor swasta.")

   #RECIDENCE TYPE
   st.write("5. Residence Type")
   f, ax =plt.subplots()
   ax = sns.countplot(data=df, x="Residence_type")
   st.pyplot(f)
   st.write("Jenis tempat tinggalnya sama untuk orang yang ada dalam dataset.")

   #SMOKING STATUS
   st.write("6. Smoking Status")
   f, ax =plt.subplots()
   ax = sns.countplot(data=df, x="smoking_status")
   st.pyplot(f)
   st.write("Banyak orang tidak pernah merokok seumur hidupnya. Namun, tidak diketahui pasti status Unknown dari dataset.")

   #SROKE
   st.write("7. Stroke")
   f, ax =plt.subplots()
   ax = sns.countplot(data=df, x="stroke")
   st.pyplot(f)
   st.write("""Dari variabel dependen di atas, kita benar-benar memiliki lebih sedikit orang yang menderita stroke. Berarti dataset kita tidak seimbang. Sehingga, harus menggunakan teknik pengambilan sampel untuk membuat keseimbangan data.""")

   #DISTRIBUTION PLOT
   st.subheader("C. Distibution Plot")
   #AVG GLOCOSE_LEVEL
   st.write("1.  Rata-rata Level Glukosa")
   fig = plt.figure(figsize=(7,7))
   sns.distplot(df.avg_glucose_level, color="green", label="avg_glucose_level", kde= True)
   plt.legend()
   st.pyplot(fig)

   #BMI
   st.write("2.  Indeks Masa Tubuh")
   fig = plt.figure(figsize=(7,7))
   sns.distplot(df.bmi, color="orange", label="bmi", kde= True)
   plt.legend()
   st.pyplot(fig)

   #STROKE VS NO STROKE BY BMI
   st.write("3.  Stroke VS No Stroke berdasarkan BMI")
   fig = plt.figure(figsize=(12,10))
   ax1=sns.distplot(df[df['stroke'] == 0]["bmi"], color='green', label="No Stroke", kde= True) # No Stroke - green
   ax2 =sns.distplot(df[df['stroke'] == 1]["bmi"], color='red', label="Stroke", kde= True) # Stroke - Red
   plt.legend()
   st.pyplot(fig)
   st.write("Dari grafik diatas, terlihat bahwa kepadatan orang yang kelebihan berat badan yang menderita stroke lebih banyak.")

   #STROKE VS NO STROKE BY AVG GLUCOSE_LEVEL
   st.write("4.  Stroke VS No Stroke berdasarkan Avg Glucose Level")
   fig = plt.figure(figsize=(12,10))
   sns.distplot(df[df['stroke'] == 0]["avg_glucose_level"], color='green', label="No Stroke", kde= True) # No Stroke - green
   sns.distplot(df[df['stroke'] == 1]["avg_glucose_level"], color='red', label="Stroke", kde= True) # Stroke - Red
   plt.xlim([30,330])
   plt.legend()
   st.pyplot(fig)
   st.write("Dari grafik diatas, terlihat bahwa kepadatan penduduk yang memiliki kadar glukosa kurang dari 100 lebih banyak yang menderita stroke.")

   #STROKE VS NO STROKE BY AGE
   st.write("4.  Stroke VS No Stroke berdasarkan Umur")
   fig = plt.figure(figsize=(12,10))
   sns.distplot(df[df['stroke'] == 0]["age"], color='green', label="No Stroke", kde= True) # No Stroke - green
   sns.distplot(df[df['stroke'] == 1]["age"], color='red', label="Stroke", kde= True) # Stroke - Red
   plt.xlim([18,100])
   plt.legend()
   st.pyplot(fig)
   st.write("Dari grafik diatas, terlihat bahwa kepadatan penduduk berusia di atas 50 tahun yang menderita stroke lebih banyak.")

   #SCATTER PLOT
   st.subheader("D. Scatter Plot")

   #AGE VS BMI
   st.write("1.  Age Vs BMI")
   fig = plt.figure(figsize=(7,7))
   graph = sns.scatterplot(data=df, x="age", y="bmi", hue='gender')
   graph.axhline(y= 25, linewidth=4, color='r', linestyle= '--')
   st.pyplot(fig)
   st.write("Dari plot di atas, kita dapat melihat bahwa banyak orang yang memiliki IMT di atas 25 mengalami kelebihan berat badan dan obesitas.")

   #AGE VS BMI
   st.write("2.  Age Vs Avg Glucose Level")
   fig = plt.figure(figsize=(7,7))
   graph = sns.scatterplot(data=df, x="age", y="avg_glucose_level", hue='gender')
   graph.axhline(y= 150, linewidth=4, color='r', linestyle= '--')
   st.pyplot(fig)
   st.write("Dari gambar di atas, kita dapat melihat bahwa orang yang memiliki kadar glukosa di atas 150 relatif lebih sedikit dibandingkan orang di bawah ini. Jadi, kita dapat mengatakan bahwa orang di atas 150 mungkin menderita diabetes.")

   #VIOLIN PLOT
   st.subheader("E. Violin Plot")
   fig = plt.figure(figsize=(13,13))
   plt.subplot(2,3,1)
   sns.violinplot(x = 'gender', y = 'stroke', data = df)
   plt.subplot(2,3,2)
   sns.violinplot(x = 'hypertension', y = 'stroke', data = df)
   plt.subplot(2,3,3)
   sns.violinplot(x = 'heart_disease', y = 'stroke', data = df)
   plt.subplot(2,3,4)
   sns.violinplot(x = 'ever_married', y = 'stroke', data = df)
   plt.subplot(2,3,5)
   sns.violinplot(x = 'work_type', y = 'stroke', data = df)
   plt.xticks(fontsize=9, rotation=45)
   plt.subplot(2,3,6)
   sns.violinplot(x = 'Residence_type', y = 'stroke', data = df)
   st.pyplot(fig)

   #PAIR PLOT
   st.subheader("F. Pair Plot")
   sns.pairplot(data=df,hue='stroke',size=2,palette='OrRd')
   st.pyplot(plt.show())


with tab2:
    st.write("""Teknik preprocessing data dapat meningkatkan kualitas data, sehingga membantu meningkatkan akurasi dan efisiensi proses penambangan selanjutnya. 
        Data preprocessing merupakan langkah penting dalam proses penemuan pengetahuan, karena keputusan yang berkualitas harus didasarkan pada data yang berkualitas. 
        Mendeteksi anomali data, memperbaikinya lebih awal, dan mengurangi data yang akan dianalisis dapat memberikan hasil yang sangat besar untuk pengambilan keputusan.""")
    
    st.subheader("Dataset Mentah")
    df = load_dataset()
    df

    st.write("Jumlah Baris dan Kolom :", df.shape)
    st.write("Jumlah Kelas :", len(df['stroke'].unique()))
    
    #Preprocessing

    st.subheader("Data Cleanning")
    st.write("Drop kolom id")
    df=df.drop(['id'],axis=1)
    df
    st.write("Mengatasi Hilangnya Data (Missing Value)")

    null= df.isnull().sum()
    null

    st.write("Dari tabel diatas dapat dilihat bahwa jumlah Missing Value pada kolom BMI sebanyak ",df.bmi.isnull().sum(), """
        dalam hal ini dapat diisi dengan strategi mean (rata-rata dari jumlah kolom)""")
    
    imp = SimpleImputer(strategy='mean')
    df['bmi'] = imp.fit_transform(df[['bmi']])
    
    st.write("Berikut ini merupakan data setelah nilai yang kosong diisi dengan strategi mean")
    df

    st.subheader("One Hot Encoding")
    
    categorical_cols = df.select_dtypes("object")
    df_new = pd.get_dummies(df, columns=categorical_cols.columns)
    df_new

    st.write("Jumlah Baris dan Kolom :", df_new.shape)
    st.write("Jumlah Kelas :", len(df_new['stroke'].unique()))
    


with tab3:
    st.header("Normalization")
    df_new2=df_new
    #Normalisasi
    nama_normal = st.selectbox(
        'Pilih Normalisasi',
        ('Standard Scaler', 'MinMax Scaler'))
    choose_normalisasi= normalisasi(df_new2,nama_normal)
    choose_normalisasi

    X = choose_normalisasi.drop(["stroke"], axis=1).values
    y = choose_normalisasi["stroke"].values
        

    #Proses Klasifikasi
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    st.header("Modelling")
    st.write("Pilihlah model yang akan dibandingkan!")

    grup_prediksi=[]
    acc=[]
    grup_precision=[]
    grup_recall=[]
    grup_f1=[]
    nb=st.checkbox("Naive-Bayes GaussianNB")
    if nb:
        model_nb=GaussianNB()
        model_nb.fit(X_train, y_train)
        y_pred= model_nb.predict(X_test)
        accuracy_nb=round(accuracy_score(y_pred, y_test)*100, ndigits = 2)
        precision_nb=round(precision_score(y_pred, y_test),ndigits=2)
        recall_nb =  round(recall_score(y_test, y_pred), ndigits=2)
        f1_nb = round(f1_score(y_test,y_pred), ndigits=2)
        grup_prediksi.append("Naive-Bayes GaussianNB")
        acc.append(accuracy_nb)
        grup_precision.append(precision_nb)
        grup_recall.append(recall_nb)
        grup_f1.append(f1_nb)

    knn=st.checkbox("K-Nearest Neighbor")
    if knn:
        K_value= st.slider("k",1,15)
        model_knn=KNeighborsClassifier(n_neighbors=K_value)
        model_knn.fit(X_train, y_train)
        y_pred= model_knn.predict(X_test)
        accuracy_knn=round(accuracy_score(y_pred, y_test)*100, ndigits = 2)
        precision_knn=round(precision_score(y_pred, y_test),ndigits=2)
        recall_knn =  round(recall_score(y_test, y_pred), ndigits=2)
        f1_knn = round(f1_score(y_test,y_pred), ndigits=2)
        grup_prediksi.append("K-Nearest Neighbor")
        acc.append(accuracy_knn)
        grup_precision.append(precision_knn)
        grup_recall.append(recall_knn)
        grup_f1.append(f1_knn)

    dt=st.checkbox("Decision Tree")
    if dt:
        kriteria = st.radio("Pilih Criterion",("entropy","gini"))
        model_dt=DecisionTreeClassifier(criterion=kriteria)
        model_dt.fit(X_train, y_train)
        y_pred= model_dt.predict(X_test)
        accuracy_dt=round(accuracy_score(y_pred, y_test)*100, ndigits = 2)
        precision_dt=round(precision_score(y_pred, y_test),ndigits=2)
        recall_dt =  round(recall_score(y_test, y_pred), ndigits=2)
        f1_dt = round(f1_score(y_test,y_pred), ndigits=2)
        grup_prediksi.append("Decision Tree")
        acc.append(accuracy_dt)
        grup_precision.append(precision_dt)
        grup_recall.append(recall_dt)
        grup_f1.append(f1_dt)

    rf=st.checkbox("Random Forest")
    if rf:
        n_estimators=st.slider("n_estimators", 2, 15)
        model_rf=RandomForestClassifier(n_estimators=n_estimators)
        model_rf.fit(X_train, y_train)
        y_pred= model_rf.predict(X_test)
        accuracy_rf=round(accuracy_score(y_pred, y_test)*100, ndigits = 2)
        precision_rf=round(precision_score(y_pred, y_test),ndigits=2)
        recall_rf =  round(recall_score(y_test, y_pred), ndigits=2)
        f1_rf = round(f1_score(y_test,y_pred), ndigits=2)
        grup_prediksi.append("Random Forest")
        acc.append(accuracy_rf)
        grup_precision.append(precision_rf)
        grup_recall.append(recall_rf)
        grup_f1.append(f1_rf)

    if st.button('Bandingkan'):
        st.subheader("Accuracy Score")
        for i in grup_prediksi:
            for j in acc:
                if grup_prediksi.index(i)==acc.index(j):
                    st.info(f"""{i} = {j}%""")
        x_pos = np.arange(len(grup_prediksi))
        plt.bar(x_pos, acc, align='center', alpha=0.5, color='blue')
        plt.xticks(x_pos, grup_prediksi, rotation=50)
        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy Score')
        st.pyplot(plt.show())

        st.subheader("Precision Score")
        for i in grup_prediksi:
            for j in grup_precision:
                if grup_prediksi.index(i)==grup_precision.index(j):
                    st.info(f"""{i} = {j}""")
        x_pos = np.arange(len(grup_prediksi))
        plt.bar(x_pos, grup_precision, align='center', alpha=0.5, color='blue')
        plt.xticks(x_pos, grup_prediksi, rotation=50)
        plt.ylabel('Precision')
        plt.title('Precision Score')
        st.pyplot(plt.show())

        st.subheader("Recall Score")
        for i in grup_prediksi:
            for j in grup_recall:
                if grup_prediksi.index(i)==grup_recall.index(j):
                    st.info(f"""{i} = {j}""")
        x_pos = np.arange(len(grup_prediksi))
        plt.bar(x_pos, grup_recall, align='center', alpha=0.5,  color='blue')
        plt.xticks(x_pos, grup_prediksi, rotation=50)
        plt.ylabel('Recall')
        plt.title('Recall Score')
        st.pyplot(plt.show())

        st.subheader("F1 Score")
        for i in grup_prediksi:
            for j in grup_f1:
                if grup_prediksi.index(i)==grup_f1.index(j):
                    st.info(f"""{i} = {j}""")
        x_pos = np.arange(len(grup_prediksi))
        plt.bar(x_pos, grup_f1, align='center', alpha=0.5,  color='blue')
        plt.xticks(x_pos, grup_prediksi, rotation=50)
        plt.ylabel('F1')
        plt.title('F1 Score')
        st.pyplot(plt.show())

    

with tab4:
    #Normalisasi
    name_normal = st.radio(
        'Pilih Normalisasi',
        ('Standard Scaler', 'MinMax Scaler'))
    pilih_normalisasi= normalisasi(df_new,name_normal)

    X = pilih_normalisasi.drop(["stroke"], axis=1).values
    y = pilih_normalisasi["stroke"].values
        

    #Proses Klasifikasi
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    nama_model=st.radio('Pilih Model',('Naive-Bayes GaussianNB','K-Nearest Neighbor', 'Decission Tree', 'Random Forest'))
    params = tambah_parameter(nama_model)
    model = pilih_klasifikasi(nama_model,params)
    
    model.fit(X_train, y_train)

    st.info(f"""
        Anda sekarang menggunakan jenis normalisasi {nama_normal} dan jenis model {nama_model}.
        """)

    

    with st.form("my_form"):
        age = st.number_input('Age')
        gender = st.selectbox('Gender',('Male', 'Female', 'Other'))
        hypertension = st.selectbox('Hypertension',('Yes', 'No'))
        HeartDisease = st.selectbox('Heart Disease',('Yes', 'No'))
        married = st.selectbox('Ever Married',('Yes', 'No'))
        workType = st.selectbox('Work Type',('Children', 'Govt Job', 'Never Worked', 'Private', 'Self-Employed'))
        Residence_type = st.selectbox('Residence Type',('Rural', 'Urban'))
        avgGlucose = st.number_input('Average Glucose Level(mg/dL)')
        bmi = st.number_input('Body Massa Index (Kg/m2)')
        smoke = st.selectbox('Smoking Status',('Unknown','Formerly Smoked', 'Never smoked','Smokes'))

        
        submitted = st.form_submit_button("Submit")

        #ubah data input 
        if gender=='Male':
            Female=0
            Male=1
            Other=0
        elif gender=='Female':
            Female=1
            Male=0
            Other=0
        else:
            Female=0
            Male=0
            Other=1

        if hypertension=='Yes':
            hypertension=1
        else:
            hypertension=0

        if HeartDisease=='Yes':
            heart_disease=1
        else:
            heart_disease=0

        if married=='Yes':
            Yes=1
            No=0
        else:
            Yes=0
            No=0


        if workType=='Children':
            Govt_job=0
            Never_worked=0
            Private=0
            Self_employed=0
            children=1
        elif married=='Govt job':
            Govt_job=1
            Never_worked=0
            Private=0
            Self_employed=0
            children=0
        elif married=='Never Worked':
            Govt_job=0
            Never_worked=1
            Private=0
            Self_employed=0
            children=0
        elif married=='Private':
            Govt_job=0
            Never_worked=0
            Private=1
            Self_employed=0
            children=0
        else:
            Govt_job=0
            Never_worked=0
            Private=0
            Self_employed=1
            children=0

        if Residence_type=='Urban':
            Urban=1
            Rural=0
        else:
            Urban=0
            Rural=1

        if smoke=='Formerly Smoked':
            formerly_smoked=1
            never_smoked=0
            smokes=0
            Unknown=0
        elif smoke=='Never smoked':
            formerly_smoked=0
            never_smoked=1
            smokes=0
            Unknown=0
        elif smoke=='Smoked':
            formerly_smoked=0
            never_smoked=0
            smokes=1
            Unknown=0
        else:
            formerly_smoked=0
            never_smoked=0
            smokes=0
            Unknown=1


        a=np.array([[age, hypertension, heart_disease, avgGlucose, bmi, Female, Male, Other, No, Yes, Govt_job, Never_worked, Private, Self_employed, children, Rural, Urban, Unknown, formerly_smoked, never_smoked, smokes]])
            
        data_input=pd.DataFrame(a, columns=['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'gender_Female', 'gender_Male', 'gender_Other','ever_married_No', 'ever_married_Yes', 'work_type_Govt_job', 'work_type_Never_worked', 'work_type_Private', 'work_type_Self-employed', 'work_type_children', 'Residence_type_Rural', 'Residence_type_Urban', 'smoking_status_Unknown', 'smoking_status_formerly smoked','smoking_status_never smoked', 'smoking_status_smokes'])
        
        
    if submitted:
        stroke_pred=model.predict(data_input)
        if stroke_pred[0]==1:
             st.error('Terdeteksi penyakit Stroke')
        else:
             st.success('Tidak terdeteksi penyakit Stroke')