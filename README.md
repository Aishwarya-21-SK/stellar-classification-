# stellar-classification-
12/24/24, 9:20 PM
 stellar.ipynb - Colab
 !pip install pandas scikit-learn matplotlib seaborn
 Requirement already satisfied: pandas in /usr/local/lib/python3.10/dist-packages (2.2.2)
 Requirement already satisfied: scikit-learn in /usr/local/lib/python3.10/dist-packages (1.6.0)
 Requirement already satisfied: matplotlib in /usr/local/lib/python3.10/dist-packages (3.8.0)
 Requirement already satisfied: seaborn in /usr/local/lib/python3.10/dist-packages (0.13.2)
 Requirement already satisfied: numpy>=1.22.4 in /usr/local/lib/python3.10/dist-packages (from pandas) (1.26.4)
 Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.10/dist-packages (from pandas) (2.8.2)
 Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas) (2024.2)
 Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.10/dist-packages (from pandas) (2024.2)
 Requirement already satisfied: scipy>=1.6.0 in /usr/local/lib/python3.10/dist-packages (from scikit-learn) (1.13.1)
 Requirement already satisfied: joblib>=1.2.0 in /usr/local/lib/python3.10/dist-packages (from scikit-learn) (1.4.2)
 Requirement already satisfied: threadpoolctl>=3.1.0 in /usr/local/lib/python3.10/dist-packages (from scikit-learn) (3.5.0)
 Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib) (1.3.1)
 Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.10/dist-packages (from matplotlib) (0.12.1)
 Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib) (4.55.3)
 Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib) (1.4.7)
 Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib) (24.2)
 Requirement already satisfied: pillow>=6.2.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib) (11.0.0)
 Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib) (3.2.0)
 Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.8.2->pandas) (1.17.0)
 import pandas as pd
 from sklearn.model_selection import train_test_split
 from sklearn.preprocessing import StandardScaler
 from sklearn.svm import SVC
 from sklearn.neighbors import KNeighborsClassifier
 from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
 import seaborn as sns
 import matplotlib.pyplot as plt
 from google.colab import files
 uploaded = files.upload()
 Choose Files No file chosen 
enable
 Upload widget is only available when the cell has been executed in the current browser session. Please rerun this cell to
 data = pd.read_csv(list(uploaded.keys())[0])
 print(data.head())
         obj_ID       alpha      delta         u         g         r  \
 0  1.237661e+18  135.689107  32.494632  23.87882  22.27530  20.39501   
1  1.237665e+18  144.826101  31.274185  24.77759  22.83188  22.58444   
2  1.237661e+18  142.188790  35.582444  25.26307  22.66389  20.60976   
3  1.237663e+18  338.741038  -0.402828  22.13682  23.77656  21.61162   
4  1.237680e+18  345.282593  21.183866  19.43718  17.58028  16.49747   
          i         z  run_ID  rerun_ID  cam_col  field_ID   spec_obj_ID  \
 0  19.16573  18.79371    3606       301        2        79  6.543777e+18   
1  21.16812  21.61427    4518       301        5       119  1.176014e+19   
2  19.34857  18.94827    3606       301        2       120  5.152200e+18   
3  20.50454  19.25010    4192       301        3       214  1.030107e+19   
4  15.97711  15.54461    8102       301        3       137  6.891865e+18   
    class  redshift  plate    MJD  fiber_ID  
0  GALAXY  0.634794   5812  56354       171  
1  GALAXY  0.779136  10445  58158       427  
2  GALAXY  0.644195   4576  55592       299  
3  GALAXY  0.932346   9149  58039       775  
4  GALAXY  0.116123   6121  56187       842  
print(data.info())
 <class 'pandas.core.frame.DataFrame'>
 RangeIndex: 100000 entries, 0 to 99999
 Data columns (total 18 columns):
 #   Column       Non-Null Count   Dtype  ---  ------       --------------   -----  
 0   obj_ID       100000 non-null  float64
 1   alpha        100000 non-null  float64
 2   delta        100000 non-null  float64
 3   u            100000 non-null  float64
 4   g            100000 non-null  float64
 5   r            100000 non-null  float64
 6   i            100000 non-null  float64
 7   z            100000 non-null  float64
 8   run_ID       100000 non-null  int64  
 9   rerun_ID     100000 non-null  int64  
 10  cam_col      100000 non-null  int64  
 11  field_ID     100000 non-null  int64  
 12  spec_obj_ID  100000 non-null  float64
 https://colab.research.google.com/drive/1AVNV39UlE1tNGbP5FLyTDDUXKx0l4QET#printMode=true
 1/3
 13  class        100000 non-null  object 
 14  redshift     100000 non-null  float64
 15  plate        100000 non-null  int64  
 16  MJD          100000 non-null  int64  
 17  fiber_ID     100000 non-null  int64  
dtypes: float64(10), int64(7), object(1)
 memory usage: 13.7+ MB
 None
 features = ['u', 'g', 'r', 'i', 'z', 'redshift']
 X = data[features]
 y = data['class']
 y = y.astype('category').cat.codes
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 scaler = StandardScaler()
 X_train = scaler.fit_transform(X_train)
 X_test = scaler.transform(X_test)
 svm_model = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
 svm_model.fit(X_train, y_train)
 y_pred_svm = svm_model.predict(X_test)
 knn_model = KNeighborsClassifier(n_neighbors=5)
 knn_model.fit(X_train, y_train)
 y_pred_knn = knn_model.predict(X_test)
 print("SVM Classification Report:")
 print(classification_report(y_test, y_pred_svm))
 SVM Classification Report:
              precision    recall  f1-score   support
           0       0.97      0.97      0.97     11860
           1       0.97      0.89      0.93      3797
           2       0.94      1.00      0.97      4343
    accuracy                           0.96     20000
   macro avg       0.96      0.95      0.96     20000
 weighted avg       0.96      0.96      0.96     20000
 print("k-NN Classification Report:")
 print(classification_report(y_test, y_pred_knn))
 k-NN Classification Report:
              precision    recall  f1-score   support
           0       0.97      0.97      0.97     11860
           1       0.95      0.91      0.93      3797
           2       0.96      1.00      0.98      4343
    accuracy                           0.96     20000
   macro avg       0.96      0.96      0.96     20000
 weighted avg       0.96      0.96      0.96     20000
 fig, axes = plt.subplots(1, 2, figsize=(12, 5))
 sns.heatmap(confusion_matrix(y_test, y_pred_svm), annot=True, fmt="d", ax=axes[0], cmap="Blues")
 axes[0].set_title("SVM Confusion Matrix")
 sns.heatmap(confusion_matrix(y_test, y_pred_knn), annot=True, fmt="d", ax=axes[1], cmap="Greens")
 axes[1].set_title("k-NN Confusion Matrix")
 plt.show()
 12/24/24, 9:20 PM stellar.ipynb - Colab
 https://colab.research.google.com/drive/1AVNV39UlE1tNGbP5FLyTDDUXKx0l4QET#printMode=true 2/3
12/24/24, 9:20 PM
 stellar.ipynb - Colab
 accuracy_svm = accuracy_score(y_test, y_pred_svm)
 accuracy_knn = accuracy_score(y_test, y_pred_knn)
 print(f"SVM Accuracy: {accuracy_svm:.2f}")
 print(f"k-NN Accuracy: {accuracy_knn:.2f}")
 SVM Accuracy: 0.96
 k-NN Accuracy: 0.96
 3/3
 https://colab.research.google.com/drive/1AVNV39UlE1tNGbP5FLyTDDUXKx0l4QET#printMode=true
