import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#Load the dataset.
file_path = 'GaussianNB-Model\heart.csv'
df = pd.read_csv(file_path)

#Replaces any missing values with the mean figure of the column. While not essential for 'heart.csv' this has been added for further study with different datasets.
if df.isnull().sum().any():
    print("Missing values detected. Imputing missing values with column mean...")
    df.fillna(df.mean(), inplace=True)

#Checks for columns that are populated with non-numerical values, in 'heart.csv' this is represented in the 'Sex' column. 
for column in df.columns:
    if df[column].dtype == 'object':
        print(f"Encoding non-numeric column: {column}")
        encoder = LabelEncoder()
        df[column] = encoder.fit_transform(df[column])

#Assigns X as the features and y as the label.
X = df.iloc[:, :-1]  #All columns except for HeartDisease.
y = df.iloc[:, -1]   #Only the HeartDisease label column.

#K-Means clustering as part of pre-processing.
print("Applying K-Means clustering...")
n_clusters = 3 
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans_clusters = kmeans.fit_predict(X)
#Adds a numerical value 0, 1, or 2 to represent which cluster the feature has been assigned to.
X['Cluster'] = kmeans_clusters

#Dataset split into training and test sets, 80% train & 20% test.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Normalises data as there is a large difference in numerical ranges.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Create and train the Gaussian Naive Bayes model.
gnb = GaussianNB()
gnb.fit(X_train, y_train)

#Make predictions on presence of heart disease.
y_pred = gnb.predict(X_test)

#Model evalutation metrics.
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Accuracy:")
print(accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


