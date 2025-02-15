
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
import warnings


warnings.filterwarnings('ignore', category=FutureWarning)


file_path = 'healthcare-dataset-stroke-data.csv'  
data = pd.read_csv(file_path)

# Tangani nilai yang hilang (jika ada)
data.dropna(inplace=True)

# Hapus atribut 'id'
if 'id' in data.columns:
    data.drop(columns=['id'], inplace=True)
    print("'id' telah dihapus dari dataset.")

# Tampilkan jumlah total data
print(f"Jumlah total data asli: {len(data)}")

# Encode variabel kategorikal
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Pisahkan fitur dan target
X = data.drop(columns=['stroke'])  # Asumsikan 'stroke' adalah kolom target
y = data['stroke']

# Menangani ketidakseimbangan data dengan SMOTE
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

# Tampilkan jumlah data setelah SMOTE
print(f"Jumlah data setelah penyeimbangan: {len(X_balanced)}")

# Normalisasi fitur (diperlukan untuk metode chi2)
scaler = MinMaxScaler()
X_balanced_normalized = scaler.fit_transform(X_balanced)

# Seleksi Fitur menggunakan Metode Filter
filter_selector = SelectKBest(score_func=chi2, k=5)  # Pilih 5 fitur terbaik
X_filtered_selected = filter_selector.fit_transform(X_balanced_normalized, y_balanced)
filter_support = filter_selector.get_support()
filter_selected_features = X.columns[filter_support]

# Seleksi Fitur menggunakan Metode Wrapper
wrapper_selector = RFE(estimator=RandomForestClassifier(n_estimators=100, random_state=42), n_features_to_select=5)
X_wrapper_selected = wrapper_selector.fit_transform(X_balanced_normalized, y_balanced)
wrapper_support = wrapper_selector.support_
wrapper_selected_features = X.columns[wrapper_support]

# Gabungkan Fitur yang Dipilih dari Metode Filter dan Wrapper
combined_features = list(set(filter_selected_features).union(wrapper_selected_features))
X_combined = X_balanced[combined_features]

# Pastikan 'heart_disease' dan 'hypertension' termasuk dalam fitur yang dipilih
if 'heart_disease' not in combined_features:
    combined_features.append('heart_disease')
    X_combined['heart_disease'] = X_balanced['heart_disease']
if 'hypertension' not in combined_features:
    combined_features.append('hypertension')
    X_combined['hypertension'] = X_balanced['hypertension']

# Bagi data menjadi train-test
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_balanced, test_size=0.3, random_state=42)

# Tampilkan jumlah data latih dan uji
print(f"Jumlah data latih: {len(X_train)}")
print(f"Jumlah data uji: {len(X_test)}")

# Latih Random Forest Classifier dengan jumlah estimator akhir
rf_model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)

# Evaluasi model
accuracy = rf_model.score(X_test, y_test)

print(f"Akurasi Model: {accuracy * 100:.2f}%")

# Confusion Matrix
y_pred = rf_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Visualisasi Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Tidak Stroke', 'Stroke'], yticklabels=['Tidak Stroke', 'Stroke'])
plt.xlabel('Prediksi')
plt.ylabel('Aktual')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
class_report = classification_report(y_test, y_pred, target_names=['Tidak Stroke', 'Stroke'], zero_division=1)
print("Classification Report:")
print(class_report)

# Visualisasi akurasi model pada jumlah estimator yang berbeda
train_accuracies = []
test_accuracies = []
estimators_range = range(1, 121, 10)  # Jumlah estimator dari 1 hingga 120 dengan interval 10

for n_estimators in estimators_range:
    temp_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, class_weight='balanced')
    temp_model.fit(X_train, y_train)
    train_accuracies.append(temp_model.score(X_train, y_train))
    test_accuracies.append(temp_model.score(X_test, y_test))

plt.figure(figsize=(10, 6))
plt.plot(estimators_range, train_accuracies, label='Training Accuracy', color='blue')
plt.plot(estimators_range, test_accuracies, label='Test Accuracy', color='red')
plt.xlabel('Number of Estimators (Epochs)')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Over Training Epochs')
plt.legend()
plt.grid()
plt.show()

# Output fitur yang terpilih
print("Fitur yang Terpilih:")
print(combined_features)

print("Fitur yang Dipilih dari Metode Filter:")
print(filter_selected_features)

print("Fitur yang Dipilih dari Metode Wrapper:")
print(wrapper_selected_features)

print("Fitur yang Digabungkan:")
print(combined_features)
