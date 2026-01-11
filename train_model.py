# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import make_pipeline 
# from sklearn.preprocessing import StandardScaler 
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report
# import pickle


# print("Đang đọc dữ liệu...")
# data = pd.read_csv('data_training.csv')

# X = data.drop('class', axis=1) 

# y = data['class']


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)


# model = RandomForestClassifier(n_estimators=100, random_state=1234)

# print("Đang training...")
# model.fit(X_train, y_train)

# y_predict = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_predict)

# print("--------------------------------")
# print(f"Độ chính xác (Accuracy): {accuracy * 100:.2f}%")
# print("--------------------------------")
# print("Báo cáo chi tiết:")
# print(classification_report(y_test, y_predict))

# with open('pose_model.pkl', 'wb') as f:
#     pickle.dump(model, f)
    
# print("Đã lưu model thành công vào file 'pose_model.pkl'!")






import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle


print("Đang đọc dữ liệu...")
# Load Training Data
train_data = pd.read_csv('train.csv')
X_train = train_data.drop('class', axis=1)
y_train = train_data['class']

# Load Testing Data
test_data = pd.read_csv('test.csv')
X_test = test_data.drop('class', axis=1)
y_test = test_data['class']


# model = RandomForestClassifier(n_estimators=100, random_state=1234)
model = RandomForestClassifier(n_estimators=100, 
                               random_state=1234, 
                               class_weight='balanced')
# model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=1234)
print("Đang training...")
model.fit(X_train, y_train)

y_predict = model.predict(X_test)
accuracy = accuracy_score(y_test, y_predict)

print("--------------------------------")
print(f"Độ chính xác (Accuracy): {accuracy * 100:.2f}%")
print("--------------------------------")
print("Báo cáo chi tiết:")
print(classification_report(y_test, y_predict))
print("Ma trận nhầm lẫn (Confusion Matrix):")
print(confusion_matrix(y_test, y_predict))

with open('pose_model2.pkl', 'wb') as f:
    pickle.dump(model, f)
    
print("Đã lưu model thành công vào file 'pose_model2.pkl'!")