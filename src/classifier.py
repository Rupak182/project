from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns
import os



def train_xgboost(X,y):

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    model = XGBClassifier(n_estimators=100,learning_rate=0.1,max_depth=5,eval_metric='logloss')
    model.fit(X_train,y_train)
    
    os.makedirs('models', exist_ok=True)
    model.save_model('models/xgboost_model.json')
    print("XGBoost model saved to models/xgboost_model.json")
    
    y_pred = model.predict(X_test)
    print("Classification Report:\n",classification_report(y_test,y_pred,target_names=['Real','Fake']))
    cm = confusion_matrix(y_test,y_pred)
    sns.heatmap(cm,annot=True,fmt='d',cmap="Blues", xticklabels=['Real','Fake'],yticklabels=['Real','Fake'])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved to confusion_matrix.png")
    