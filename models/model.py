from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC



def run_svc(data):

    array = data.values
    X = array[:, 0:4]
    Y = array[:, 4]

    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.3, random_state=42)

    model = SVC(gamma='auto')
    model.fit(X_train, Y_train)
    predictions = model.predict(X_validation)

    return {'predictions': predictions, 'Y_validation': Y_validation}

def eval_results(results: dict):
    print("------Accuracy Score-----")
    print(accuracy_score(results['Y_validation'], results['predictions']))
    print("")
    print("-----Confusion Matrix----")
    print(confusion_matrix(results['Y_validation'], results['predictions']))
    print("")
    print("--Classification Report--")
    print(classification_report(results['Y_validation'], results['predictions']))