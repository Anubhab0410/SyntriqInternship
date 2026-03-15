import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

def load_data(file_path, has_genre=True):
    names = ['ID', 'TITLE', 'GENRE', 'DESCRIPTION'] if has_genre else ['ID', 'TITLE', 'DESCRIPTION']
    return pd.read_csv(file_path, sep=' ::: ', engine='python', names=names)

print("Loading data...")
train_df = load_data('train_data.txt')
test_solution_df = load_data('test_data_solution.txt')

print("Vectorizing descriptions...")
tfidf = TfidfVectorizer(stop_words='english', max_features=10000)
X_train = tfidf.fit_transform(train_df['DESCRIPTION'])
y_train = train_df['GENRE']
X_test = tfidf.transform(test_solution_df['DESCRIPTION'])
y_test = test_solution_df['GENRE']

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, solver='liblinear', random_state=42),
    "SVM (LinearSVC)": LinearSVC(dual=False, random_state=42),
    "Multinomial Naive Bayes": MultinomialNB(),
    "SGD Classifier": SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
}

results = []
trained_models = {}

print("\n--- Comparing Models ---")
for name, model in models.items():
    start = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    elapsed = time.time() - start
    
    trained_models[name] = model 
    results.append({"Model": name, "Accuracy": acc, "Time (s)": elapsed})
    print(f"{name:25} | Accuracy: {acc:.4%}")

results_df = pd.DataFrame(results)
best_model_info = results_df.loc[results_df['Accuracy'].idxmax()]
best_model_name = best_model_info['Model']
best_model_obj = trained_models[best_model_name]

print("\n" + "="*60)
print(f"WINNING MODEL: {best_model_name}")
print(f"FINAL ACCURACY: {best_model_info['Accuracy']:.4%}")
print("="*60)

print("\n--- Detailed Classification Report (Best Model) ---")
y_pred_best = best_model_obj.predict(X_test)
print(classification_report(y_test, y_pred_best, zero_division=0))

print("\n--- Testing Best Model on New Movie Descriptions ---")

sample_descriptions = [
    "A young boy discovers he has magical powers and attends a school for wizards.",
    "Two detectives hunt a serial killer who uses the seven deadly sins as motives.",
    "A hilarious story of three friends who go on a crazy road trip.",
    "A heartbreaking tale of love and sacrifice set during World War II."
]

sample_tfidf = tfidf.transform(sample_descriptions)
predicted_genres = best_model_obj.predict(sample_tfidf)

for desc, genre in zip(sample_descriptions, predicted_genres):
    print(f"\nDescription: {desc}")
    print(f"Predicted Genre: {genre.upper()}")