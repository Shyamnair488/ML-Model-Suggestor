from flask import Flask, render_template, request, jsonify
from flask_wtf import FlaskForm
from wtforms import FileField
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, accuracy_score
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA, NMF
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from tabulate import tabulate

app = Flask(__name__)
app.config['SECRET_KEY'] = '05743738d450c98b5d69b349e88bdcdd'  # Use your generated secret key

class UploadForm(FlaskForm):
    file = FileField('Upload CSV File')

def handle_supervised(data, target_column):
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

    # Check the type of the target variable
    target_type = data_imputed[target_column].dtype

    if target_type == 'object':
        # Classification models
        le = LabelEncoder()
        data_imputed[target_column] = le.fit_transform(data_imputed[target_column])

        classification_models = [
            ("Random Forest Classifier", RandomForestRegressor()),
            ("Gradient Boosting Classifier", GradientBoostingRegressor()),
            ("KNeighbors Classifier", KNeighborsRegressor()),
            ("MLP Classifier", MLPRegressor())
        ]

        return classification_models, None, None, None
    elif target_type in ['int64', 'float64']:
        # Regression models
        X_train, X_test, y_train, y_test = train_test_split(data_imputed.drop(columns=[target_column]), data_imputed[target_column], test_size=0.2, random_state=42)

        regression_models = [
            ("Linear Regression", LinearRegression()),
            ("Ridge Regression", Ridge()),
            ("Lasso Regression", Lasso()),
            ("ElasticNet Regression", ElasticNet()),
            ("SVR", SVR()),
            ("Random Forest Regressor", RandomForestRegressor()),
            ("Gradient Boosting Regressor", GradientBoostingRegressor()),
            ("KNeighbors Regressor", KNeighborsRegressor()),
            ("MLP Regressor", MLPRegressor())
        ]

        return None, regression_models, X_train, X_test, y_train, y_test

def handle_unsupervised(data):
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

    # Clustering models
    clustering_models = [
        ("KMeans Clustering", KMeans()),
        ("Agglomerative Clustering", AgglomerativeClustering()),
        ("DBSCAN", DBSCAN()),
        ("Gaussian Mixture Model", GaussianMixture())
    ]

    # Dimensionality reduction models
    dimensionality_reduction_models = [
        ("Principal Component Analysis (PCA)", PCA()),
        ("Non-Negative Matrix Factorization (NMF)", NMF()),
        ("t-distributed Stochastic Neighbor Embedding (t-SNE)", TSNE())
    ]

    return clustering_models, dimensionality_reduction_models, data_imputed

def evaluate_classification_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def evaluate_regression_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    return r2

def suggest_model(data, target_column=None):
    if target_column is not None:  # Supervised learning
        classification_models, regression_models, X_train, X_test, y_train, y_test = handle_supervised(data, target_column)

        results = []

        if classification_models:
            for name, model in classification_models:
                accuracy = evaluate_classification_model(model.fit(X_train, y_train), X_test, y_test)
                results.append((name, model, accuracy))
        elif regression_models:
            for name, model in regression_models:
                r2 = evaluate_regression_model(model.fit(X_train, y_train), X_test, y_test)
                results.append((name, model, r2))

        return results, None
    else:  # Unsupervised learning
        clustering_models, dimensionality_reduction_models, data_imputed = handle_unsupervised(data)

        clustering_results = [(name, model.fit(data_imputed).labels_) for name, model in clustering_models]
        dimensionality_reduction_results = [(name, model.fit_transform(data_imputed)) for name, model in dimensionality_reduction_models]

        return clustering_results, dimensionality_reduction_results

def display_results(results):
    headers = ["Model", "accuracy"]
    table = tabulate(results, headers=headers, tablefmt="pretty")
    return table

@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()
    error_message = None
    results_table = None

    if request.method == 'POST':
        if form.validate_on_submit():
            file = request.files['file']
            target_column = request.form.get('target_column')

            if file and target_column:
                try:
                    data = pd.read_csv(file)
                    results, _ = suggest_model(data, target_column)
                    results_table = display_results(results)
                except Exception as e:
                    error_message = f"Error processing the file: {str(e)}"
            else:
                error_message = "Please provide both a file and a target column."

    return render_template('index.html', form=form, results=results_table, error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)
