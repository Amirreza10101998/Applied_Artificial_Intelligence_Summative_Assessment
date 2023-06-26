import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.svm import SVC
from deap import base, creator, tools, algorithms
from joblib import dump

def load_data(file_path):
    """
    Load dataset into a pandas DataFrame.
    
    Parameters:
        file_path (str): The file path of the dataset
        
    Returns:
        DataFrame or None: Pandas DataFrame containing the dataset or None if an error occurs
    """
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None

def preprocess_data(renewable_energy_data, co2_emissions_data, numeric_columns):
    """
    Preprocess the datasets: handle missing values, merge datasets, normalize numeric columns,
    and perform one-hot encoding on categorical variables.
    
    Parameters:
        renewable_energy_data (DataFrame): Renewable energy data
        co2_emissions_data (DataFrame): CO2 emissions data
        numeric_columns (list): List of column names to normalize
        
    Returns:
        DataFrame: Preprocessed data
    """
    # Handle Missing Values
    renewable_energy_data = renewable_energy_data.apply(lambda x: x.fillna(x.mean()) if x.dtype.kind in 'biufc' else x.fillna(x.mode()[0]))
    co2_emissions_data = co2_emissions_data.apply(lambda x: x.fillna(x.mean()) if x.dtype.kind in 'biufc' else x.fillna(x.mode()[0]))

    # Merge datasets
    merged_data = pd.merge(renewable_energy_data, co2_emissions_data, how='inner', on='Year')

    # Normalize the numeric columns
    scaler = MinMaxScaler()
    merged_data[numeric_columns] = scaler.fit_transform(merged_data[numeric_columns])

    # One-hot encoding of categorical variables
    return pd.get_dummies(merged_data, drop_first=True)

def evaluate(individual, X, y):
    """
    Evaluate the fitness of an individual.
    
    Parameters:
        individual (list): Binary list representing the selected features
        X (array): Feature matrix
        y (array): Target variable
        
    Returns:
        tuple: Fitness score of the individual
    """
    if sum(individual) == 0:
        return 0,  # Avoid using none of the features
    selected_features = [i for i, val in enumerate(individual) if val == 1]
    clf = SVC()
    return cross_val_score(clf, X[:, selected_features], y, cv=5).mean(),

def run_genetic_algorithm(X, y):
    """
    Run Genetic Algorithm for feature selection
    
    Parameters:
        X (array): Feature matrix
        y (array): Target variable
        
    Returns:
        list: Indices of the selected features
    """
    # Create types
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # Initialize genetic algorithm
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=X.shape[1])
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Genetic operators
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Evaluation function
    evaluate_partial = lambda individual: evaluate(individual, X, y)
    toolbox.register("evaluate", evaluate_partial)

    # Parameters
    n_gen = 20
    cxpb = 0.5
    mutpb = 0.2

    # Run GA
    population = toolbox.population(n=40)
    algorithms.eaSimple(population, toolbox, cxpb, mutpb, n_gen, verbose=True)

    # Return best solution
    best_individual = tools.selBest(population, 1)[0]
    return [index for index, value in enumerate(best_individual) if value == 1]

def train_svm(X, y, param_grid):
    """
    Train an SVM classifier with grid search cross-validation.
    
    Parameters:
        X (array): Feature matrix
        y (array): Target variable
        param_grid (dict): Parameters to tune
        
    Returns:
        SVC: Trained model
        float: Cross-validation score
    """
    clf = SVC()
    grid_search = GridSearchCV(clf, param_grid, cv=5)
    grid_search.fit(X, y)
    return grid_search.best_estimator_, grid_search.best_score_

def main():
    # Load datasets
    renewable_energy_data = load_data('uk_renewable_energy.csv')
    co2_emissions_data = load_data('GCB2022v27_MtCO2_flat.csv')

    # Exit if data loading failed
    if renewable_energy_data is None or co2_emissions_data is None:
        return

    # Define columns to normalize
    numeric_columns = ['Energy from renewable & waste sources', 'Total']

    # Preprocess data
    preprocessed_data = preprocess_data(renewable_energy_data, co2_emissions_data, numeric_columns)
    X = preprocessed_data.iloc[:, :-1].values
    y = preprocessed_data.iloc[:, -1].values

    # Run Genetic Algorithm for feature selection
    selected_features = run_genetic_algorithm(X, y)
    X_selected = X[:, selected_features]

    # Define parameters to tune for SVM
    param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto'], 'kernel': ['linear', 'rbf']}

    # Train SVM with grid search cross-validation
    model, score = train_svm(X_selected, y, param_grid)

    # Save the trained model
    dump(model, 'svm_model.joblib')

    # Print results
    print("\nSelected Features: ", selected_features)
    print("Best Cross-Validation Score: ", score)

if __name__ == "__main__":
    main()

