import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

def load_data(file_path, num_outputs):
    data = pd.read_csv(file_path)

    #print("Data head:", data.head())
    #print("Data shape:", data.shape)
    data= data.dropna()

    # Split features and targets
    if num_outputs > 0:
        X = data.iloc[:, :-num_outputs]
        y = data.iloc[:, -num_outputs:]
    else:
        X = data
        y = None


    feature_names = X.columns.tolist()

    #print("Feature names:", feature_names)
    return X,y, feature_names
X, y, feature_names = load_data('housing_labelled.csv' , 1)

def visualise_data(X, y, feature_names):
  for i in range(len(feature_names)):
    fig,ax = plt.subplots()
    ax.scatter(X.iloc[:,i], y)
    ax.set_xlabel(feature_names[i])
    ax.set_ylabel(y.columns.tolist()[0])
    plt.show(block=True)

#visualise_data(X, y, feature_names)

def simple_linear_regression(X, y, feature_names):
  for j in range(len(feature_names)):
    simple_linear_reg_model = LinearRegression().fit(X.iloc[:, [j]], y)
    y_pred = simple_linear_reg_model.predict(X.iloc[:, [j]])
    MSE = ((y - y_pred)** 2).mean()
    #print("MSE of simple linear regression using feature " , feature_names[j] , MSE)
    
simple_linear_regression(X, y, feature_names)


def multiple_linear_regression(X, y):
    multiple_linear_reg_model = LinearRegression().fit(X, y)
    y_pred = multiple_linear_reg_model.predict(X)
    MSE = ((y - y_pred)** 2).mean()
    #print("MSE of multiple linear regression using all features:" ,  MSE)
    return multiple_linear_reg_model

multiple_linear_reg_model= multiple_linear_regression(X, y)

def save_model(model, file_path):
    pickle.dump(model, open(file_path, 'wb'))

save_model(multiple_linear_reg_model, 'mul_lin_reg_model.pkl')

model = pickle.load(open('mul_lin_reg_model.pkl', 'rb'))

# Load the new, unlabelled data
X_new, _, _= load_data('housing_new_data.csv', num_outputs = 0)

# Make predictions on the novel data
y_predict = model.predict(X_new)
#print("prediction complete")

# Save predictions to a CSV file
#print(y_predict)
pd.DataFrame(y_predict).to_csv('housing_prediction.csv')


# Mode of operation: 'develop' for training or 'usage' for making predictions on unlabelled data
mode = 'usage'


# Development Mode
if mode == 'develop':
    # Load the labelled dataset
    X, y, feature_names = load_data('housing_labelled.csv', num_outputs= 1)
    # do basic  visualisation to understand relationships between each feature & the target
    visualise_data(X, y, feature_names)
    # fit various simple linear regression models, one for each feature
    simple_linear_regression(X, y, feature_names)
    # fit one multiple linear regression model using all features
    multiple_linear_reg_model = multiple_linear_regression(X, y)
    # save model so you can use it for prediction any time in the future
    save_model(multiple_linear_reg_model, 'mul_lin_reg_model.pkl')


# Evaluation Mode
elif mode == 'usage':
    # Load the trained model and scaler from files
    model = pickle.load(open('mul_lin_reg_model.pkl', 'rb'))

    # Load the new, unlabelled data
    X_new, _, _= load_data('housing_new_data.csv', num_outputs = 0)

    # Make predictions on the novel data
    y_predict = model.predict(X_new)
    print("prediction complete")

    # Save predictions to a CSV file
    print(y_predict)
    pd.DataFrame(y_predict).to_csv('housing_prediction.csv')