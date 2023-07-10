import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

def reading_file(filename):
    db = pd.read_csv(filename)
    return db

def prerpocessing_data(db):
    # Drop any rows with null values
    db.dropna(inplace=True)
    # Fix strange data
    db['bathrooms'] = db['bathrooms'].astype(int)
    db['floors'] = db['floors'].astype(int)
    # Compute the first and third quartiles of 'price', 'sqft_living'
    Q1 = db[['price','sqft_lot','sqft_living','sqft_lot15','sqft_living15','sqft_above','sqft_basement']].quantile(0.25)
    Q3 = db[['price','sqft_lot','sqft_living','sqft_lot15','sqft_living15','sqft_above','sqft_basement']].quantile(0.75)

# Compute the interquartile range of 'price' and 'sqft_living'
    IQR = Q3 - Q1

# Remove rows that have 'price', 'sqft_living' values less than Q1 - 1.5*IQR or greater than Q3 + 1.5*IQR
    db = db[~((db[['price','sqft_lot','sqft_living','sqft_lot15','sqft_living15','sqft_above','sqft_basement']] < (Q1 - 1.5 * IQR)) | (db[['price','sqft_lot','sqft_living','sqft_lot15','sqft_living15','sqft_above','sqft_basement']] > (Q3 + 1.5 * IQR))).any(axis=1)]
    return db
# Encode categorical variables if any

def random_forest(db):
# two sets, parameters and target value. We also get rid of everything that is useless to us.
    X = db.drop(["id", "date", "price", "lat", "long", "zipcode"], axis=1)
    y = db["price"]
# Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.90, random_state=101)

# Initialize the random forest regressor with desired hyperparameters
    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=101)

# Fit the model on the training data
    rf.fit(X_train, y_train)

# Predict the target variable on the testing data
    y_pred = rf.predict(X_test)

# Evaluate the model using mean squared error and r2 score metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Mean Squared Error:", mse)
    print("R2 Score:", r2)

# plot the residuals
    plt.scatter(y_test,y_pred)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Random Forest')
    plt.show()

    residuals = y_test - y_pred
    sns.displot(residuals, kde=True)
    plt.title('Residual Distribution')
    plt.xlabel('Residuals')
    plt.show()

def linear_regression(db):
# two sets, parameters and target value. We also get rid of everything that is useless to us.
    X = db.drop(["id", "date", "price", "lat", "long", "zipcode"], axis=1)
    y = db['price']

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.90, random_state=101)
    
# fit a linear regression model to the data
    reg = LinearRegression()
    reg.fit(X_train, y_train)

    coeff_df = pd.DataFrame(reg.coef_,X.columns,columns=['Coefficient'])
    predictions = reg.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

# print mean squared error and r2 score
    print("Mean Squared Error: ", mse)
    print("R2 Score: ", r2)

# plot the residuals
    plt.scatter(y_test,predictions)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Linear Regression')
    plt.show()

    residuals = y_test - predictions
    sns.displot(residuals, kde=True)
    plt.title('Residual Distribution')
    plt.xlabel('Residuals')
    plt.show()

if __name__ == '__main__':
    temp = True
    while temp == True:
        print('What aglorithm you want to choose?')
        print('r - for random forest')
        print('l - for linear regression')

        choice = input()
        if choice == 'r':
            random_forest(prerpocessing_data(reading_file('variant1.csv')))

        elif choice == 'l':
            linear_regression(prerpocessing_data(reading_file('variant1.csv')))
        else:
            print("Please provide the correct answer")

        print('Do you want to continue?')
        print('Y - Yes')
        print('N - No')
        vybor = input()
        if vybor == 'Y':
            temp = True
        if vybor == 'N':
            temp = False