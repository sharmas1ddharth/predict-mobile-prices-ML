import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


# read csv file
train_data = pd.read_csv('data.csv')
test_data = pd.read_csv('test_data.csv')

# target to predict
y = train_data.price_range

# define features
features = ['battery_power','blue','clock_speed','dual_sim','fc','four_g','int_memory','m_dep','mobile_wt','n_cores','pc','px_height','px_width','ram','sc_h','sc_w','talk_time','three_g','touch_screen','wifi']

# data used to train model
X = train_data[features].copy()
test_X = test_data[features].copy()

# train test split to split the data into train and validation data
X_train, X_val, y_train, y_val = train_test_split(X, y)


# check best model for minimum mean absolute error
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
model_3 = RandomForestRegressor(n_estimators = 100, criterion='mae', random_state=0)
model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
model_5 = RandomForestRegressor(n_estimators=500, max_depth=7, random_state=0)

models = [model_1, model_2, model_3, model_4, model_5]

# function to check model
def check_model(model, X_t = X_train, X_v = X_val, y_t = y_train, y_v = y_val):
    model.fit(X_t, y_t)
    pred = model.predict(X_v)
    return mean_absolute_error(y_v, pred)

# print mean absolute error of the above models
for i in range(len(models)):
    mae = check_model(models[i])
    print(f"Model {i+1} : {mae}")

# best model selected with the help of check_model function created above
my_model = RandomForestRegressor(n_estimators=500, max_depth=7, random_state=0)

# fit the training data
my_model.fit(X, y)

# predicting the target data by providing features
model_prediction = my_model.predict(test_X)

# save the data into variale output 
output = pd.DataFrame({
    'price_range':predic
})

# write data to csv
output.to_csv('my_predictions.csv', index=False)


