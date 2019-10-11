##%

# NOTES
# Last updated 10.7.19

# Precision Scores w/out Class Switch
# 0: 0.95 - inhale
# 1: 0.78 - no action
# 2: 0.91 - exhale

# Precision Scores w/ Class Switch
# 0: 0.95
# 2: 0.96
# New Data
# 0: 0.91
# 2: 0.92

#%%

# Import libraries

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%% DEFINE HELPER FUNCTIONS

# Create class labeling function

def label_fix(label):
    if label < -7.5:
        return 0
    elif label > 7.5:
        return 2
    else:
        return 1
    
# Create moving average to remove single outliers
        
def movingAvg(Class, windowSize):
    
    filteredClass = []
    
    for i in range(0, len(Class)):
        
        if i < windowSize - 1:
            
            filteredClass.append(Class[i])
            
        elif Class[i] != Class[i - 1]:
            
            if sum(Class[i - (windowSize - 1):(i + 1)]) / windowSize > 1:
                
                filteredClass.append(2)
                
            elif sum(Class[i - (windowSize - 1):(i + 1)]) / windowSize < 1:
                
                filteredClass.append(0)
                
            else:
                
                filteredClass.append(Class[i])
                
        elif Class[i] == Class[i - 1]:
            
            filteredClass.append(Class[i])
        
    return filteredClass

def classSwitch(Class):

    decision = []
    switchLog = []

    for i in range(0, len(Class)):
        
        if i >= 1:
            
            if Class[i] != Class[i - 1]:
            
                # Log when it changes to 1
                
                if Class[i] == 1:
                
                    switchLog.append(i)
                    
                # If it changes to 0 or 2, add to final
                
                if Class[i] == 2 or Class[i] == 0:

                    decision.append(Class[i])
            
            # If the values continue to be 1, change to value before switch
            
            if Class[i] == 1:
                
                decision.append(Class[switchLog[-1] - 1])
                
            # If the value does not change and it is not 1, add to final
            
            elif Class[i] == Class[i - 1] and Class[i] != 1:
                
                decision.append(Class[i])
                
        # Add first value to final
        
        else:
            
            decision.append(Class[i])
            
    # Return the moving average with a window of 3 for the final list
    # to remove jumps between classes
    return movingAvg(decision, 3)

#%%

# Read in data

df = pd.read_csv('fbdh1.csv')
df['Class'] = df['Flow'].apply(label_fix)

#%%
    
# Create the data matrix and normalize data columns

X = df.drop('Class', axis = 1)
X.drop('Flow', axis = 1, inplace = True)

scaler = StandardScaler()
scaler.fit(X)
scaled_features = scaler.transform(X)

X = pd.DataFrame(scaled_features, columns = X.columns[:])

# Create the classification matrix

y = df['Class']

# Perform train test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

#%%

# Develop keras sequential model with optimized parameters through grid search cv

model = keras.Sequential()
model.add(keras.layers.Dense(150, activation = tf.nn.relu, input_dim = 8))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(50, activation = tf.nn.relu))
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(3, activation = tf.nn.softmax))

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])
model.summary()

EPOCHS = 50
model.fit(X_train, y_train, epochs = EPOCHS)

#%%

# Sort both x and y test back into the correct order

X_test.sort_index(inplace = True)
y_test.sort_index(inplace = True)

predictions = model.predict(X_test)

final_pred = []

for i in range(0, len(predictions)):
    
    final_pred.append(np.argmax(predictions[i]))

# Scoring the model on testing data from same dataset

print(classification_report(y_test, final_pred))

#%%

# Perform class switch

test = classSwitch(y_test.tolist())
pred = classSwitch(final_pred)

print(classification_report(test, pred))

#%% Test on new dataset

# Read in new dataframe for testing

test = pd.read_csv('test_data_2.csv')
test['Class'] = test['Flow'].apply(label_fix)

# Create the data matrix and scale data

X_1 = test.drop('Class', axis = 1)
X_1.drop('Flow', axis = 1, inplace = True)

scaler = StandardScaler()
scaler.fit(X_1)
scaled_features = scaler.transform(X_1)
X_1 = pd.DataFrame(scaled_features, columns = X_1.columns[:])

# Create the classification matrix

y_1 = test['Class']

test_predictions = model.predict(X_1)

final_pred_test = []
for score in range(0, len(test_predictions)):
    final_pred_test.append(np.argmax(test_predictions[score]))
    
print(classification_report(y_1, final_pred_test))

#%% Perform Class Switch on new dataset

test_new_data = classSwitch(y_1.tolist())
pred_new_data = classSwitch(final_pred_test)

print(classification_report(test_new_data, pred_new_data))

#$$ Seaborn Plot

fig = plt.figure(figsize = (10, 5))
plt.subplot(2, 1, 1)
plt.title('Ground Truth Class')
sns.scatterplot(y = test['Flow'][0:450], x = range(0, 450),\
                hue = test_new_data[0:450], palette = 'coolwarm')

plt.subplot(2, 1, 2)
plt.title('Predictions')
sns.scatterplot(y = test['Flow'][0:450], x = range(0, 450),\
                hue = pred_new_data[0:450], palette = 'coolwarm')

#%% Save model

#keras_file = "Coral_Model.h5"
#keras.models.save_model(model, keras_file)
#
#converter = tf.lite.TFLiteConverter.from_keras_model_file('Coral_Model.h5')
#converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
#
#tflite_model = converter.convert()
#
#open("Coral.tflite", "wb").write(tflite_model)













