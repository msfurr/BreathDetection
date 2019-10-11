# Runs classification on sample dataset

import numpy as np
from tflite_runtime.interpreter import Interpreter
        
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

def main():

    data = np.loadtxt('test1.txt')
    data = np.float32([data])
    
    interpreter = Interpreter(model_path = "Coral.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    for i in range(0, len(data[0])):
        
        input_data = data[0][[i]]
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]['index'])
        print(pred.argmax())
    
main()