#necessary imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import tkinter as tk
from tkinter import filedialog

dirName = filedialog.askopenfile(initialdir="/",title='Please select a file') #choosing which files to read/getting the file path

df = pd.read_csv(dirName) #reading csv file
df.fillna(method ='ffill', inplace = True) #removing holes or nonvalues
df.dropna(inplace = True) #removing holes or nonvalues

x = df["TV"] #reading specific column
y = df["Sales"] #reading specific column

x = np.array((x-x.mean())/x.std()) #feature scaling
y = np.array((y-y.mean())/y.std()) #feature scaling


def initialize_wandb(): #initialize the initial random values of b and w, ok
    b = random.random() #random value of b
    w = random.random() #random value of w
    return b, w

def predict_Yhat(b, w, x): #taking the dot product of x and w and adding b which predicts the values of y hat
    return b + np.dot(x, w)

def get_cost(y, yHat): #calculate the value of the cost function
    return (1/len(y)) * np.sum((yHat-y)**2)

def update_w_and_b(x, y, y_hat, b_0, w_0, learning_rate): #updating w and b based on running gradient descent
    dw = (1/len(x)) * (sum((y_hat-y) * x))
    db = (1/len(y)) * (sum(y_hat-y))
    b_new = b_0 - (learning_rate * db)
    w_new = w_0 - (learning_rate * dw)
    return b_new, w_new

def run_gradient_descent(x, y, alpha, stopPoint): #the actual gradient descent method
    iList = [] #initializing the necessary variables
    costList = []
    stopPoint = stopPoint
    i = 0
    n = 0
    b, w = initialize_wandb() #initializing a value of oldCost because I have to
    yHattemp = predict_Yhat(b, w, x)
    oldCost = get_cost(y, yHattemp)

    b, w = initialize_wandb()

    while(n < 10): #infinite loop to insure that gd will run until stopped
        yHat = predict_Yhat(b, w, x) #getting yHat
        thisCost = get_cost(y, yHat) #getting current cost
        costList.append(thisCost) #appending current cost to list
        iList.append(i) #appending iteration number to list

        if (abs(oldCost - thisCost == stopPoint )): #logic to break the infinite loop that ensures that the cost value is at the lowest possible value
            break
        oldCost = thisCost #making the curent cost the old cost so the if statement can work properly
        
        prev_b = b
        prev_w = w
        b, w = update_w_and_b(x, y, yHat, prev_b, prev_w, alpha) #getting new balues of w and b
        i = i + 1
        
        print("estimate of b and w: ", b, w)
        print("Cost value (r2 value): ", thisCost)
        print("interation number: ", i)

    print("Final Estimate of b and w : ", b, w)
    return thisCost, b, w, i, costList, iList

costValue, b, w, i, costList, iList = run_gradient_descent(x, y, alpha = 0.01, stopPoint = 0)

print("final cost value = " + str(costValue))
print("final linear equation: y = " + str(w) + "x + " + str(b))

#two graphs that show the cost value vs the iterations and the points with the line of best fit
plt.subplot(1, 2, 1)
plt.plot(iList, costList)
plt.plot(costValue)
plt.title("J vs Iterations")
plt.xlabel("Iterations")
plt.ylabel("J")

plt.subplot(1, 2, 2)
plt.scatter(x, y)
plt.plot(x, w*x+b, "-r", label = "best fit line")
plt.title("Best line fit")
plt.xlabel("TV")
plt.ylabel("Sales")

plt.suptitle("Cost Value Graph and Best Fit Line Graph")
plt.show()