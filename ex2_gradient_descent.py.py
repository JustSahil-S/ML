import numpy as np
from ex2_Functions import *
import random 
import matplotlib.pyplot as plt


if __name__ == "__main__":
    '''
    Load in data
    '''
    data_path = 'ex2_data.json'
    data = load_JSON(data_path)
    dataLen = len(data['Unrotated_Data'])
    lenList = []
    for i in range(0, dataLen):
        lenList.append(i)
    random.shuffle(lenList)
    target=[]
    prediction =[]
    for i in range(0, dataLen):
        target.append(data.get('Unrotated_Data')[lenList[i]])
        prediction.append(data.get('Rotated_Data')[lenList[i]])
    '''
    Make train/test split
    '''
    x = slice(10000, 12000)
    y = slice(0, 10000)
    x_target = target[x]
    y_target = np.array(target[y])
    x_train = np.array(prediction[x])
    y_train = np.array(prediction[y])  
    '''
    Create initial weight matrix
    '''
    '''Matrix =np.array([[-0.3, 0.6],[0.6, -0.8]]) '''
    Matrix =np.array([[-3.5, 70.1],[-50.3, 30.2]])
    print(y_train[0])

    '''
    Define hyperparameters (Training iterations, learning rate, etc.)
    '''
    '''
    Perform gradient descent
    '''
    def GradientDescent(y_train, y_target, Matrix):  
        delta = 0.01  
        alpha = 1
        epochs = 500
        i = 1
        convergeAt = 0
        minArray = []
        while i < epochs:
            newMatrix = Matrix-(np.dot(alpha,get_Jacobian(y_train, y_target, Matrix, delta)))
            newMin = objective_function(y_train, y_target, newMatrix)
            print(f'New Min: {newMin}')
            minArray.append(newMin)
            i += 1
            if newMin < objective_function(y_train, y_target, Matrix ):
                Matrix = newMatrix 
            else:
                if convergeAt == 0:
                    convergeAt = i
                    finalMatrix = newMatrix
                    finalMin = newMin
        print(f'Trained Model: {finalMatrix}, Training Loss: {finalMin}')
        print(f'Model Training Converged At: {convergeAt} epochs')
        print(f'Prediction: {np.dot(finalMatrix, x_train[1])} Actual:{x_target[1]}')
        print(f'Test Loss: {objective_function(x_train, x_target, finalMatrix)}')
        print(f'Rotation Angle: {np.arccos(finalMatrix[0][0])} degrees')

        plt.plot(range(0, i-1), minArray, '--bo', label = "Training Loss")
        plt.plot([convergeAt], [minArray[convergeAt]], marker="o", markersize=20, markeredgecolor="red", markerfacecolor="green", label = "Convergence Point")
        plt.title('Training Loss vs Epochs')
        plt.legend()
        plt.show()

GradientDescent(y_train, y_target, Matrix)