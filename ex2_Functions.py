import json
import numpy as np


def load_JSON(path='/Users/sahil/ML/Lab___export/Lab2_data.json'):
    '''
    :param path: (str) path name the .json file is located at.
    :return data: Dictionary-like with the following attributes:
                unrotated_data : {ndarray} of shape (12000, 2)
                rotated_data : {ndarray} of shape (12000, 2)
    '''
    if path is None:
        raise ValueError('Please input path name')

    with open(path) as json_file:
        data = json.load(json_file)

    for key in data.keys():
        data[key]  = np.array(data[key])
    return data




def objective_function(y_train, y_target, Matrix):
    x = 0
    length = len(y_train)
    for i in range(0, length):
       yhat = np.dot(Matrix, y_train[i])
       dist = np.linalg.norm(yhat-y_target[i])
       x += dist
    RMSE = (1/length)*x
    return RMSE


def get_Jacobian(y_train, y_target, Matrix, delta):
    jacob = [[0, 0], [0, 0]]
    newMatrix = Matrix
    initial = objective_function(y_train, y_target, Matrix)
    for i in range(0, len(jacob)):
        for j in range(0, len(jacob[0])):
            # print(f'initial: {initial}')
            newMatrix[i][j]=newMatrix[i][j]+delta
            after = objective_function(y_train, y_target, newMatrix)
            # print(f'after: {after}')
            jacob[i][j] = ((after - initial)/delta)
    '''
    :param input: (input type)
    :return data: (return type)
    '''

    return jacob
