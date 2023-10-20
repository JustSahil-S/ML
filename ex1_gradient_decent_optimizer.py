import numpy as np
import random
def f(x):
    '''
    :param x: (array-like) function inputs

    :return value: (float) function evaluation at x
    '''
    return np.exp(-.65987596/2*x)*np.cos(x*np.sqrt(1-np.power(.65987596*.5, 2)))




def g(x):
    '''
    :param x: (array-like) function inputs

    :return value: (float) function evaluation at x
    '''
    return 20*np.sin(x)+np.power(x, 2)


def h(x):
    '''
    :param x: (array-like) function inputs

    :return value: (float) function evaluation at x
    '''
    return np.sin(x)

if __name__ == "__main__":
    '''
    Initial parameters
    '''
    alpha = """ NOTE: learning rate """
    x0 =  """ NOTE: Initial guess """


def GuessCheck():
    fxmin = True
    counter = 0
    for x0 in np.arange(-6, 6, 0.1):
        fvalue = f(x0)
        print(fvalue)
        counter += 1
        if fvalue <= fxmin:
            fxmin = fvalue
            xmin = x0
    print(f'Absolute Minimum: x = {xmin} f(x0) = {fxmin}')
    print(f'Counter: {counter}')
    print((f(xmin+0.01)-f(xmin)))

""" 
Guess and Check Optimization:
"""
def GradientDescent():    
    alpha = 0.5
    h = 0.01
    counter = 0
    x0 = random.randrange(-6, 6)
    guess = x0
    while True:
        x0grad = ((f(x0 + h) - f(x0))/h)
        x0newmin = x0-(alpha*x0grad)
        if f(x0newmin) < f(x0):
            x0 = x0newmin
            print(x0)
            counter += 1
        else:
            x0old = x0+(alpha*x0grad)
            break
    print(f'Absolute Minimum: x = {x0} f(x) = {f(x0)}')
    print(f'Counter: {counter}, initGuess{guess}')
    print(f(x0old)-f(x0))

""" 
Gradient Descent Optimizer:
"""
GradientDescent()