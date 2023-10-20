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
1.) Minimum value of f(x) is -3.18 at x = -3.7
2.) 120 guesses
3.) Yes, because the function iterates over every point in the range so 
    at a small enough step size, it will find the absolute minimum every 
    time with enough iterations but it has to check every spot in the function.
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
1.) Minimum value of f(x) is -3.18 at x = -3.68
2.) The value is not consistant due to the random number but it is generally between 5 - 10 iterations when the number 
is already close but it can go up to 140 iterations
3.) Yes, when the initial guess is a positive number, the function stops at x=2.966 where f(x) is -0.35. After graphing the function, 
    I realized it was because there is a local minimum on the positive axis which the gradient descent stops at.
4.) When my learning rate (alpha) is lower, there are more iterations of the loop until a minimum is found
5.) Yes, when I change the value of alpha, the value of x and f(x) are very slightly different, this is 
    probably because it gets more accurate as the steps become smaller.
6.) No, this is because depending on the number I generate, the minimum value changes 
    meaning there might be a relative minimum that the function is finding. This implies that ML models might not always
    be fully optimized with gradient descent and still have room to improve as the function has only hit a relative minimum and not the absolute minimum where it would have
     the least error. 
"""

GradientDescent()