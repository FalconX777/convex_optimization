import numpy as np

## Descent method

def descent_method(x_0=None, descent_direction=None, line_search=None, criterion=None, iter_max=1000, **kwargs):
  """
    Inputs:
      x_0:          in Dom(f) of shape (dim,)
      descent_direction: function returning the descent direction of shape (dim,) with input x of shape (dim,)
      line_search:  function returning the step size (float) t>0 with input x and delta_x of shapes (dim,) 
      criterion:    function returning a boolean (False to continue the iterations, True to break) with input prev_x, prev_delta_x, x, delta_x of shapes (dim,) 
      iter_max:     int, max number of iterations
    Outputs:
      x:            shape (dim,)
  """
  dim = x_0.shape[0]
  n_iter = 0

  if x_0 == None:
    prev_x = np.zeros((dim,))
  else:
    prev_x = x_0
  prev_delta_x = direction_descent(prev_x)
  t = line_search(prev_x, prev_delta_x)
  x = prev_x - t*prev_delta_x
  delta_x = direction_descent(x)

  while not criterion(prev_x, prev_delta_x, x, delta_x) and n_iter<iter_max:
    prev_x = x
    prev_delta_x = delta_x
    t = line_search(prev_x, prev_delta_x)
    x = prev_x - t*prev_delta_x
    delta_x = direction_descent(x)
  
  return x


# General backtracking line search

def backtracting_line_search(f,grad_f,alpha,beta):
  """
    Return the general backtracking line search function
  """
  def backtracking_function(x, delta_x):
    t = 1
    while f(x+t*delta_x) >= f(x) + alpha*t*(grad_f(x)*delta_x).sum():
      t *= beta
    return t
  return backtracking_function



# Gradient descent
  
def gradient_backtracking_line_search(f,alpha,beta):
  """
    Return the backtracking line search function used in gradient descent algorithm
  """
  def gradient_backtracking_function(x, delta_x):
    t = 1
    while f(x+t*delta_x) >= f(x) - alpha*t*(delta_x**2).sum():
      t *= beta
    return t
  return gradient_backtracking_function

def gradient_descent_direction(grad_f):
  """
    Return the update direction function used in the constrained gradient descent algorithm
  """
  def gradient_direction_function(x):
    return -grad_f(x)
  return gradient_direction_function

def gradient_criterion(epsilon):
  """
    Return the criterion function used in the constrained gradient descent algorithm
  """
  def gradient_criterion_function(prev_x, prev_delta_x, x, delta_x):
    return (delta_x**2).sum()<=epsilon
  return gradient_criterion_function


# Newton descent

def newton_backtracking_line_search(f,alpha,beta):
  """
    Return the backtracking line search function used in Newton descent algorithm
  """
  def newton_backtracking_function(x, delta_x):
    t = 1
    while f(x+t*delta_x) >= f(x) - alpha*t*(delta_x**2).sum():
      t *= beta
    return t
  return newton_backtracking_function

def newton_descent_direction(grad_f, hess_f):
  def newton_direction_function(x):
    return np.linalg.lstsq(-hess_f(x), grad_f(x))
  return newton_direction_function

def newton_criterion(grad_f, epsilon):
  def newton_criterion_function(prev_x, prev_delta_x, x, delta_x):
    return -(grad_f(x),delta_x).sum()/2<=epsilon
  return newton_criterion_function


# Constrained Descent s.t Ax = b, A of shape (p,q) and rk(A) = p, with Newton method, on y = (x \nu) of shape (q + p,)

def constrained_newton_backtracking_line_search(grad_f,A,b,alpha,beta):
  """
    Return the backtracking line search function used in constrained Newton descent algorithm (s.t Ax = b, A of shape (p,q) and rk(A) = p)
  """
  def constrained_newton_backtracking_function(y, delta_y):
    def r(y):
      x = y[:A.shape[1]]
      nu = y[A.shape[1]:]
      return np.concatenate([grad_f(x)+A.T@nu, A@x-b], axis=0)
    t = 1
    r_delta = r(y+t*delta_y)
    r_y_2 = np.sqrt((r(y)**2).sum())
    while np.sqrt((r_delta**2).sum()) > (1-alpha*t)*r_y_2:
      t *= beta
      r_delta = r(y+t*delta_y)
    return t
  return newton_backtracking_function

def constrained_newton_descent_direction(grad_f, hess_f, A, b):
  """
    Return the update direction function used in the constrained Newton descent algorithm (s.t Ax = b, A of shape (p,q) and rk(A) = p)
  """
  def constrained_newton_direction_function(y):
    x = y[:A.shape[1]]
    nu = y[A.shape[1]:]
    H_1 = np.concatenate([hess_f(x), A.T], axis=1)
    H_2 = np.concatenate([A, np.zeros(A.T.shape)], axis=1)
    H = np.concatenate([H_1, H_2], axis=0)
    w = np.concatenate([-grad_f(x)+A.T@nu, -A@x+b], axis=0)
    return np.linalg.lstsq(H, w)
  return constrained_newton_direction_function

def constrained_newton_criterion(grad_f,A,b,epsilon):
  """
    Return the criterion function used in the constrained Newton descent algorithm (s.t Ax = b, A of shape (p,q) and rk(A) = p)
  """
  def constrained_newton_criterion_function(prev_y, prev_delta_y, y, delta_y):
    def r(y):
      x = y[:A.shape[1]]
      nu = y[A.shape[1]:]
      return np.concatenate([grad_f(x)+A.T@nu, A@x-b], axis=0)
    return np.sqrt((r(y)**2).sum())<=epsilon
  return newton_criterion_function