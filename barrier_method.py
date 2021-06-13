import numpy as np

def f(x,Q,p,A,b,t):
    return t*(np.dot(x.T,np.dot(Q,x)) + np.dot(p.T,x)) - np.sum(np.log(-np.dot(A,x)+b))

def df(x,Q,p,A,b,t):
    return t*np.dot(Q+Q.T,x) + t*p - np.expand_dims(np.sum((A/(np.dot(A,x)-b)),axis=0),axis=1)

def df2(x,Q,p,A,b,t):
    rst = t*(Q+Q.T)
    for i in range(A.shape[0]):
        A_i = np.expand_dims(A[i],axis=1)
        rst += np.dot(A_i.T,A_i)/(np.dot(A[i],x)-b[i])**2
    return rst

def backtracking(x,dx,dfx,Q,p,A,b,t,alpha,beta):
    st = 1
    while f(x+st*dx,Q,p,A,b,t)>=f(x,Q,p,A,b,t)+alpha*st*np.dot(dfx.T,dx):
        st = beta*st
    return st
    
def centering_step(Q,p,A,b,t,v0,eps,iter_max=1000):
    v_i = [v0]
    alpha = 1/4
    beta = 1-1e-2
    lambda_x2 = 5
    x = v0.copy()
    it = 0
    while lambda_x2 > 2*eps and it<iter_max:
        dfx = df(x,Q,p,A,b,t)
        df2x = df2(x,Q,p,A,b,t)
        dx = np.linalg.solve(df2x,-dfx)
        st = backtracking(x,dx,dfx,Q,p,A,b,t,alpha,beta)
        x += st*dx
        v_i.append(x.copy())
        lambda_x2 = np.dot(-dfx.T,dx)
        it += 1
    return v_i

def barrier_method(Q,p,A,b,v0,eps,mu=2,iter_max=1000):
    """
        Solve the convex problem min x.T Q x + x.T p, s.t Ax = b using barrier method with precision eps, and updating factor mu.
    """
    v_i = [v0]
    m = len(b)
    t = 1
    it = 0
    v_star = v0.copy()
    while m/t >= eps and it<iter_max:
        v_star = centering_step(Q,p,A,b,t,v_star,eps)[-1]
        v_i.append(v_star)
        t = mu*t
    return v_i