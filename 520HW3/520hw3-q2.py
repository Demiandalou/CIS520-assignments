# Streamwise regression.
import numpy as np
import copy
from sklearn.linear_model import LinearRegression
# x = [[2,1,1],[4,2,1],[3.3,1.3,5.8]]
origin_x = [[2,4,3.3],[1,2,1.3],[1,1,5.8]]
y = [[5,8,2]]
origin_x = np.array(origin_x)
y = np.array(y)
y = y.T

lambda_ = 0.2

def L2_error(y, y_hat):
    L2_error_ =  sum((y[i] - y_hat[i])**2 for i in range(len(y)))
    return L2_error_
FIRST = 1
SECOND = 1
THIRD = 0

w = [0,0,0]
# tmpwx = [np.mean(y) for i in range(len(y))]
tmpwx = [np.mean(y) for i in range(len(y))]
err = L2_error(tmpwx, y) + 0*lambda_/len(y)
# print(tmpwx,err)
print(w)
print(err)

print('only x1')
x = origin_x[:,0]
x=x.reshape(-1,1)
y=y.reshape(-1,1)
# w = np.linalg.inv(x.T @ x + lambda_*np.array([1,0,0])) @ x.T @ y
reg = LinearRegression()
reg.fit(x,y)
w = reg.coef_
print(w)
y_pred=reg.predict(x)
err = L2_error(y_pred, y) + sum([1,0,0])*lambda_ # SSE
print(err)

# x = origin_x[:,0:2]
# # x=x.reshape(-1,1)
# y=y.reshape(-1,1)
# # w = np.linalg.inv(x.T @ x + lambda_*np.array([1,0,0])) @ x.T @ y
# reg = LinearRegression()
# reg.fit(x,y)
# w = reg.coef_
# print(w)
# y_pred=reg.predict(x)
# err = L2_error(y_pred, y) + sum([1,1,0])*lambda_ # SSE
# print(err)
print('only x2')
x = origin_x[:,1]
x=x.reshape(-1,1)
y=y.reshape(-1,1)
# w = np.linalg.inv(x.T @ x + lambda_*np.array([1,0,0])) @ x.T @ y
reg = LinearRegression()
reg.fit(x,y)
w = reg.coef_
print(w)
y_pred=reg.predict(x)
err = L2_error(y_pred, y) + sum([1,1,0])*lambda_ # SSE
print(err)

print('x2, x3')
x = origin_x[:,1:3]
# x=x.reshape(-1,1)
y=y.reshape(-1,1)
# w = np.linalg.inv(x.T @ x + lambda_*np.array([1,0,0])) @ x.T @ y
reg = LinearRegression()
reg.fit(x,y)
w = reg.coef_
print(w)
y_pred=reg.predict(x)
err = L2_error(y_pred, y) + sum([0,1,1])*lambda_ # SSE
print(err)
# FINAL FEATURE 1,2

print('--------------------------------')

print('only x2')
x = origin_x[:,1]
x=x.reshape(-1,1)
y=y.reshape(-1,1)
# w = np.linalg.inv(x.T @ x + lambda_*np.array([1,0,0])) @ x.T @ y
reg = LinearRegression()
reg.fit(x,y)
w = reg.coef_
print(w)
y_pred=reg.predict(x)
err = L2_error(y_pred, y) + sum([0,1,0])*lambda_ # SSE
print(err)

print('x2,x3')
x = origin_x[:,1:3]
# x=x.reshape(-1,1)
y=y.reshape(-1,1)
# w = np.linalg.inv(x.T @ x + lambda_*np.array([1,0,0])) @ x.T @ y
reg = LinearRegression()
reg.fit(x,y)
w = reg.coef_
print(w)
y_pred=reg.predict(x)
err = L2_error(y_pred, y) + sum([0,1,1])*lambda_ # SSE
print(err)

print('x1,x2,x3')
x = origin_x[:,:]
# x=x.reshape(-1,1)
y=y.reshape(-1,1)
# w = np.linalg.inv(x.T @ x + lambda_*np.array([1,0,0])) @ x.T @ y
reg = LinearRegression()
reg.fit(x,y)
w = reg.coef_
print(w)
y_pred=reg.predict(x)
err = L2_error(y_pred, y) + sum([1,1,1])*lambda_ # SSE
print(err)
# FINAL FEATURE 2,3
exit()

def get_x(a,x):
    tmp = []
    for i in range(len(a)):
        if i==1:
            tmp.append(x[i])
    return np.array(tmp)


a = np.array([1,0,0])
tmpx = x[:1]
reg = LinearRegression()
reg.fit(tmpx.T,y)
w = reg.coef_
err = L2_error(np.array(tmpx[0])*w[0][0], y) + sum(a)*lambda_ # SSE
print(a,' & ',w,' & ',err,'\\\\ \\hline')

if THIRD:
    a = np.array([0,0,0])
    order = [0,1,2]
    err = L2_error(np.zeros(y.shape) , y) + sum(a)*lambda_ # SSE
    # w = [round(i,3) for i in list(w.flatten())]
    w = [0,0,0]
    print(a,' & ',w,' & ',round(err[0],3),'\\\\ \\hline')
    preverr = err

    for i in range(3):
        tmpa = copy.deepcopy(a)
        tmpa[order[i]] = 1
        print('tmpa',tmpa)
        tmpx = get_x(tmpa,x)
        # print(x)
        # print(y)
        reg = LinearRegression()
        reg.fit(tmpx.T,y)
        w = reg.coef_
        print('w',w)
        print('tmpx',tmpx)
        # w = np.linalg.inv(x.T @ x + lambda_*a) @ x.T @ y
        err = L2_error(np.array(tmpx[0])*w[0][0], y) + sum(tmpa)*lambda_ # SSE
        w = [round(i,3) for i in list(w.flatten())]
        # print(tmpa,' & ',w,' & ',round(err[0],3),'\\\\ \\hline')
        print(tmpa,' & ',w,' & ',err,'\\\\ \\hline')
        if err < preverr:
            preverr = err
            a = tmpa
    print(a)
    exit()

    # q2-1b
    a = np.array([0,0,0])
    order = [1,2,0]
    err = L2_error(np.zeros(y.shape) , y) + sum(a)*lambda_ # SSE
    # w = [round(i,3) for i in list(w.flatten())]
    w = [0,0,0]
    print(a,' & ',w,' & ',round(err[0],3),'\\\\ \\hline')
    preverr = err
    for i in range(3):
        tmpa = copy.deepcopy(a)
        tmpa[order[i]] = 1
        # print(a)
        w = np.linalg.inv(x.T @ x + lambda_*a) @ x.T @ y
        err = L2_error(x @ w , y) + sum(a)*lambda_ # SSE
        w = [round(i,3) for i in list(w.flatten())]
        # print(tmpa,' & ',w,' & ',round(err[0],3),'\\\\ \\hline')
        print(tmpa,' & ',w,' & ',err,'\\\\ \\hline')
        if err < preverr:
            preverr = err
            a = tmpa
    print(a)



