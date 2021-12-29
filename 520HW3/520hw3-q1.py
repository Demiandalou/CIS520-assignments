import pickle
with open('hw3_q1.pkl' , 'rb') as f:
    data = pickle.load(f)
print(type(data))
print(data.keys())
print(data['x'].shape)
print(data['y'].shape)

def L2_error(y, y_hat):
    L2_error_ =  sum((y[i] - y_hat[i])**2 for i in range(len(y))) / len(y)
    return L2_error_
import numpy as np
x = data['x'] # 50,3
y = data['y']
# print(x[:,1,2].reshape((50,2)).shape)
print(np.concatenate((x[:,0].reshape((50,1)),x[:,2].reshape((50,1))),axis = 1).shape)
# q1.1
w11 = np.linalg.inv(x.T @ x) @ x.T @ y
print('q1.1, w=',w11)

lambda_ = 2
w = np.linalg.inv(x.T @ x + lambda_*np.eye(x.shape[1])) @ x.T @ y # 
print('q1.2, w=',w)

from sklearn.linear_model import Lasso
clf = Lasso(alpha=1/2/len(y))
clf.fit(x,y)
print('q1.3,',clf.coef_)

print('\n\n')

alpha = [[0,0,0],[1,0,0],[0,1,0],[0,0,1],
         [1,1,0],[1,0,1],[0,1,1],[1,1,1],]
all_x = [None,
    x[:,0].reshape((50,1)),x[:,1].reshape((50,1)),x[:,2].reshape((50,1)),
    x[:,0:2].reshape((50,2)),np.concatenate((x[:,0].reshape((50,1)),x[:,2].reshape((50,1))),axis = 1),
    x[:,1:3].reshape((50,2)), x,
]

from sklearn.linear_model import LinearRegression
w = [0,0,0]
tmpwx = [np.mean(y) for i in range(len(y))]
a = alpha[0]
err = L2_error(tmpwx, y) + sum(a)*lambda_/len(y)
# print(tmpwx,err)
print(a,' & ',w,' & ',round(err[0],3),'\\\\ \\hline')

alpha = np.array(alpha)
errs = []
for i in range(1,len(alpha)):
    a = alpha[i]
    x = all_x[i]
    # print(x,type(x))
    # print(y,type(y))
    # w = np.linalg.inv(x.T @ x + lambda_*a) @ x.T @ y
    reg = LinearRegression()
    reg.fit(x,y)
    w = reg.coef_
    w = [round(i,3) for i in list(w.flatten())]
    # err = L2_error(x @ w , y)
    err = L2_error(x @ w , y) + sum(a)*lambda_/len(y)
    print(a,' & ',w,' & ',round(err[0],3),'\\\\ \\hline')
    errs.append(err[0])
print(min(errs))


exit()
# 6a
tmp1 = sum([i**2 for i in w11])
tmp2 = sum([j**2 for j in (y-x@w11)])
ratio = tmp1/tmp2
print('q1.6a', ratio)

# 6c
for lambda_ in range(10):
    w = np.linalg.inv(x.T @ x + lambda_*np.eye(x.shape[1])) @ x.T @ y
    tmp1 = sum([i**2 for i in w])
    tmp2 = sum([i**2 for i in w11])
    print('lambda',lambda_,'result',tmp1/tmp2)
