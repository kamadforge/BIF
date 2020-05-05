import numpy as np

#2dim xor
def dataset1():

    n=100
    x=np.random.normal(0,1,[n,10])
    #print(x)
    py=np.exp(x[:,0]*x[:,1])
    #print(py)

#orange skin
def dataset2():

    n = 100
    x = np.random.normal(0, 1, [n, 10])
    #print(x)
    py = np.exp(x[:,0]**2+ x[:,1]**2+x[:,2]**2+x[:,3]**3-4)
    print(py)

#nonlinear additive model
def dataset3():
    n = 100
    x = np.random.normal(0, 1, [n, 10])
    #print(x)
    py = np.exp(-100*np.sin(2*x[:,0])+2*np.abs(x[:,1])+x[:,2]+np.exp(-x[:,3]))

#switch feature
def dataset4():

    n = 100
    choice = np.random.choice(2, n)
    x = np.random.normal(0, 1, [n, 9])
    y=np.zeros(n)

    for ind, i in enumerate(choice):
        if i==0:
            x1=np.random.normal(-3,1)
            x[ind,0]=x1
            y[ind]=np.exp(x[ind,1]**2+ x[ind,2]**2+x[ind,3]**2+x[ind,4]**3-4) #orange
        else:
            x2=np.random.normal(3,1)
            x[ind,0]=x2
            y[ind]=np.exp(-100*np.sin(2*x[ind,5])+2*np.abs(x[ind,6])+x[ind,7]+np.exp(-x[ind,8])) #nonlinear



dataset1()