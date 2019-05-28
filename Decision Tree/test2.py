import matplotlib.pyplot as plt 
import numpy as np 

# Functions
def f1(x):
    return 4*(x[0]-100) ** 2 + (x[1]-100) ** 2

def f2(x):
    return (x[0]-0.5) ** 2 + (x[1]-0.5) ** 2

def gradient_f1(x):
    return np.array([4 * 2 * (x[0]-100),2 * (x[1]-100)])

def gradient_f2(x):
    return np.array([2 * (x[0]-0.5),2 * (x[1]-0.5)])
x1 = [np.array([32,282])]
alpha=0.17
i = 0
while True:
    x1.append(x1[i] - alpha*gradient_f1(x1[i]))
    if (np.abs(x1[i+1] - x1[i])<1).all():
        break
    i += 1

x2 = [np.array([0.32,0.705])]
i = 0
while True:
    x2.append(x2[i] - alpha * gradient_f2(x2[i]))
    print(np.abs(x2[i+1] - x2[i]))
    if (np.abs(x2[i+1] - x2[i])<0.01).all():
        break
    i += 1


x1 = np.asarray(x1)
x2 = np.asarray(x2)

plt.subplot(121,aspect=1)
plt.title('Before 0-1 Nomalization')
xx,yy = np.meshgrid(np.linspace(0,200,1000),np.linspace(-100,300,1000))
plt.contour(xx,yy,f1([xx,yy]),12,colors='k',extend='both')
plt.plot(x1[:,0],x1[:,1],'ko-')

plt.subplot(122,aspect=1)
plt.title('After 0-1 Nomalization')
xx,yy = np.meshgrid(np.linspace(0,1,100),np.linspace(0,1,100))
plt.contour(xx,yy,f2([xx,yy]),12,colors='k',extend='both')
plt.plot(x2[:,0],x2[:,1],'ko-')
plt.show()

