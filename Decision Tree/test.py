import matplotlib.pyplot as plt 
import numpy as np 

# Functions
def f1(x):
    return x[0]**2+x[1]**2

def gradient_f1(x):
    return np.array([2*x[0],2*x[1]])


# Inital Point

x1 = [np.array([0.92,0.84])]
x2 = [np.array([92,0.84])]

alpha=0.1
i = 0
while True:
    x1.append(x1[i] - alpha*gradient_f1(x1[i]))
    if (np.abs(x1[i+1] - x1[i])<0.01).all():
        break
    i += 1
print(i)

alpha= 0.1
i = 0
while True:
    x2.append(x2[i] - alpha*gradient_f1(x2[i]))
    if (np.abs(x2[i+1] - x2[i])<0.01).all():
        break
    i += 1
print(i)

x1 = np.asarray(x1)
x2 = np.asarray(x2)
# Figure
plt.subplot(122,aspect=1.0)
xx,yy = np.meshgrid(np.linspace(-1,1,100),np.linspace(-1,1,100))
plt.contour(xx,yy,f1([xx,yy]),colors='k',exntend='both')
plt.plot(x1[:,0],x1[:,1],'ko-')

plt.subplot(121,aspect=1)
xx,yy = np.meshgrid(np.linspace(-100,100,100),np.linspace(-100,100,100))
plt.contour(xx,yy,f1([xx,yy]),colors='k',exntend='both')
plt.plot(x2[:,0],x2[:,1],'ko-')

plt.show()