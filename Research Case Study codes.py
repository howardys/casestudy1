#!/usr/bin/env python
# coding: utf-8

#    # Supplement codes for Case study 1: Circuit QED

# ## Author: Yi Shi

# In[11]:


from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import schemdraw
import schemdraw.elements as elm

import warnings
warnings.filterwarnings("ignore")


# In[12]:


bloch_sphere=qutip.Bloch()

bloch_sphere.point_color = ['r','g','y','black','b','orange']
bloch_sphere.point_size = [70,70,70]
bloch_sphere.add_points([1,0,0])
bloch_sphere.add_points([-1,0,0])
bloch_sphere.add_points([0,1,0])
bloch_sphere.add_points([0,-1,0])
bloch_sphere.add_points([0,0,1])
bloch_sphere.add_points([0,0,-1])
bloch_sphere.zlabel = [r'$\left|{C}_{\alpha}^{+}\right\rangle$',r'$\left|{C}_{\alpha}^{-}\right\rangle$']
bloch_sphere.xlabel = [r'$|\alpha\rangle$',r'$|-\alpha\rangle$']
bloch_sphere.ylabel = [r'$\left|{C}_{\alpha}^{-i}\right\rangle$',r'$\left|{C}_{\alpha}^{+i}\right\rangle$']
bloch_sphere.title=['x']
bloch_sphere.view=[-60,30]
bloch_sphere.show()


# In[13]:


bloch_sphere=qutip.Bloch()

bloch_sphere.point_color = ['r','g','y','black','b','orange']
bloch_sphere.point_size = [70,70,70]
bloch_sphere.add_points([1,0,0])
bloch_sphere.add_points([-1,0,0])
bloch_sphere.add_points([0,1,0])
bloch_sphere.add_points([0,-1,0])
bloch_sphere.add_points([0,0,1])
bloch_sphere.add_points([0,0,-1])
bloch_sphere.zlabel = [r'$|S_z;+\rangle$',r'$|S_z;-\rangle$']
bloch_sphere.xlabel = [r'$|S_x;+\rangle$',r'$|S_x;-\rangle$']
bloch_sphere.ylabel = [r'$|S_y;+\rangle$',r'$|S_y;-\rangle$']
bloch_sphere.title=['x']
bloch_sphere.show()


# In[3]:


def catstate(N_in,alpha_in):
    coeff = np.exp((-abs(alpha_in)**2)/2)
    N_term = np.linspace(0,N_in-1,N_in)
    N_term = [int(i) for i in N_term]
    sum_term = [((alpha_in**i)/np.sqrt(np.math.factorial(i)))*basis(N_in,i) for i in N_term]
    state = coeff*sum(sum_term)
    state = state.unit()
    return state


N=20
eps_2 = 2*np.pi*17.75
K = 2*np.pi*6.7
a = destroy(N)
adag = create(N)
ini_state = basis(N,0)
cat_state_plus = (catstate(N,np.sqrt(eps_2/K))+catstate(N,-np.sqrt(eps_2/K))).unit()

H_initialize = -K*(adag**2)*(a**2)+eps_2*(adag**2 - a**2)
times = np.linspace(0,np.pi/2/eps_2,1000)
results = sesolve(H_initialize,ini_state,times,[ket2dm(cat_state_plus),ket2dm(ini_state)])
times = [i*1E3 for i in times]


# In[4]:


plt.figure(dpi=300)
plt.plot(times,results.expect[0],label=r'$\langle C_{\alpha}^+|\rho|C_{\alpha}^+ \rangle$',color='r')
plt.plot(times,results.expect[1],label=r'$\langle 0|\rho|0\rangle$',color='b')
plt.hlines(results.expect[0][-1],0,14,linestyle='dashed',linewidth=0.5)
plt.yticks([0,0.25,0.5,0.75,round(results.expect[0][-1],2),1])
plt.ylim(0,1)
plt.xlim(0,14)
plt.xlabel('Time (ns)',fontsize=18)
plt.ylabel('Fidelity',fontsize=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(loc=4)
plt.show()


# In[5]:


H_cat = -K*(adag**2)*(a**2)+eps_2*(adag**2 + a**2)
H_cat_1 = -K*(adag**2)*(a**2)
times = np.linspace(0,1,1000)
results1 = sesolve(H_cat,cat_state_plus,times,[ket2dm(cat_state_plus)])
results2 = sesolve(H_cat_1,cat_state_plus,times,[ket2dm(cat_state_plus)])


# In[6]:


plt.figure(dpi=300)
plt.plot(times,results1.expect[0],label='With stabilization pulse',color='r',linewidth=0.8)
plt.plot(times,results2.expect[0],label='Without stabilization pulse',color='b',linewidth=0.8)
plt.xlabel('Time '+r'$(\mu s)$',fontsize=18)
plt.ylabel('Fidelity '+r'$\langle C_{\alpha}^+ |\rho|C_{\alpha}^+ \rangle$',fontsize=18)
plt.ylim(0,)
plt.xlim(0,1)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(loc=4)
plt.show()


# In[7]:


N=20
eps_2 = 2*np.pi*17.75E6
K = 2*np.pi*6.7E6
a = destroy(N)
adag = create(N)

H_cat_1 = -K*(adag**2)*(a**2)+eps_2*(adag**2 + a**2)
H_cat_2 = -K*(adag**2)*(a**2)


# In[8]:


Re_a = np.linspace(-2.5,2.5,100)
Im_a = np.linspace(-1.5,1.5,80)

def Energy_cat(N_in,alpha_in,H):
    coeff = np.exp((-abs(alpha_in)**2)/2)
    N_term = np.linspace(0,N_in-1,N_in)
    N_term = [int(i) for i in N_term]
    sum_term = [((alpha_in**i)/np.sqrt(np.math.factorial(i)))*basis(N_in,i) for i in N_term]
    state = coeff*sum(sum_term)
    state = state.unit()
    return (expect(H,state)/K)


# In[9]:


E1c1 = np.zeros((100,80))
E1c2 = np.zeros((100,80)) 
i = 0
while i < 100:
    j = 0
    while j < 80:
        alpha = Re_a[i] + Im_a[j]*1J
        E1c1[i,j] = Energy_cat(N,alpha,H_cat_1)
        E1c2[i,j] = Energy_cat(N,alpha,H_cat_2)
        j += 1
    i += 1


# In[10]:


X,Y = np.meshgrid(Im_a,Re_a)
fig, ax = plt.subplots(dpi=300)
c = ax.pcolormesh(Y,X,E1c1,cmap='viridis')
ax.axis([Y.min(),Y.max(),X.min(),X.max()])
fig.colorbar(c,ax=ax)
plt.xlabel('Re(a)',fontsize=18)
plt.ylabel('Im(a)',fontsize=18)
plt.title(r'$E/\hbar K \quad(\epsilon_2/2\pi=17.75MHz)$',fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.text(-3.8,1.6,'a',fontsize=20)
plt.show()


# In[11]:


X,Y = np.meshgrid(Im_a,Re_a)
fig, ax = plt.subplots(dpi=300)
c = ax.pcolormesh(Y,X,E1c2,cmap='viridis')
ax.axis([Y.min(),Y.max(),X.min(),X.max()])
fig.colorbar(c,ax=ax)
plt.xlabel('Re(a)',fontsize=18)
plt.ylabel('Im(a)',fontsize=18)
plt.title(r'$E/\hbar K \quad(\epsilon_2=0)$',fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.text(-3.8,1.6,'b',fontsize=20)
plt.show()


# In[12]:


def catstate(N_in,alpha_in):
    coeff = np.exp((-abs(alpha_in)**2)/2)
    N_term = np.linspace(0,N_in-1,N_in)
    N_term = [int(i) for i in N_term]
    sum_term = [((alpha_in**i)/np.sqrt(np.math.factorial(i)))*basis(N_in,i) for i in N_term]
    state = coeff*sum(sum_term)
    state = state.unit()
    return state

n = 2.6
K = 2*np.pi*6.7
eps_2 = 15.5*2*np.pi
eps_x_abs = 0.74*2*np.pi

arg = np.linspace(-np.pi,np.pi,200)
eps_x = [eps_x_abs*np.exp(i*1J) for i in arg]

ini_state = basis(N,0)
catstate_1 = catstate(N,np.sqrt(eps_2/K))+ catstate(N,-np.sqrt(eps_2/K))
catstate_1 = catstate_1.unit()

times = np.linspace(0,1,10000)


P0 = np.zeros((10000,200))
P0cat = np.zeros((10000,200))

ini_state = basis(N,0)


i = 0
while i < 200:
    H_s = -K*(adag**2)*(a**2)+eps_2*(adag**2+a**2) + eps_x[i]*adag + np.conj(eps_x[i])*a
    results=sesolve(H_s,catstate_1,times,[ket2dm(catstate_1)])
    P0cat[:,i] = results.expect[0]
    i += 1
    
arg = [i/np.pi for i in arg]


# In[13]:


X,Y = np.meshgrid(arg,times)
fig, ax = plt.subplots(dpi=300)
c = ax.pcolormesh(Y,X,P0cat,cmap='cividis')
ax.axis([Y.min(),Y.max(),X.min(),X.max()])
fig.colorbar(c,ax=ax)
plt.hlines(arg[100],0,1,linestyle='dashed',linewidth=1,color='g')
plt.hlines(arg[125],0,1,linestyle='dashed',linewidth=1,color='b')
plt.hlines(arg[150],0,1,linestyle='dashed',linewidth=1,color='r')
plt.xlabel('Time t '+r'$(\mu s)$')
plt.ylabel('Drive phase, arg'+r'$(\epsilon_x)/\pi$')
plt.title(r'$P (|C_{\alpha}^+\rangle)$')
plt.show()


# In[14]:


plt.figure(dpi=300,figsize=(2,4))
plt.subplot(3, 1, 1)
plt.plot(times,P0cat[:,150],color='r')
plt.xticks([])
plt.yticks([0,1])
plt.ylabel(r'$P (|C_{\alpha}^+\rangle)$')
plt.xlim(0,1)

plt.subplot(3, 1, 2)
plt.plot(times,P0cat[:,125],color='b')
plt.xticks([])
plt.yticks([0,1])
plt.ylabel(r'$P (|C_{\alpha}^+\rangle)$')
plt.xlim(0,1)

plt.subplot(3, 1, 3)
plt.plot(times,P0cat[:,100],color='g')
plt.yticks([0,1])
plt.ylabel(r'$P (|C_{\alpha}^+\rangle)$')
plt.xlim(0,1)
plt.xlabel('Time t '+r'$(\mu s)$')
plt.show()


# In[15]:


# set the parameters
K = 2*np.pi*6.7
eps_2 = 17.5*2*np.pi
eps_x = 6.5*2*np.pi
Omegax = np.sqrt(eps_2/K)*eps_x*4


# In[16]:


#states
axstate_1 = catstate(N,np.sqrt(eps_2/K))
axstate_2 = catstate(N,-np.sqrt(eps_2/K))

aystate_1 = axstate_1 - 1J*axstate_2
aystate_1 = aystate_1.unit()
aystate_2 = axstate_1 + 1J*axstate_2
aystate_2 = aystate_2.unit()

azstate_1 = axstate_1 + axstate_2
azstate_1 = azstate_1.unit()
azstate_2 = axstate_1 - axstate_2
azstate_2 = azstate_2.unit()


# In[17]:


#martix in phase space
X = axstate_1*axstate_1.dag() - axstate_2*axstate_2.dag()
Y = aystate_1*aystate_1.dag() - aystate_2*aystate_2.dag()
Z = azstate_1*azstate_1.dag() - azstate_2*azstate_2.dag()


# In[18]:


times = np.linspace(0,1,1000)


# In[19]:


H_s1 = -K*(adag**2)*(a**2)
H_s2 = eps_2*(adag**2+a**2)
#assume there is a pi phase in x pulse
H_x = eps_x*(a+adag)


# In[20]:


def coordinates(resultstate):
    return  expect(X,resultstate),expect(Y,resultstate),expect(Z,resultstate)


# In[21]:


bloch_sphere=qutip.Bloch()

bloch_sphere.point_color = ['r','g','y','black','b','orange']
bloch_sphere.point_size = [70,70,70]
bloch_sphere.add_points([1,0,0])
bloch_sphere.add_points([-1,0,0])
bloch_sphere.add_points([0,1,0])
bloch_sphere.add_points([0,-1,0])
bloch_sphere.add_points([0,0,1])
bloch_sphere.add_points([0,0,-1])
bloch_sphere.zlabel = [r'$|S_z;+\rangle$',r'$|S_z;-\rangle$']
bloch_sphere.xlabel = [r'$|S_x;+\rangle$',r'$|S_x;-\rangle$']
bloch_sphere.ylabel = [r'$|S_y;+\rangle$',r'$|S_y;-\rangle$']
bloch_sphere.title=['x']
bloch_sphere.show()
H = H_s1 + H_s2

resultsx1=sesolve(H,axstate_1,times)
resultsx2=sesolve(H,axstate_2,times)
resultsy1=sesolve(H,aystate_1,times)
resultsy2=sesolve(H,aystate_2,times)
resultsz1=sesolve(H,azstate_1,times)
resultsz2=sesolve(H,azstate_2,times)


# In[22]:


bloch_sphere=qutip.Bloch()

bloch_sphere.point_color = ['r','g','y','black','b','orange']
bloch_sphere.point_size = [70,70,70]
bloch_sphere.add_points([1,0,0])
bloch_sphere.add_points([-1,0,0])
bloch_sphere.add_points([0,1,0])
bloch_sphere.add_points([0,-1,0])
bloch_sphere.add_points([0,0,1])
bloch_sphere.add_points([0,0,-1])
bloch_sphere.zlabel = [r'$|S_z;+\rangle$',r'$|S_z;-\rangle$']
bloch_sphere.xlabel = [r'$|S_x;+\rangle$',r'$|S_x;-\rangle$']
bloch_sphere.ylabel = [r'$|S_y;+\rangle$',r'$|S_y;-\rangle$']
bloch_sphere.title=['x']
bloch_sphere.show()
bloch_sphere=qutip.Bloch()

bloch_sphere.point_color = ['r','g','y','black','b','orange']
bloch_sphere.point_size = [70,70,70]
bloch_sphere.add_points([1,0,0])
bloch_sphere.add_points([-1,0,0])
bloch_sphere.add_points([0,1,0])
bloch_sphere.add_points([0,-1,0])
bloch_sphere.add_points([0,0,1])
bloch_sphere.add_points([0,0,-1])
bloch_sphere.zlabel = [r'$|S_z;+\rangle$',r'$|S_z;-\rangle$']
bloch_sphere.xlabel = [r'$|S_x;+\rangle$',r'$|S_x;-\rangle$']
bloch_sphere.ylabel = [r'$|S_y;+\rangle$',r'$|S_y;-\rangle$']
bloch_sphere.title=['x']
bloch_sphere.show()
psi_1=coordinates(resultsx1.states[-1])
psi_2=coordinates(resultsx2.states[-1])
psi_3=coordinates(resultsy1.states[-1])
psi_4=coordinates(resultsy2.states[-1])
psi_5=coordinates(resultsz1.states[-1])
psi_6=coordinates(resultsz2.states[-1])

bloch_sphere=Bloch()
bloch_sphere.point_color = ['orange','g','y','orchid','r','b']
bloch_sphere.point_marker = ['s','s','^','^','^','^']
bloch_sphere.point_size = [170,170,170]
bloch_sphere.add_points([psi_1[0],psi_1[1],psi_1[2]])
bloch_sphere.add_points([psi_2[0],psi_2[1],psi_2[2]])
bloch_sphere.add_points([psi_3[0],psi_3[1],psi_3[2]])
bloch_sphere.add_points([psi_4[0],psi_4[1],psi_4[2]])
bloch_sphere.add_points([psi_5[0],psi_5[1],psi_5[2]])
bloch_sphere.add_points([psi_6[0],psi_6[1],psi_6[2]])
bloch_sphere.view = [-60,30]
bloch_sphere.zlabel = ['$+Z$','']
bloch_sphere.xlabel = ['$+X$','']
bloch_sphere.ylabel = ['$+Y$','']
bloch_sphere.title=['x']
bloch_sphere.show()


# In[23]:


dt = np.pi/2/Omegax

def tx(t,args):
    if 0.5 < t < 0.5+dt:
        return 1
    else:
        return 0
H=[H_s1+H_s2,[H_x,tx]]


# In[24]:


resultsx1=sesolve(H,axstate_1,times)
resultsx2=sesolve(H,axstate_2,times)
resultsy1=sesolve(H,aystate_1,times)
resultsy2=sesolve(H,aystate_2,times)
resultsz1=sesolve(H,azstate_1,times)
resultsz2=sesolve(H,azstate_2,times)


# In[25]:


psi_1=coordinates(resultsx1.states[-1])
psi_2=coordinates(resultsx2.states[-1])
psi_3=coordinates(resultsy1.states[-1])
psi_4=coordinates(resultsy2.states[-1])
psi_5=coordinates(resultsz1.states[-1])
psi_6=coordinates(resultsz2.states[-1])

bloch_sphere=Bloch()
bloch_sphere.fig
bloch_sphere.point_color = ['orange','g','y','orchid','r','b']
bloch_sphere.point_marker = ['s','s','^','^','^','^']
bloch_sphere.point_size = [170,170,170]
bloch_sphere.add_points([psi_1[0],psi_1[1],psi_1[2]])
bloch_sphere.add_points([psi_2[0],psi_2[1],psi_2[2]])
bloch_sphere.add_points([psi_3[0],psi_3[1],psi_3[2]])
bloch_sphere.add_points([psi_4[0],psi_4[1],psi_4[2]])
bloch_sphere.add_points([psi_5[0],psi_5[1],psi_5[2]])
bloch_sphere.add_points([psi_6[0],psi_6[1],psi_6[2]])
bloch_sphere.view = [-60,30]
bloch_sphere.zlabel = ['$+Z$','']
bloch_sphere.xlabel = ['$+X$','']
bloch_sphere.ylabel = ['$+Y$','']
bloch_sphere.title=['x']
bloch_sphere.show()


# In[26]:


phi = -np.pi/2
dt = np.pi/2/K

def tz(t,args):
    if 0.3 < t < 0.3+dt:
        return 0
    elif 0.3+dt < t:
        return np.exp(1J*2*phi)
    else:
        return 1


H=[H_s1,[H_s2,tz]]


# In[27]:


#states
axstate_1 = catstate(N,np.sqrt(eps_2/K))
axstate_2 = catstate(N,-np.sqrt(eps_2/K))

aystate_1 = axstate_1 - 1J*axstate_2
aystate_1 = aystate_1.unit()
aystate_2 = axstate_1 + 1J*axstate_2
aystate_2 = aystate_2.unit()

azstate_1 = axstate_1 + axstate_2
azstate_1 = azstate_1.unit()
azstate_2 = axstate_1 - axstate_2
azstate_2 = azstate_2.unit()


# In[28]:


resultsx1=sesolve(H,axstate_1,times)
resultsx2=sesolve(H,axstate_2,times)
resultsy1=sesolve(H,aystate_1,times)
resultsy2=sesolve(H,aystate_2,times)
resultsz1=sesolve(H,azstate_1,times)
resultsz2=sesolve(H,azstate_2,times)


# In[29]:


#the states are redefined

axstate_1 = catstate(N,np.exp(1J*phi)*np.sqrt(eps_2/K))
axstate_2 = catstate(N,-np.exp(1J*phi)*np.sqrt(eps_2/K))

aystate_1 = axstate_1 - 1J*axstate_2
aystate_1 = aystate_1.unit()
aystate_2 = axstate_1 + 1J*axstate_2
aystate_2 = aystate_2.unit()

azstate_1 = axstate_1 + axstate_2
azstate_1 = azstate_1.unit()
azstate_2 = axstate_1 - axstate_2
azstate_2 = azstate_2.unit()


# In[30]:


#martix in phase space
X = axstate_1*axstate_1.dag() - axstate_2*axstate_2.dag()
Y = aystate_1*aystate_1.dag() - aystate_2*aystate_2.dag()
Z = azstate_1*azstate_1.dag() - azstate_2*azstate_2.dag()


# In[31]:


def coordinates(resultstate):
    return  expect(X,resultstate),expect(Y,resultstate),expect(Z,resultstate)


# In[32]:


psi_1=coordinates(resultsx1.states[-1])
psi_2=coordinates(resultsx2.states[-1])
psi_3=coordinates(resultsy1.states[-1])
psi_4=coordinates(resultsy2.states[-1])
psi_5=coordinates(resultsz1.states[-1])
psi_6=coordinates(resultsz2.states[-1])

bloch_sphere=Bloch()
bloch_sphere.point_color = ['orange','g','y','orchid','r','b']
bloch_sphere.point_marker = ['s','s','^','^','^','^']
bloch_sphere.point_size = [170,170,170]
bloch_sphere.add_points([psi_1[0],psi_1[1],psi_1[2]])
bloch_sphere.add_points([psi_2[0],psi_2[1],psi_2[2]])
bloch_sphere.add_points([psi_3[0],psi_3[1],psi_3[2]])
bloch_sphere.add_points([psi_4[0],psi_4[1],psi_4[2]])
bloch_sphere.add_points([psi_5[0],psi_5[1],psi_5[2]])
bloch_sphere.add_points([psi_6[0],psi_6[1],psi_6[2]])
bloch_sphere.view = [-60,30]
bloch_sphere.zlabel = ['$+Z$','']
bloch_sphere.xlabel = ['$+X$','']
bloch_sphere.ylabel = ['$+Y$','']
bloch_sphere.show()


# In[2]:


d = schemdraw.Drawing()
d += elm.Inductor2().down()
d += elm.Line().right()
d += elm.Capacitor().up()
d += elm.Line().left()
d.draw().save("a.png", dpi=300)
d.draw()


# In[3]:


d = schemdraw.Drawing()
d += elm.Josephson().down()
d += elm.Line().right()
d += elm.Capacitor().up()
d += elm.Line().left()
d.draw().save("b.png", dpi=300)
d.draw()


# In[5]:


d = schemdraw.Drawing()
d += elm.Inductor2().down()
d += elm.Line().right()
d += elm.Capacitor().up()
d += elm.Line().left()
d += elm.Line().right()
d += elm.Capacitor().right()
d += elm.Josephson().down()
d += elm.Line().right()
d += elm.Capacitor().up()
d += elm.Line().left()

d.draw().save("c.png", dpi=300)
d.draw()


# In[61]:


Nc = 5
ni = np.linspace(-Nc,Nc,2*Nc+1)
i = 0
nhat = ket2dm(basis(2*Nc+1,0))
nhat = nhat - ket2dm(basis(2*Nc+1,0))
while i < 2*Nc+1:
    nhat += ni[i]*ket2dm(basis(2*Nc+1,i))
    i += 1


# In[62]:


N = 100
ngsequence = np.linspace(-5,5,N)
ngsequence_unitNc = [i/Nc for i in ngsequence]


# In[63]:


###
Ec = 20E9
Ej = 2E9

JCO = basis(2*Nc+1,0)*basis(2*Nc+1,0).dag()
JCO = JCO - basis(2*Nc+1,0)*basis(2*Nc+1,0).dag()
i=0
while i < 2*Nc+1-1:
    JCO += -Ej/2 * (basis(2*Nc+1,i+1)*basis(2*Nc+1,i).dag()+ basis(2*Nc+1,i)*basis(2*Nc+1,i+1).dag())
    i += 1
    
Estates1 = np.zeros((5,100))
Evectorsg1 = []
Evectorse1 = []
i = 0
while i < N:
    ng = ngsequence[i]
    E_es = 4 * Ec * ((nhat - ng*qeye(2*Nc+1))**2)
    H = E_es + JCO
    Estates1[:,i] = H.eigenenergies()[:5]
    Evectorsg1.append(H.eigenstates()[1][0])
    Evectorse1.append(H.eigenstates()[1][1])
    i += 1


# In[68]:


plt.figure(dpi=300)

Estates1Ghz1 = [i/1E9 for i in Estates1[0,:]]
Estates1Ghz2 = [i/1E9 for i in Estates1[1,:]]
Estates1Ghz3 = [i/1E9 for i in Estates1[2,:]]
Estates1Ghz4 = [i/1E9 for i in Estates1[3,:]]
Estates1Ghz5 = [i/1E9 for i in Estates1[4,:]]

plt.plot(ngsequence_unitNc,Estates1Ghz1,label=r'$E_0$')
plt.plot(ngsequence_unitNc,Estates1Ghz2,label=r'$E_1$')
plt.plot(ngsequence_unitNc,Estates1Ghz3,label=r'$E_2$')
plt.plot(ngsequence_unitNc,Estates1Ghz4,label=r'$E_3$')
plt.plot(ngsequence_unitNc,Estates1Ghz5,label=r'$E_4$')
plt.ylabel("Energy (GHz)",fontsize=18)
plt.xlabel(r'$n_g/N_c$',fontsize=18)
plt.legend(loc=1,fontsize=12)

plt.xlim(-1,1)
plt.hlines(0,-1,1,linewidth=0.2,linestyle='dashed',color='black')
plt.ylim(-50,600)
plt.text(-0.5,550,r'$E_J = 2GHz, E_c = 20 GHz$',fontsize='x-large')
plt.text(-1.35,640,'a',fontsize=20)
plt.xticks(np.arange(min(ngsequence_unitNc), max(ngsequence_unitNc)+0.5, 0.5),fontsize=15)
plt.yticks(fontsize=16)
plt.show()


# In[69]:


###
Ec = 5E9
Ej = 5E9

JCO = basis(2*Nc+1,0)*basis(2*Nc+1,0).dag()
JCO = JCO - basis(2*Nc+1,0)*basis(2*Nc+1,0).dag()
i=0
while i < 2*Nc+1-1:
    JCO += -Ej/2 * (basis(2*Nc+1,i+1)*basis(2*Nc+1,i).dag()+ basis(2*Nc+1,i)*basis(2*Nc+1,i+1).dag())
    i += 1
    
Estates2 = np.zeros((5,100))
Evectorsg2 = []
Evectorse2 = []
i = 0
while i < N:
    ng = ngsequence[i]
    E_es = 4 * Ec * ((nhat - ng*qeye(2*Nc+1))**2)
    H = E_es + JCO
    Estates2[:,i] = H.eigenenergies()[:5]
    Evectorsg2.append(H.eigenstates()[1][0])
    Evectorse2.append(H.eigenstates()[1][1])
    i += 1


# In[70]:


plt.figure(dpi=300)

Estates2Ghz1 = [i/1E9 for i in Estates2[0,:]]
Estates2Ghz2 = [i/1E9 for i in Estates2[1,:]]
Estates2Ghz3 = [i/1E9 for i in Estates2[2,:]]
Estates2Ghz4 = [i/1E9 for i in Estates2[3,:]]
Estates2Ghz5 = [i/1E9 for i in Estates2[4,:]]

plt.plot(ngsequence_unitNc,Estates2Ghz1,label=r'$E_0$')
plt.plot(ngsequence_unitNc,Estates2Ghz2,label=r'$E_1$')
plt.plot(ngsequence_unitNc,Estates2Ghz3,label=r'$E_2$')
plt.plot(ngsequence_unitNc,Estates2Ghz4,label=r'$E_3$')
plt.plot(ngsequence_unitNc,Estates2Ghz5,label=r'$E_4$')
plt.ylabel("Energy (GHz)",fontsize=18)
plt.xlabel(r'$n_g/N_c$',fontsize=18)
plt.legend(loc=1,fontsize=12)

plt.xlim(-1,1)
plt.hlines(0,-1,1,linewidth=0.2,linestyle='dashed',color='black')
plt.ylim(-50,600)
plt.text(-0.5,550,r'$E_J = 5GHz, E_c = 5 GHz$',fontsize='x-large')
plt.text(-1.35,640,'b',fontsize=20)
plt.xticks(np.arange(min(ngsequence_unitNc), max(ngsequence_unitNc)+0.5, 0.5),fontsize=15)
plt.yticks(fontsize=15)
plt.show()


# In[71]:


###
Ec = 5E8
Ej = 50E9

JCO = basis(2*Nc+1,0)*basis(2*Nc+1,0).dag()
JCO = JCO - basis(2*Nc+1,0)*basis(2*Nc+1,0).dag()
i=0
while i < 2*Nc+1-1:
    JCO += -Ej/2 * (basis(2*Nc+1,i+1)*basis(2*Nc+1,i).dag()+ basis(2*Nc+1,i)*basis(2*Nc+1,i+1).dag())
    i += 1
    
Estates3 = np.zeros((5,100))
Evectorsg3 = []
Evectorse3 = []
i = 0
while i < N:
    ng = ngsequence[i]
    E_es = 4 * Ec * ((nhat - ng*qeye(2*Nc+1))**2)
    H = E_es + JCO
    Estates3[:,i] = H.eigenenergies()[:5]
    Evectorsg3.append(H.eigenstates()[1][0])
    Evectorse3.append(H.eigenstates()[1][1])
    i += 1


# In[72]:


plt.figure(dpi=300)

Estates3Ghz1 = [i/1E9 for i in Estates3[0,:]]
Estates3Ghz2 = [i/1E9 for i in Estates3[1,:]]
Estates3Ghz3 = [i/1E9 for i in Estates3[2,:]]
Estates3Ghz4 = [i/1E9 for i in Estates3[3,:]]
Estates3Ghz5 = [i/1E9 for i in Estates3[4,:]]

plt.plot(ngsequence_unitNc,Estates3Ghz1,label=r'$E_0$')
plt.plot(ngsequence_unitNc,Estates3Ghz2,label=r'$E_1$')
plt.plot(ngsequence_unitNc,Estates3Ghz3,label=r'$E_2$')
plt.plot(ngsequence_unitNc,Estates3Ghz4,label=r'$E_3$')
plt.plot(ngsequence_unitNc,Estates3Ghz5,label=r'$E_4$')
plt.ylabel("Energy (GHz)",fontsize=18)
plt.xlabel(r'$n_g/N_c$',fontsize=18)
plt.legend(loc=1,fontsize=12)

plt.xlim(-1,1)
plt.hlines(0,-1,1,linewidth=0.2,linestyle='dashed',color='black')
plt.ylim(-60,600)
plt.text(-0.5,550,r'$E_J = 50GHz, E_c = 0.5 GHz$',fontsize='x-large')
plt.text(-1.4,640,'c',fontsize=20)
plt.xticks(np.arange(min(ngsequence_unitNc), max(ngsequence_unitNc)+0.5, 0.5),fontsize=15)
plt.yticks(fontsize=15)
plt.show()


# In[73]:


Ep = 5E9


EJ_over_EC = np.linspace(0.1,100,1000)
EJ = [np.sqrt((Ep**2)*i/8) for i in EJ_over_EC]
EC = [(Ep**2)/(8*i) for i in EJ]

ng12 = 0.5
alpha05 = []

i = 0
while i < 1000:
    JCO = basis(2*Nc+1,0)*basis(2*Nc+1,0).dag()
    JCO = JCO - basis(2*Nc+1,0)*basis(2*Nc+1,0).dag()
    
    j=0
    while j < 2*Nc+1-1:
        JCO += -EJ[i]/2 * (basis(2*Nc+1,j+1)*basis(2*Nc+1,j).dag()+ basis(2*Nc+1,j)*basis(2*Nc+1,j+1).dag())
        j += 1
    
    E_es = 4 * EC[i] * ((nhat - ng12*qeye(2*Nc+1))**2)
    H = E_es + JCO
    EE = H.eigenstates()[0]
    E0 = EE[0]
    E1 = EE[1]
    E2 = EE[2]

    alpha05.append((E2-2*E1+E0)/(E1-E0))
    i += 1
adata05 = alpha05


# In[80]:


Ep = 5E9


EJ_over_EC = np.linspace(0.1,100,1000)
EJ = [np.sqrt((Ep**2)*i/8) for i in EJ_over_EC]
EC = [(Ep**2)/(8*i) for i in EJ]

ng12 = 0.001
alpha05 = []
tau = []
taum = []
i = 0
while i < 1000:
    JCO = basis(2*Nc+1,0)*basis(2*Nc+1,0).dag()
    JCO = JCO - basis(2*Nc+1,0)*basis(2*Nc+1,0).dag()
    
    j=0
    while j < 2*Nc+1-1:
        JCO += -EJ[i]/2 * (basis(2*Nc+1,j+1)*basis(2*Nc+1,j).dag()+ basis(2*Nc+1,j)*basis(2*Nc+1,j+1).dag())
        j += 1
    
    E_es = 4 * EC[i] * ((nhat - ng12*qeye(2*Nc+1))**2)
    H = E_es + JCO
    EE = H.eigenstates()[0]
    E0 = EE[0]
    E1 = EE[1]
    E2 = EE[2]
    tau.append(1/((E1-E0)*10*10**(-9)))
    taum.append(-1/((E1-E0)*10*10**(-9)))
    alpha05.append((E2-2*E1+E0)/(E1-E0))
    i += 1
adata00 = alpha05


# In[84]:


plt.figure(dpi=300)
plt.plot(EJ_over_EC,adata00,color = 'b',label=r'$\alpha_r$'+' for ng=0.001')
plt.plot(EJ_over_EC,adata05,color = 'r',label=r'$\alpha_r$'+' for ng=0.5')
plt.plot(EJ_over_EC,tau,color = 'g',label=r'$\alpha_r^{min}$')
plt.plot(EJ_over_EC,taum,color = 'g')
plt.axhline(y=0, color='black', linestyle='dashed',linewidth=0.5)
plt.xlim(0,50)
plt.ylim(-0.5,1)
plt.ylabel(r'$\alpha_r$',fontsize=18)
plt.xlabel(r'$E_J/E_C$',fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc=1,fontsize=12)
plt.show()


# In[ ]:




