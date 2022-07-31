import numpy
import itertools
import copy
import scipy.linalg
import scipy.optimize
import matplotlib.pyplot

# Geometry of pam model to print the basis for
# N: number of top (also bottom) row sites; there are 2N total sites
# u: number of up electrons
# d: number of down electrons
N= 4
u= 1
d= 1

# Printing options
# prints states and/or fock basis elements with their corresponding statenumber
printstates= True
printfock= False

# States and basis interpretation:
# In state notation:
# The first u entries represent the up electrons
# The last d entries represent the down electrons
# Each number gives the site number of the corresponding electron (from 0 to 2N- 1 including endpoints)
# In fock notation:
# The first 2N entries represent the presence of an up electron on the corresponding site or not
# The last 2N entries represent the presence of a down electron on the corresponding site or not

def repeat(x, i, j):
    setx= set(x[i: j])
    return len(x[i: j])!= len(setx)
def permuted(x, y):
    upxset= set(x[:u])
    upyset= set(y[:u])
    downxset= set(x[u: u+ d])
    downyset= set(y[u: u+ d])
    return upxset==upyset and downxset==downyset
def sortstate(s):
    ss= copy.copy(s)
    sortups= ss[:u]
    sortups.sort()
    sortdowns= ss[u: u+ d]
    sortdowns.sort()
    return sortups+ sortdowns
def makestates():
    x= [i for i in range(2*N)]
    z= itertools.product(x, repeat= u+ d)
    z= list(z)
    for a in z[:]:
        if(repeat(a, 0, u) or repeat(a, u, u+ d)): z.remove(a)
    dummyz= z        
    for a in dummyz:
        for b in dummyz:
            if((a!=b) and permuted(a, b)):
                z.remove(b)
    states= [list(a) for a in z]
    sorting= [sortstate(a) for a in states]
    sorting.sort()
    states= sorting
    states= [list(s) for s in set(tuple(ss) for ss in states)]
    sorting= [sortstate(a) for a in states]
    sorting.sort()
    states= sorting
    return states
def makebasis(states):
    statess= copy.copy(states)
    basis= []
    for a in statess:
        x= [0 for i in range(2*2*N)]
        for i in range(u):
            x[a[i]]= x[a[i]]+ 1
        for i in range(d):
            x[N+ a[u+ i]]= x[N+ a[u+ i]]+ 1
        basis.append(x)
    return basis
def sindex(s):
    ss= copy.copy(s)
    ss= sortstate(ss)
    return states.index(ss)
# ---------------------------------------------------------------------------------------
     

states= makestates()
slength= len(states)
basis= makebasis(states)

if printstates:
    for i in range(slength): print(states[i], i)
if printfock:
    for i in range(slength): print(basis[i], i)