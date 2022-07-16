import matplotlib.pyplot
import numpy
import itertools
import copy
import random
import scipy.linalg
import scipy.optimize

# Variables deciding what we are trying to find the fidelity for
# N: amount of sites
# u: amount of up electrons
# d: amount of down electrons
# starting: where the electrons begin at (using [up, up, ..., down, down, ...] notation; not Fock) starting from site 0 to site N-1
# ending: where to (hopefully) find the electrons to be at; using the same notation as starting
# t: coupling values (N- 1 of them)
# time: time to evaluate the fidelity at 
# e: interaction energy (usually denoted as u, but here it is e)
N= 6
u= 1
d= 1
starting= [0, N- 1]
ending= [N- 1, 0]
t= [32.83202954091803,96.73565364177868,4.714279048083176,96.73565364177868,32.83202954091803]
e= 10.5
time= 86.98084358376835

# Functions used later for basic setting up
# Generates states and basis
# ---------------------------------------------------------------------------------------
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
    x= [i for i in range(N)]
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
    return states
def makebasis(states):
    basis= []
    for a in states:
        x= [0 for i in range(2*N)]
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

# generates hamiltonian
def hammer(t, e):
    hammil= numpy.zeros((slength, slength))
    for ii in range(slength):
        st= copy.copy(states[ii])
        for a in itertools.product(range(u+ d), range(2)):
            new= copy.copy(st)
            newentry= 0
            if(0< st[a[0]]< N and a[1]== 0):
                new[a[0]]= new[a[0]]- 1
                newentry= t[st[a[0]]- 1]
            elif(st[a[0]]< N-1 and a[1]== 1):
                new[a[0]]= new[a[0]]+ 1
                newentry= t[st[a[0]]]
            new= sortstate(new)
            if new in states:
                hammil[ii, sindex(new)]= -newentry
        for jj in range(u):
            if st[jj] in st[u: u+ d]: hammil[ii, ii]= e
    return hammil

# function for calculating fidelity
def fid(t, time, e):
    hamil= hammer(t, e)
    return (numpy.abs(scipy.linalg.expm(complex(0, -time)*hamil)@initial)**2)[state]
# ---------------------------------------------------------------------------------------

# Making the states and basis list, also initializing the states we are evolving from (initial) and trying to go to (using the state number of this)
states= makestates()
slength= len(states)
basis= makebasis(states)
initial= numpy.zeros(slength)
initial[sindex(starting)]= 1
state= sindex(ending) 


# print the fidelity
print(f"Fidelity= {fid(t, time, e)}")


