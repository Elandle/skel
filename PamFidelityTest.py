import numpy
import itertools
import copy
import scipy.linalg
import scipy.optimize
import matplotlib.pyplot
# choose a method to input parameters to test the fidelity of
# inputmethod= 0: input parameters to test as a string in the format of "data" files
# inputmethod= 1: input parameters directly
inputmethod= 0

if inputmethod== 0:
    run= "fixeduandulow Pam 0.9714766204770491 0.5 10.5 69.23215658719987 4 1 1 0,0 3,3 96.3347313107792,65.66424525199076,96.3347313107792 0.6510978619097898,0.8478977201560912,0.8478977201560912,0.6510978619097898"
    run= run.split()
    e= float(run[3])
    elow= float(run[4])
    time= float(run[5])
    N= int(run[6])
    u= int(run[7])
    d= int(run[8])
    starting= [int(i) for i in run[9].split(",")]
    ending= [int(i) for i in run[10].split(",")]
    t= [float(i) for i in run[11].split(",")]
    v= [float(i) for i in run[12].split(",")]

# Variables deciding what we are trying to find the fidelity for
# N: amount of sites
# u: amount of up electrons
# d: amount of down electrons
# starting: where the electrons begin at (using [up, up, ..., down, down, ...] notation; not Fock) starting from site 0 to site N-1
# ending: where to (hopefully) find the electrons to be at; using the same notation as starting
# t: top row coupling values (N- 1 of them)
# v: top to bottom row coupling values (N of them) 
# time: time to evaluate the fidelity at 
# e: interaction energy (usually denoted as u, but here it is e)
if inputmethod== 1:
    N= 6
    u= 1
    d= 1
    t= [4.15211, 91.03207, 52.90782, 91.03207, 4.15211]
    v= [0.3961, 0.80875, 1.13669, 1.13669, 0.80875, 0.3961]
    e= 0.5
    elow= 10.5
    starting= [0, 5]
    ending= [5, 0]
    time= 37.90272094375652

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

# generates hamiltonian
def hammer(t, v, e, elow):
    hammil= numpy.zeros((slength, slength))
    for ii in range(slength):
        st= copy.copy(states[ii])
        for a in itertools.product(range(u+ d), range(3)):
            new= copy.copy(st)
            newentry= 0
            if(0< st[a[0]]< N and a[1]== 0):
                new[a[0]]= new[a[0]]- 1
                newentry= t[st[a[0]]- 1]
            elif(st[a[0]]< N-1 and a[1]== 1):
                new[a[0]]= new[a[0]]+ 1
                newentry= t[st[a[0]]]
            elif(a[1]== 2):
                newcopy= copy.copy(new)
                hopslower= min(newcopy[a[0]], (newcopy[a[0]]+ N)%(2*N))
                hopsupper= max(newcopy[a[0]], (newcopy[a[0]]+ N)%(2*N))
                if a[0]< u:
                    upelectron= True
                else:
                    upelectron= False
                if upelectron:
                    hopcount= sum(1 for iii in newcopy[:u] if hopslower< iii< hopsupper)
                else:
                    hopcount= sum(1 for iii in newcopy[u:u+d] if hopslower< iii< hopsupper)
                hopsign= hopcount%2
                new[a[0]]= (new[a[0]]+ N)%(2*N)
                newentry= pow(-1, hopsign)*v[st[a[0]]%N]
            new= sortstate(new)
            if new in states:
                hammil[ii, sindex(new)]= hammil[ii, sindex(new)]- newentry
        for jj in range(u):
            if st[jj]< N and st[jj] in st[u: u+ d]: hammil[ii, ii]= hammil[ii, ii]+ e
            if st[jj]>= N and st[jj] in st[u: u+ d]: hammil[ii, ii]= hammil[ii, ii]+ elow
    return hammil 

# function for calculating fidelity
def fid(t, v, time, e, elow):
    hamil= hammer(t, v, e, elow)
    matplotlib.pyplot.imshow((numpy.abs(scipy.linalg.expm(complex(0, -time)*hamil))**2), cmap="bwr", vmin= -1, vmax= 1)
    matplotlib.pyplot.colorbar()
    matplotlib.pyplot.gcf().set_dpi(400)
    matplotlib.pyplot.show()
    return (numpy.abs(scipy.linalg.expm(complex(0, -time)*hamil)@initial)**2)[state]


# Making the states and basis list, also initializing the states we are evolving from (initial) and trying to go to (using the state number of this)
states= makestates()
slength= len(states)
basis= makebasis(states)
initial= numpy.zeros(slength)
initial[sindex(starting)]= 1
state= sindex(ending) 


# print the fidelity
print(f"Fidelity= {fid(t, v, time, e, elow)}")