import matplotlib.pyplot
import numpy
import itertools
import copy
import scipy.linalg
import scipy.optimize

# Variables deciding what we are trying to find the fidelity for
# N: amount of sites; must be an even natural number for how this code is setup (code can easily be altered for different ones)
# u: amount of up electrons
# d: amount of down electrons
# starting: where the electrons begin at (using [up, up, ..., down, down, ...] notation; not Fock) starting from site 0 to site N-1
# ending: where to (hopefully) find the electrons to be at; using the same notation as starting
N= 6
u= 1
d= 1
starting= [0, N- 1]
ending= [N- 1, 0]

# Initialize bounds on variables here if used, otherwise a chosen default bound will be used
# Comment out if not being used.
# Format: tuple of length 2 tuples. In each length 2 tuple is the variable range that the respective variable can be in.
# (t[0] tuple, t[1] tuple, t[2] tuple, ..., t[middle- 1] tuple, time, e)
# bounds= ((,), (,), (,), (,), (,))
bounds= tuple([(0, 100) for i in range(N//2-1)]+ [(0, 100)]+ [(1, 100)])

# Iterations to do dual annealing for, and maximum search iterations for each dual annealing run
iterations= 10
maxiter= 1000

# Code configuration
# Variables declared here are primarily for noting the exact setup a fidelity is calculated for when the results are looked at later.
model= "Hubbard"
normalizationdata= "midChristandl"

# Files to append results to; note data is appended each run and does not make a new file each run
# HubbardDualAnnealingMidChristandlData: results are printed in a way that is easier to import into something like pandas or excel
# HubbardDualAnnealingMidChristandlMinima: prints local minima found; contains a space separating each iteration
# The first file contains more "finalized" results of an iteration, while the second updates the progress of the code as it runs
filedata= open("HubbardDualAnnealingMidChristandlData.txt", "a")
fileminima= open("HubbardDualAnnealingMidChristandlMinima.txt", "a")

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
    states= [list(s) for s in set(tuple(ss) for ss in states)]
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

# Main functions used in dual annealing
# hammer: make hamiltonian
# minfid: calculates 1-fidelity; function for dual annealing to minimize
# ---------------------------------------------------------------------------------------
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

# Dual annealing requires the function to be minimized have its input be a vector, so the input is just x and we index what we want
# mid Christandl value is hardcoded here for normalization; can change this if wanted
# mid Christandl implicitly assumes an even amount of sites (because there is a middle coupling)

# x=[t[0], t[1], t[2], ..., t[N/2-2], time, e]
def minfid(x):
    t= numpy.concatenate((x[:(N//2-1)], numpy.array([christandl]), x[-3::-1]))
    time= x[-2]
    e= x[-1]
    hamil= hammer(t, e)
    return 1-(numpy.abs(scipy.linalg.expm(complex(0, -time)*hamil)@initial)**2)[state]
# ---------------------------------------------------------------------------------------

# callback function of dual annealing
# is ran every time dual annealing finds a local minimum; purpose is to print the data of the local minimum
def printminima(x, f, context):
    tstring= ",".join(map(str, list(x[:(N//2-1)])+[christandl]+list(x[-3::-1])))
    #               midChristandl   Hubbard  fidel    u      time   N   u   d   starting state   ending state      t
    addminima= f"{normalizationdata} {model} {1-f} {x[-1]} {x[-2]} {N} {u} {d} {startingstring} {endingstring} {tstring}\n"
    fileminima.write(addminima)
    fileminima.flush()

# Making the states and basis list, also initializing the states we are evolving from (initial) and trying to go to (using the state number of this)
states= makestates()
slength= len(states)
basis= makebasis(states)
initial= numpy.zeros(slength)
initial[sindex(starting)]= 1
state= sindex(ending) 

# Setting up some strings used for printing results and minima
startingstring= ",".join(map(str, starting))
endingstring= ",".join(map(str, ending))


# Dual annealing
# -------------------------------------------------------------------------------
# Setting the Christandl value for middle t normalization
christandl= N/2

# Default variable bounds if none are declared earlier
if "bounds" not in globals():
    #                t's                               time          e
    bounds= tuple([(0, 100) for i in range(N//2-1)]+ [(0, 100)]+ [(0, 100)])
    
# Separating things printed if the files already has some stuff in it
filedata.write("\n")
fileminima.write("\n")

for i in range(iterations):
    result= scipy.optimize.dual_annealing(minfid, bounds= bounds, maxiter= maxiter, callback= printminima)
    # the rest of the lines in the for loop are for printing data collected
    x= result.x
    fidelity= 1- result.fun
    tstring= ",".join(map(str, list(x[:(N//2-1)])+[christandl]+list(x[-3::-1])))
    #               midChristandl   Hubbard  fidel       u      time   N   u   d   starting state   ending state      t
    adddata= f"{normalizationdata} {model} {fidelity} {x[-1]} {x[-2]} {N} {u} {d} {startingstring} {endingstring} {tstring}\n"
    filedata.write(adddata)
    filedata.flush()
    fileminima.write("\n")
    fileminima.flush()