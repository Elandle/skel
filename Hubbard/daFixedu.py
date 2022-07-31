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
# e: Coulomb energy (u), stays fixed
N= 4
u= 1
d= 1
starting= [0, 0]
ending= [N- 1, N- 1]
e= 3

# Guess for minimum to find
# Set guessmethod to a value other than 0 or 1 if not being used
# Care must be taken using guesses, since they are setup dependent
# guessmethod== 0 takes in as a string a run "data" string
# guessmethod== 1 uses a manually defined guess
# Note: guessmethod== 0 overrides variables declared just above
guessmethod= -1
if guessmethod== 0:
    guessrun= "fixedu Hubbard 0.967 3 3.10878542 4 1 1 0,0 3,3 4.90011563,6.89365035,4.90004827"
    guessrun= guessrun.split()
    e= float(guessrun[3])
    guesstime= float(guessrun[4])
    N= int(guessrun[5])
    u= int(guessrun[6])
    d= int(guessrun[7])
    starting= [int(i) for i in guessrun[8].split(",")]
    ending= [int(i) for i in guessrun[9].split(",")]
    guesst= [float(i) for i in guessrun[10].split(",")]
    guess= guesst[:N//2]+ [guesstime]
if guessmethod== 1:
    #         t  time
    guess= [           ]

# Initialize bounds on variables
# boundsmethod== 0 uses manually defined bounds
# boundsmethod== 1 uses a default bound setup
# boundsmethod== 0 uses another manually defined bounds
# Code is only setup for boundsmethod taking the value 0,1, or 2, since having no bounds might result in something unphysical
boundsmethod= 2
if boundsmethod== 0:
    #                          t                     time
    bounds= tuple([(0, 200) for i in range(N//2)]+ [(0, 200)])
if boundsmethod== 1:
    bounds= tuple([(0, 100) for i in range(N//2)]+ [(0, 100)])
if boundsmethod== 2:
    bounds= tuple([(0, 2)]+ [(0, 2)]+ [(0, 100)])

# Iterations to do dual annealing for, and maximum search iterations for each dual annealing run
iterations= 1
maxiter= 1000000

# Code configuration
# Variables declared here are primarily for noting the exact setup a fidelity is calculated for when the results are looked at later.
model= "Hubbard"
normalizationdata= "fixedu"

# Files to append results to; note data is appended each run and does not make a new file each run
filedata= open("hubbdaFixeduData.txt", "a")
fileminima= open("hubbdaFixeduMinima.txt", "a")

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
            if st[jj] in st[u: u+ d]: hammil[ii, ii]= hammil[ii, ii]+ e
    return hammil

# Dual annealing requires the function to be minimized have its input be a vector, so the input is just x and we index what we want

# x=[t[0], t[1], t[2], ..., t[N//2], time]
def minfideven(x):
    t= numpy.concatenate((x[:N//2], x[:N//2- 1][::-1]))
    time= x[-1]
    hamil= hammer(t, e)
    return 1-(numpy.abs(scipy.linalg.expm(complex(0, -time)*hamil)@initial)**2)[state]
def minfidodd(x):
    t= numpy.concatenate((x[:-1], x[:-1][::-1]))
    time= x[-1]
    hamil= hammer(t, e)
    return 1-(numpy.abs(scipy.linalg.expm(complex(0, -time)*hamil)@initial)**2)[state]
# ---------------------------------------------------------------------------------------

# callback function of dual annealing
# is ran every time dual annealing finds a local minimum; purpose is to print the data of the local minimum
def printminimaeven(x, f, context):
    t= numpy.concatenate((x[:N//2], x[:N//2- 1][::-1]))
    tstring= ",".join(map(str, list(t)))
    time= x[-1]
    fidelity= 1-f
    #                   fixedu       Hubbard  fidelity   u   time   N   u   d   starting state   ending state      t
    addminima= f"{normalizationdata} {model} {fidelity} {e} {time} {N} {u} {d} {startingstring} {endingstring} {tstring}\n"
    fileminima.write(addminima)
    fileminima.flush()
def printminimaodd(x, f, context):
    t= numpy.concatenate((x[:N//2], x[:N//2][::-1]))
    tstring= ",".join(map(str, list(t)))
    time= x[-1]
    fidelity= 1-f
    #                   fixedu       Hubbard  fidelity   u   time   N   u   d   starting state   ending state      t
    addminima= f"{normalizationdata} {model} {fidelity} {e} {time} {N} {u} {d} {startingstring} {endingstring} {tstring}\n"
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
# Separating things printed if the files already has some stuff in it
filedata.write("\n")
fileminima.write("\n")

if N%2== 0:
    for i in range(iterations):
        if "guess" in globals():
            result= scipy.optimize.dual_annealing(minfideven, bounds= bounds, maxiter= maxiter, callback= printminimaeven, x0= guess)
        else:
            result= scipy.optimize.dual_annealing(minfideven, bounds= bounds, maxiter= maxiter, callback= printminimaeven)
        # the rest of the lines in the for loop are for printing data collected
        x= result.x
        fidelity= 1- result.fun
        time= x[-1]
        t= numpy.concatenate((x[:N//2], x[:N//2- 1][::-1]))
        tstring= ",".join(map(str, list(t)))
        #                   fixedu       Hubbard  fidelity   u   time   N   u   d   starting state   ending state      t
        adddata= f"{normalizationdata} {model} {fidelity} {e} {time} {N} {u} {d} {startingstring} {endingstring} {tstring}\n"
        filedata.write(adddata)
        filedata.flush()
        fileminima.write("\n")
        fileminima.flush()
else:
    for i in range(iterations):
        if "guess" in globals():
            result= scipy.optimize.dual_annealing(minfidodd, bounds= bounds, maxiter= maxiter, callback= printminimaodd, x0= guess)
        else:
            result= scipy.optimize.dual_annealing(minfidodd, bounds= bounds, maxiter= maxiter, callback= printminimaodd)
        # the rest of the lines in the for loop are for printing data collected
        x= result.x
        fidelity= 1- result.fun
        time= x[-1]
        t= numpy.concatenate((x[:N//2], x[:N//2][::-1]))
        tstring= ",".join(map(str, list(numpy.concatenate((x[:-1], x[:-1][::-1])))))
        #                   fixedu       Hubbard  fidelity   u   time   N   u   d   starting state   ending state      t
        adddata= f"{normalizationdata} {model} {fidelity} {e} {time} {N} {u} {d} {startingstring} {endingstring} {tstring}\n"
        filedata.write(adddata)
        filedata.flush()
        fileminima.write("\n")
        fileminima.flush()