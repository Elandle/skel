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
# e: Coulomb energy (u) for top row, stays fixed
# elow: Coulomb energy (ulow) for bottom row, stays fixed
N= 5
u= 1
d= 1
starting= [0, 0]
ending= [N- 1, N- 1]
e= 3
elow= 6

# Guess for minimum to find
# Set guessmethod to a value other than 0 or 1 if not being used
# Care must be taken using guesses, since they are setup dependent
# guessmethod== 0 takes in as a string a run "data" string
# guessmethod== 1 uses a manually defined guess
# Note: guessmethod== 0 overrides variables declared just above
guessmethod= -1
if guessmethod== 0:
    guessrun= "fixeduandulow Pam 0.9965550995321618 0.0 10.5 32.608245099366044 8 1 1 0,0 7,7 3.423311545248468,0.482022092133021,4.99999181682617,2.313131313392421,4.99999181682617,0.482022092133021,3.423311545248468 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1"
    guessrun= guessrun.split()
    e= float(guessrun[3])
    elow= float(guessrun[4])
    guesstime= float(guessrun[5])
    N= int(guessrun[6])
    u= int(guessrun[7])
    d= int(guessrun[8])
    starting= [int(i) for i in guessrun[9].split(",")]
    ending= [int(i) for i in guessrun[10].split(",")]
    guesst= [float(i) for i in guessrun[11].split(",")]
    guessv= [float(i) for i in guessrun[12].split(",")]
    guess= guesst[:N//2]+ guessv[:(N+ 1)//2]+ [guesstime]
if guessmethod== 1:
    #       t   v   time
    guess= [            ]

# Initialize bounds on variables
# boundsmethod== 0 uses manually defined bounds
# boundsmethod== 1 uses a default bound setup
# Code is only setup for boundsmethod taking the value 0 or 1, since having no bounds might result in something unphysical
boundsmethod= 0
if boundsmethod== 0:
    #                            t                               v                       time
    bounds= tuple([(0, 20) for i in range(N//2)]+ [(1, 20) for i in range((N+ 1)//2)]+ [(0, 60)])
if boundsmethod== 1:
    bounds= tuple([(0, 100) for i in range(N//2)]+ [(0, 100) for i in range((N+ 1)//2)]+ [(0, 100)])

# Iterations to do dual annealing for, and maximum search iterations for each dual annealing run
iterations= 1
maxiter= 100000

# Code configuration
# Variables declared here are primarily for noting the exact setup a fidelity is calculated for when the results are looked at later.
model= "Pam"
normalizationdata= "fixeduandulow"

# Files to append results to; note data is appended each run and does not make a new file each run
filedata= open("pamdaFixeduAndulowData.txt", "a")
fileminima= open("pamdaFixeduAndulowMinima.txt", "a")

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

# Dual annealing requires the function to be minimized have its input be a vector, so the input is just x and we index what we want

# x=[t[0], t[1], t[2], ..., t[N//2], v[0], v[1], ..., time]
def minfideven(x):
    t= numpy.concatenate((x[:N//2], x[:N//2- 1][::-1]))
    v= numpy.concatenate((x[N//2: N], x[N//2: N][::-1]))
    time= x[-1]
    hamil= hammer(t, v, e, elow)
    return 1-(numpy.abs(scipy.linalg.expm(complex(0, -time)*hamil)@initial)**2)[state]
def minfidodd(x):
    t= numpy.concatenate((x[:N//2], x[:N//2][::-1]))
    v= numpy.concatenate((x[N//2: N], x[N//2: N- 1][::-1]))
    time= x[-1]
    hamil= hammer(t, v, e, elow)
    return 1-(numpy.abs(scipy.linalg.expm(complex(0, -time)*hamil)@initial)**2)[state]
# ---------------------------------------------------------------------------------------

# callback function of dual annealing
# is ran every time dual annealing finds a local minimum; purpose is to print the data of the local minimum
def printminimaeven(x, f, context):
    t= numpy.concatenate((x[:N//2], x[:N//2- 1][::-1]))
    v= numpy.concatenate((x[N//2: N], x[N//2: N][::-1]))
    tstring= ",".join(map(str, list(t)))
    vstring= ",".join(map(str, list(v)))
    time= x[-1]
    fidelity= 1-f
    #                   fixedu         Pam    fidelity   u   ulow   time   N   u   d   starting state   ending state      t         v
    addminima= f"{normalizationdata} {model} {fidelity} {e} {elow} {time} {N} {u} {d} {startingstring} {endingstring} {tstring} {vstring}\n"
    fileminima.write(addminima)
    fileminima.flush()
def printminimaodd(x, f, context):
    t= numpy.concatenate((x[:N//2], x[:N//2][::-1]))
    v= numpy.concatenate((x[N//2: N], x[N//2: N- 1][::-1]))
    tstring= ",".join(map(str, list(t)))
    vstring= ",".join(map(str, list(v)))
    time= x[-1]
    fidelity= 1-f
    #                   fixedu         Pam    fidelity   u   ulow   time   N   u   d   starting state   ending state      t         v
    addminima= f"{normalizationdata} {model} {fidelity} {e} {elow} {time} {N} {u} {d} {startingstring} {endingstring} {tstring} {vstring}\n"
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
        t= numpy.concatenate((x[:N//2], x[:N//2- 1][::-1]))
        v= numpy.concatenate((x[N//2: N], x[N//2: N][::-1]))
        tstring= ",".join(map(str, list(t)))
        vstring= ",".join(map(str, list(v)))
        time= x[-1]
        #                 fixedu         Pam    fidelity   u   ulow   time   N   u   d   starting state   ending state      t         v
        adddata= f"{normalizationdata} {model} {fidelity} {e} {elow} {time} {N} {u} {d} {startingstring} {endingstring} {tstring} {vstring}\n"
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
        v= numpy.concatenate((x[N//2: N], x[N//2: N- 1][::-1]))
        tstring= ",".join(map(str, list(t)))
        vstring= ",".join(map(str, list(v)))
        #                 fixedu         Pam    fidelity   u   ulow   time   N   u   d   starting state   ending state      t         v
        adddata= f"{normalizationdata} {model} {fidelity} {e} {elow} {time} {N} {u} {d} {startingstring} {endingstring} {tstring} {vstring}\n"
        filedata.write(adddata)
        filedata.flush()
        fileminima.write("\n")
        fileminima.flush()