import numpy
import itertools
import copy
import scipy.linalg
import scipy.optimize

# Inputs are Hubbard runs

run= "fixedu Hubbard 0.9996872297072689 3.0 24.072728139913668 4 1 1 0,0 3,3 3.0399099148943582,19.959220852105457,3.0399099148943582"
run= run.split()
ehubbard= float(run[3])
timehubbard= float(run[4])
N= int(run[5])
u= int(run[6])
d= int(run[7])
starting= [int(i) for i in run[8].split(",")]
ending= [int(i) for i in run[9].split(",")]
thubbard= [float(i) for i in run[10].split(",")]


t= thubbard
time= timehubbard
e= ehubbard
# Initialize bounds on variables
#                             v                          c        elow        time 
bounds= tuple([(1, 120) for i in range((N+ 1)//2)]+ [(0, 4)]+ [(1, 120)]+ [(0, 120)])

# Iterations to do dual annealing for, and maximum search iterations for each dual annealing run
iterations= 1
maxiter= 1000000

# Code configuration
# Variables declared here are primarily for noting the exact setup a fidelity is calculated for when the results are looked at later.
model= "Pam"
normalizationdata= "HtoPFourth"

# Files to append results to; note data is appended each run and does not make a new file each run
# PamDualAnnealingFixedu: results are printed in a way that is easier to import into something like pandas or excel
# PamDualAnnealingFixedu: prints local minima found; contains a space separating each iteration
# The first file contains more "finalized" results of an iteration, while the second updates the progress of the code as it runs
filedata= open("HtoPFourthData.txt", "a")
fileminima= open("HtoPFourthMinima.txt", "a")

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

# x=[v, c, elow, time]
def minfid(x):
    v= numpy.concatenate((x[:N//2], x[:N//2][::-1]))
    elow= x[-2]
    time= x[-1]
    c= x[-3]
    t= [c*k for k in thubbard]
    hamil= hammer(t, v, e, elow)
    return 1-(numpy.abs(scipy.linalg.expm(complex(0, -time)*hamil)@initial)**2)[state]
# ---------------------------------------------------------------------------------------

# callback function of dual annealing
# is ran every time dual annealing finds a local minimum; purpose is to print the data of the local minimum
def printminima(x, f, context):
    v= numpy.concatenate((x[:N//2], x[:N//2][::-1]))
    elow= x[-2]
    time= x[-1]
    c= x[-3]
    t= [c*k for k in thubbard]
    tstring= ",".join(map(str, list(t)))
    vstring= ",".join(map(str, list(v)))
    #               fixedu             Pam   fidel  u   ulow   time   N   u   d   starting state   ending state      t         v
    addminima= f"{normalizationdata} {model} {1-f} {e} {elow} {time} {N} {u} {d} {startingstring} {endingstring} {tstring} {vstring}\n"
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

result= scipy.optimize.dual_annealing(minfid, bounds= bounds, maxiter= maxiter, callback= printminima)
# the rest of the lines in the for loop are for printing data collected
x= result.x
v= numpy.concatenate((x[:N//2], x[:N//2][::-1]))
elow= x[-2]
time= x[-1]
c= x[-3]
t= [c*k for k in thubbard]
fidelity= 1- result.fun
tstring= ",".join(map(str, list(t)))
vstring= ",".join(map(str, list(v)))
#               fromhubbardvc    Pam      fidel    u   ulow   time   N   u   d   starting state   ending state      t      v
adddata= f"{normalizationdata} {model} {fidelity} {e} {elow} {x[-1]} {N} {u} {d} {startingstring} {endingstring} {tstring} {vstring}\n"
filedata.write(adddata)
filedata.flush()
fileminima.write("\n")
fileminima.flush()