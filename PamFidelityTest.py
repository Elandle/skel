import numpy
import itertools
import copy
import scipy.linalg
import scipy.optimize
# choose a method to input parameters to test the fidelity of
# inputmethod= 0: input parameters to test as a string in the format of "data" files
# inputmethod= 1: input parameters directly
inputmethod= 0

if inputmethod== 0:
    run= "fixeduandulow Pam 0.5344991639776946 0.5 8.260523833334446 4.787179862301293 6 2 2 0,1,0,1 4,5,4,5 10.0,5.676180554309135,9.517869207379908,5.676180554309135,10.0 0.9458895772903503,0.6156074255912841,0.6209481306200096,0.6209481306200096,0.6156074255912841,0.9458895772903503"
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
    N= 4
    u= 1
    d= 1
    starting= [0, 0]
    ending= [N- 1, N- 1]
    e= 0.5
    elow= 10.5
    time= 75.56654811452918
    t= [2.4139484426251574,77.77211814669997,2.4139484426251574]
    v= [0.2315576433372141,1.8691993002878327,1.8691993002878327,0.2315576433372141]

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
def hammer(t, v, e):
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
    hamil= hammer(t, v, e)
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
