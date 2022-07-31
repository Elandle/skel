import numpy
import itertools
import copy
import scipy.linalg
import scipy.optimize
import matplotlib.pyplot
import matplotlib.animation
import matplotlib.colors

# Note: this generates a probability matrix for the Hubbard model

# Choosing an input method for a set of parameters to make a probability matrix animation for
# inputmethod== 0: takes run data strings as input
# inputmethod== 1: manually input parameters
inputmethod= 0

if inputmethod== 0:
    run= "fixedu Hubbard 0.9999748434788246 3 0.00646908573090488 4 1 1 0,0 3,3 420.5755447771856,485.6051400369883,420.5755447771856"
    run= run.split()
    e= float(run[3])
    time= float(run[4])
    N= int(run[5])
    u= int(run[6])
    d= int(run[7])
    starting= [int(i) for i in run[8].split(",")]
    ending= [int(i) for i in run[9].split(",")]
    t= [float(i) for i in run[10].split(",")]
if inputmethod== 1:
    N= 4
    u= 1
    d= 1
    starting= [0, 0]
    ending= [N- 1, N- 1]
    e= 3
    t= [2, 3, 2]

# Choose animation settings
startingtime= 0
endingtime= 0.5
timestep= 0.0001
interval= 500
timeround= 4
vmin= -1
vcenter= 0
vmax= 1
minimumfidelity= 0.98


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

states= makestates()
slength= len(states)
basis= makebasis(states)
initial= numpy.zeros(slength)
initial[sindex(starting)]= 1
state= sindex(ending) 

def matrixcheck(matrix):
    m= numpy.shape(matrix)[0]
    good= True
    i= 0
    while good and i<m:
        row= matrix[i, :]
        if numpy.amax(row)< minimumfidelity:
            good= False
        i= i+1
    return good

hamil= hammer(t, e)
norm = matplotlib.colors.TwoSlopeNorm(vmin= vmin, vcenter= vcenter, vmax= vmax)
figure, axes= matplotlib.pyplot.subplots()
imshows= []
for i in numpy.arange(startingtime, endingtime, timestep):
    matrix= numpy.abs(scipy.linalg.expm(complex(0, -i)*hamil))**2
    if matrixcheck(matrix):
        image= axes.imshow(matrix, cmap= "bwr", norm= norm)
        title = axes.text(0.5,1.05,f"Time= {round(i, timeround)}", size= matplotlib.pyplot.rcParams["axes.titlesize"], ha= "center", transform= axes.transAxes, )
        imshows.append([image, title])
figure.colorbar(image, ax= axes, norm= norm)
animation= matplotlib.animation.ArtistAnimation(figure, imshows, interval= interval, blit= False, repeat_delay= 0, repeat= True)
animation.save("pmMinfid.gif")
matplotlib.pyplot.show()