import numpy
import itertools
import copy
import scipy.linalg
import scipy.optimize
import matplotlib.pyplot
import matplotlib.animation
import matplotlib.colors

# Note: this generates a probability matrix for the Pam model

# Choosing an input method for a set of parameters to make a probability matrix animation for
# inputmethod== 0: takes run data strings as input
# inputmethod== 1: manually input parameters
inputmethod= 0

if inputmethod== 0:
    run= "fixeduandulow Pam 0.9976206992659102 0.0 10.5 172.6912156467187 6 1 1 0,0 5,5 8.163858423237452,3.4369559917039116,9.894572285197075,3.4369559917039116,8.163858423237452 0.7455234808799717,0.1,0.1,0.1,0.1,0.7455234808799717"
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
if inputmethod== 1:
    N= 4
    u= 1
    d= 1
    starting= [0, 0]
    ending= [N- 1, N- 1]
    e= 3
    elow= 10
    t= [2, 3, 2]
    v= [1, 1, 1, 1]

# Choose animation settings
startingtime= 0
endingtime= 20
timestep= 0.1
interval= 500
timeround= 1
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

hamil= hammer(t, v, e, elow)
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