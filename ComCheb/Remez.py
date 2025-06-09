from numpy import arange,vstack,hstack,ones,argmax,argmin,linspace,angle,array,sort,ndarray,exp,pi,inf
from numpy.linalg import solve
from numpy.random import random

from IPython.display import clear_output
from matplotlib.pyplot import figure,plot as pltplot,subplot,tight_layout,yscale,scatter


def __A(z,a,gamma,m,n,l,rc):
    tmp    = gamma(z)**arange(l,n*m+l,n)[:,None]
    tmp   *= exp(-1j*a)
    if rc:
        tmp    = vstack([ones(len(z)),tmp]).real
    else:
        tmp    = vstack([ones(len(z)),tmp,1j*tmp]).real
    return tmp

def __c(z,a,gamma,N):
    return (exp(-1j*a)*gamma(z)**N).real

def __remez_ex(m,n,l,t,a,r,gamma,rc):
    N = m*n+l
    u = solve(__A(t,a,gamma,m,n,l,rc).T,__c(t,a,gamma,N))
    L = u[1:]
    if rc:
        pass
    else:
        L = L[:m]+1j*L[m:]
    h = u[0]


    def phi(x):
        if isinstance(x,(list,ndarray)):
            z = gamma(x)[:,None]
        else:
            z = gamma(x)
        z = z**arange(l,n*m+l,n)
        return z@L
    
    E = lambda x : abs(gamma(x)**N-phi(x))
    

    
    x = linspace(0,1,10001)
    x = sort(hstack([x,t]))
    tmp = E(x)
    k = argmax(tmp)

    for _ in range(5):
        if k==0:
            x = linspace(x[0],x[1],101)
        elif k==len(x)-1:
            x = linspace(x[-2],x[-1],101)
        else :
            x = linspace(x[k-1],x[k+1],201)
        
        tmp = E(x)
        k = argmax(tmp)
    
    max_div = tmp[k]
    x = x[k]
    theta = angle(gamma(x)**N-phi(x))

    if theta<0:
        theta += 2*pi

    v       = exp(-1j*theta)*gamma(x)**arange(l,n*m+l,n)
    if rc:
        v       = hstack([1,v]).real
    else:
        v       = hstack([1,v,1j*v]).real
    d       = solve(__A(t,a,gamma,m,n,l,rc),v)
    #I       = r/d
    #I[I<0]  = inf
    #k       = argmin(I)
    #t[k]    = x
    #a[k]    = theta
    #delta   = r[k]/d[k]
    #r       = r-delta*d
    #r[k]    = delta

    I       = array(len(r)*[inf])
    I[d>0]  = r[d>0]/d[d>0]
    #I       = r/d
    #I[I<0]  = mp.inf
    k       = argmin(I)
    t[k]    = x
    a[k]    = theta
    delta   = r[k]/d[k]
    r       = r-delta*d
    r[k]    = delta
    
    return t,a,r,L,max_div,phi

def Remez(gamma,N, t=None, a=None,symmetry = 1,rc=False ,prec=1e-10,maxit=100,plot=False,pinfo=False):
    '''Performs Tangs algorithm for the given contour.
    Input:
    ------
        gamma    : function; it describes the wanted contour, it has to be able to handle numpy arrays.
        N        : integer; sets the order of the wanted Cheybshev polynomial.

    Optional:
    ---------
        t        : 1D numpy array; sets initial reference for the Remez routine. If None, then t is
                   choosen equidistantly.
        a        : 1D numpy array; sets the inital angles for the Renez routine. If None, then a is
                   choosen randomly.
        symmetry : integer; describes the symmetry factor. (n-regular polygon has symm. factor of n)
        rc       : bool; (default False) enable this setting if the coefficients are real. (speed up)
        prec     : float; wanted threshold for tangs algorithm.
        maxit    : integer; number of maximum iterations.
        plot     : bool; if True (default), then the chebyshev polynomial as well as the
                   relative error (for Tang's algorithm) is plotted.
        pinfo    : bool; short for print info. If True (default) then in each iteration most
                   important informations are printed. 

    Output:
    -------
        t        : 1D numpy array; final reference.
        a        : 1D numpy array; final angles.
        L        : 1D numpy array; cofficients of the polynomial Qm given by T_N(z)=z^lQm(z^n),
                   while T_N describes the Chebyshev polynomial of order N = nm+l.
        max_div  : float; maximum deviation of final approximation.
        rel_err  : float; relative error w.r.t. Tang's algorithm.
    '''
    if type(N)==type(t)==type(a)==type(None):
        print("'n','t','a' must not all be 'None'!")
    
    n = symmetry
    m = N//n
    if rc:
        m1 = m+1
    else:
        m1 = 2*m+1
    l = N%n

    gamma1 = gamma
    gamma  = lambda t: gamma1(t/n)

    

    if type(t)==type(None):
        t = array(linspace(0,1,m1))
    if type(a)==type(None):
        a = 2*pi *array(random(m1))

    if m1 != len(t):
        print('t has wrong length w.r.t. to the wanted order!')
        
    if m1 != len(a):
        print('a has wrong length w.r.t. to the wanted order!')
    
    r    = solve(__A(t,a,gamma,m,n,l,rc),array([1]+(m1-1)*[0]))

    a[r<0]+= pi
    a[a>=2*pi]-=2*pi
    r    = solve(__A(t,a,gamma,m,n,l,rc),array([1]+(m1-1)*[0]))

    rel_err = []

    if pinfo:
        print('Iteration\t h_p\t\t\t h_D\t\t\t relative error')
        clear = 0
    for counter in range(maxit):
        t,a,r,L,max_div,phi = __remez_ex(m,n,l,t,a,r,gamma,rc)
        #max_div = max(max_div,max(abs(gamma(t)**N-phi(t))))

        h = __c(t,a,gamma,N).T@r
        if pinfo:
            #clear_output(wait=True)
            tmp = ''.join([str(counter+1),'\t\t ',f'{max_div:.8e}','\t\t ',f'{h:.8e}','\t\t ',
                  f'{(max_div-h)/h:.8e}'])
            #print(clear*'      ',end='\r')
            #print(tmp,end='\r')
            print(tmp)
            #clear = len(tmp)
        
        rel_err += [abs(max_div-h)/abs(h)]
        if rel_err[-1]<prec:
            break
    if pinfo:
        print('\n')
    t = t/n
    phi1 = phi
    phi  = lambda t: phi1(n*t)
    #L = L*n**arange(l,n*m+l,n)
    if plot:
        figure(figsize=(10,5))
        subplot(121)
        x = linspace(0,1,1001)    
        pltplot(x,abs(gamma1(x)**N-phi(x)))
        scatter(t,abs(gamma1(t)**N-phi(t)))
        yscale('log')

        subplot(122)
        pltplot(range(1,len(rel_err)+1),rel_err)
        yscale('log')
        tight_layout()
    return t,a,L,max_div,rel_err[-1]
    
    
