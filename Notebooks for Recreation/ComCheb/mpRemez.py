from mpmath import mp
from numpy import arange,hstack,vstack,ones,array,argmax,argmin,sort,ndarray,vectorize

from IPython.display import clear_output
from IPython import get_ipython
from matplotlib.pyplot import figure,plot as pltplot ,subplot,tight_layout,yscale,scatter

from pandas import DataFrame

from ComCheb.Cheby import Cheby

def __is_called_from_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter Notebook or JupyterLab
        elif shell == 'TerminalInteractiveShell':
            return False  # IPython terminal
        else:
            return False  # Other types (e.g., script)
    except NameError:
        return False      # Probably standard Python interpreter (e.g., CLI script)



def __A(z,a,gamma,m,n,l,rc):
    exp    = vectorize(mp.exp)
    real   = vectorize(mp.re)

    tmp    = gamma(z)**arange(l,n*m+l,n)[:,None]
    tmp   *= exp(-1j*a)
    #tmp    = real(vstack([ones(len(z)),tmp,1j*tmp]))

    if rc:
        tmp    = real(vstack([ones(len(z)),tmp]))
    else:
        tmp    = real(vstack([ones(len(z)),tmp,1j*tmp]))
    return tmp

def __c(z,a,gamma,N):
    exp    = vectorize(mp.exp)
    real   = vectorize(mp.re)
    return real(exp(-1j*a)*gamma(z)**N)

def __remez_ex(m,n,l,t,a,r,gamma,samples=1001,reps=5,prec=1e-15,rc=False):
    exp    = vectorize(mp.exp)
    real   = vectorize(mp.re)

    N = n*m+l
    try:
        u = array(mp.qr_solve(mp.matrix(__A(t,a,gamma,m,n,l,rc).T),mp.matrix(__c(t,a,gamma,N)))[0])
    except:    
        u = array(mp.lu_solve(mp.matrix(__A(t,a,gamma,m,n,l,rc).T),mp.matrix(__c(t,a,gamma,N))))
    L = u[1:]
    if rc :
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
    
    
    #E = lambda x : abs(gamma(x)**N-phi(x))
    E = lambda x : abs(Cheby(N,L,symmetry=n,gamma=gamma)(x))
    

    x = array(mp.linspace(0,1,samples))
    x = sort(hstack([x,t]))




    tmp = E(x)
    k = argmax(tmp)
    for _ in range(reps):
        if k==0:
            x = array(mp.linspace(x[0],x[1],101))
        elif k==len(x)-1:
            x = array(mp.linspace(x[-2],x[-1],101))
        else :
            x = array(mp.linspace(x[k-1],x[k+1],201))
        
        tmp = E(x)
        k = argmax(tmp)
    
    max_div = tmp[k]
    h = __c(t,a,gamma,N).T@r
    if (max_div-h)/abs(h)<prec:
        return t,a,r,L,max_div,h,phi


    x = x[k]
    #theta = mp.log(gamma(x)**N-phi(x)).imag
    theta = mp.log(Cheby(N,L,symmetry=n,gamma=gamma)(x)).imag

    if theta<0:
        theta += 2*mp.pi
    v       = exp(-1j*theta)*gamma(x)**arange(l,n*m+l,n)
    if rc:
        v       = real(hstack([1,v]))
    else:
        v       = real(hstack([1,v,1j*v]))
    try:
        d   = array(mp.qr_solve(mp.matrix(__A(t,a,gamma,m,n,l,rc)),mp.matrix(v))[0])
    except:
        d   = array(mp.lu_solve(mp.matrix(__A(t,a,gamma,m,n,l,rc)),mp.matrix(v)))

    I       = array(len(r)*[mp.inf])
    I[d>0]  = r[d>0]/d[d>0]
    #I       = r/d
    #I[I<0]  = mp.inf
    k       = argmin(I)
    t[k]    = x
    a[k]    = theta
    delta   = r[k]/d[k]
    r       = r-delta*d
    r[k]    = delta

    
    return t,a,r,L,max_div,h,phi

def mpRemez(gamma,N,t=None,a=None,symmetry = 1,rc=False,prec=1e-10,maxit=100,
            samples=1001,reps=10,plot=False,pinfo = False):
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
        samples  : integer; number of inital samples for computing maximum norm
        reps     : integer; number of refinements for computing maximum norm. Rule of thumb, set
                   it bigger or equal to -log10(prec).
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
    
    N = int(N)
    n = symmetry
    m = N//n
    if rc:
        m1 = m+1
    else:
        m1 = 2*m+1
    l = N%n


    if type(t)==type(None):
        t = array(mp.randmatrix(1,m1))
    else:
        t = symmetry *  array(mp.matrix(t))

    if type(a)==type(None):
        a = 2*mp.pi*array(mp.randmatrix(1,m1))
    else:
        a = array(mp.matrix(a))

    if m1 != len(t):
        print('t has wrong length w.r.t. to the wanted order!')
        
    if m1 != len(a):
        print('a has wrong length w.r.t. to the wanted order!')

    
    gamma1 = gamma
    gamma  = lambda t: gamma1(t/n)

    try:
        r    = array(mp.qr_solve(mp.matrix(__A(t,a,gamma,m,n,l,rc)),mp.matrix([1]+(m1-1)*[0]))[0])
    except:
        
        r    = array(mp.lu_solve(mp.matrix(__A(t,a,gamma,m,n,l,rc)),mp.matrix([1]+(m1-1)*[0])))

    a[r<0]+= mp.pi
    a[a>=2*mp.pi]-=2*mp.pi
    try:
        r    = array(mp.qr_solve(mp.matrix(__A(t,a,gamma,m,n,l,rc)),mp.matrix([1]+(m1-1)*[0]))[0])
    except:
        
        r    = array(mp.lu_solve(mp.matrix(__A(t,a,gamma,m,n,l,rc)),mp.matrix([1]+(m1-1)*[0])))

    rel_err = []

    df = DataFrame(columns=['Iteration','h_p','h_D','h_D increase','relative Error'])
    #df.index.name = 'Iteration'

    h1 = 0

    if pinfo and __is_called_from_notebook()==False:
        print('Iteration\t h_p\t\t\t h_D\t\t\t relative error')
    for counter in range(maxit):
        t,a,r,L,max_div,h,phi = __remez_ex(m,n,l,t,a,r,gamma,samples,reps,prec,rc)



        #h = __c(t,a,gamma,N).T@r
        rel_err += [float((max_div-h)/abs(h))]
        df.loc[counter+1] = [counter+1,float(max_div),float(h),float(h-h1),float((max_div-h)/abs(h))] 
        h1 = h
        if pinfo:
            if __is_called_from_notebook(): 
            #clear_output(wait=True)
            #print(counter+1,'\t',float(max_div),'\t',float(h),'\t',rel_err[-1])
                print(df[-10:].to_string(index=False))
                clear_output(wait=True)
            else:
                sgn = ['',' '][h>=0]
                tmp = ''.join([str(counter+1),'\t\t ',f'{float(max_div):.8e}','\t\t ',sgn
                               ,f'{float(h):.8e}','  \t ',sgn,
                                f'{float((max_div-h)/h):.8e}'])
                print(tmp)
        
        
        if rel_err[-1]<prec:
            break


    t = t/n
    phi1 = phi
    phi  = lambda t: phi1(n*t)
    #L = L*n**arange(l,n*m+l,n)

    if plot :
        figure(figsize=(10,5))
        subplot(121)
        x = array(mp.linspace(0,1,10001)) 
        pltplot(x,abs(gamma1(x)**N-phi(x)))
        scatter(t,abs(gamma1(t)**N-phi(t)))
        yscale('log')

        subplot(122)
        pltplot(range(1,len(rel_err)+1),rel_err)
        yscale('log')
        tight_layout()
    return t,a,L,max_div,rel_err[-1]
    
    
