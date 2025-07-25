from numpy import vectorize,diag,array,arange,hstack
from mpmath import mp

def Roots(L,N,symmetry=1):
    mp_type = L.dtype=='O'
    if mp_type==False:
        dps1 = mp.dps
        mp.dps = 16
    n = symmetry
    m = N//n
    l = N%n

    exp = vectorize(mp.exp)

    A = diag((m-1)*[1],-1).astype('O')
    A[:,-1] = L
    e = array(mp.eig(mp.matrix(A))[0])
    e = e**(mp.mpf(1)/n)
    e = e*exp(2j*mp.pi*arange(n)/mp.mpf(n))[:,None]
    e = e.ravel()
    e = hstack([e,l*[0]])

    if mp_type==False:
        mp.dps = dps1
        e = e.astype(complex)    
    return e