from numpy import ndarray,arange

def Cheby(N,L,gamma = None,symmetry =1):
    n = symmetry
    m = N//n
    l = N%n
    def phi(z):
        if callable(gamma):
            z = gamma(z)
        if isinstance(z,(ndarray,)):
            z1 = z[:,None]
        else:
            z1 = z
        return z**N-(z1**arange(l,n*m+l,n))@L
    return phi