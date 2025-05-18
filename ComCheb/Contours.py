from mpmath import mp
from numpy import array,linspace,sin,pi,exp,vectorize,ndarray,argmin




def Polygon(N,rad=1,sl=0):
    if sl>0:
        rad = sl/2/sin(pi/N)
    z = rad*exp(2j*pi*linspace(0,1,N+1))
    
    def gamma(t):
        if isinstance(t,(list,ndarray)):
            return array([gamma(t) for t in t])
        t = t-int(t)
        t = N*t
        k = int(t)
        t = t-k
        return z[k]+t*(z[k+1]-z[k])
    return gamma


def mpPolygon(N,rad=1,sl=0):
    exp = vectorize(mp.exp)
    if sl>0:
        rad = sl/2/mp.sin(mp.pi/N)

    z = rad*exp(2j*mp.pi*array(mp.linspace(0,1,N+1)))
    
    def mpgamma(t):
        if isinstance(t,(list,ndarray)):
            return array([mpgamma(t) for t in t])
        t = mp.mpf(t)-int(t)
        t = N*t
        k = int(t)
        t = t-k
        return z[k]+t*(z[k+1]-z[k])
    return mpgamma


def Hypocycloid(m=3,r=1):
    if type(m)!= int:
        print('m must be an integer!')
        return
    if m<3:
        print('m must be bigger or equal to 3!')
        return 
    
    def gamma(t):
        return r*exp(2j*pi*t)+r**(-m+1)*exp(-2j*pi*(m-1)*t)/(m-1)
    return gamma

def mpHypocycloid(m=3,r=1):
    if type(m)!= int:
        print('m must be an integer!')
        return
    if m<3:
        print('m must be bigger or equal to 3!')
        return 
    exp = vectorize(mp.exp)
    def mpgamma(t):
        t = 2j*mp.pi*t
        return r*exp(t)+(r*exp(t))**(-m+1)/(m-1)
    return mpgamma


def CircLune(a=1,r=1):
    if a<=0 or a>2:
        print('a needs to satisfy 0<a<2!')
        return
    def gamma(t):
        w = r*exp(2j*pi*t)
        return a * (1+ ((w-1)/(w+1))**a)/(1- ((w-1)/(w+1))**a)
    return gamma

def mpCircLune(a=1,r = 1):
    if a<=0 or a>2:
        print('a needs to satisfy 0<a<2!')
        return
    exp = vectorize(mp.exp)
    def mpgamma(t):
        w = r*exp(2j*mp.pi*t)
        return a * (1+ ((w-1)/(w+1))**a)/(1- ((w-1)/(w+1))**a)
    return mpgamma


def Lemniscate(m=2,r=1):
    def gamma(t):
        if isinstance(t,(list,ndarray)):
            return array([gamma(t) for t in t])

        t = t-int(t)
        t = m*t
        k = int(t)
        t = t+0.5
        if t-int(t)==0.5:
            t = -0.5

        return exp(2j*pi*k/m)*(r**m*exp(2j*pi *t)+1)**(1/m)
    return gamma


def mpLemniscate(m=2,r=1):
    r = mp.mpf(r)
    def mpgamma(t):
        if isinstance(t,(list,ndarray)):
            return array([mpgamma(mp.mpf(t)) for t in t])
        t = t-int(t)
        t = mp.mpf(m*t)
        k = mp.mpf(int(t))
        t = t+0.5
        if mp.mpf(t-int(t))==mp.mpf(0.5):
            k += 1
            tmp = mp.mpc(-1)
        else:
            tmp = mp.exp(2j*mp.pi *t)
        return mp.exp(2j*mp.pi*k/m)*(r**mp.mpf(m)*tmp+1)**(1/mp.mpf(m))
    return mpgamma


def mpE3(variant):
    if variant ==1:
        def E3v1 (t):
            r = 2
            if isinstance(t,(list,ndarray,mp.matrix)):
                return array([E3v1(t) for t in t])

            t = mp.mpf(t)
            t = t- int(t)
            t = t*3
            v = 1-r*mp.exp(2j*mp.pi*t)
            e = 1
            if t<0:
                t +=1

            if t<mp.mpf('1/6'):
                e = 1
            elif t< mp.mpf('7/6'):
                e = (-1-mp.mpf(-3)**0.5)/2
            elif t<mp.mpf('11/6'):
                e = 1
            elif t< mp.mpf('17/6'):
                e = (-1+mp.mpf(-3)**0.5)/2


            C = e*((27*v+(729*v**2+108)**0.5  )/2)**mp.mpf('1/3')
            return -mp.mpf('1/3')*(C-3/C)
        return E3v1

    elif variant == 2:
        @mp.memoize
        def switch(maxit=1000):
            r = mp.sqrt('31/27')
            def F(t):
                if isinstance(t,(list,ndarray,mp.matrix)):
                    return array([F(t) for t in t])
                v = 1-r*mp.exp(2j*mp.pi*t)
                tmp = 27*v+(729*v**2+108)**0.5
                return abs(mp.log(tmp).imag/mp.pi+0.5)
            t = mp.linspace(0.05,0.06,1001)
            tmp = F(t)
            k = argmin(tmp)
            prec = mp.dps
            mp.dps+=10
            for l in range(maxit):
                if tmp[k]<10**-prec:
                    break
                
                if k==0:
                    t = mp.linspace(t[0],t[1],101)
                elif k == len(t)-1:
                    t = mp.linspace(t[k-1],t[k],101)
                else:
                    t = mp.linspace(t[k-1],t[k+1],201)
                tmp = F(t)
                k = argmin(tmp)
            mp.dps-=10
            return t[k]



        def E3v2 (t):
            r = mp.sqrt('31/27')
            if isinstance(t,(list,ndarray,mp.matrix)):
                return array([E3v2(t) for t in t])

            t = mp.mpf(t)
            t = t - int(t)
            t = 3*t

            v = 1-r*mp.exp(2j*mp.pi*t)
            e = 1
            if t<0:
                t +=1

            s = switch()
            if t<s:
                e = 1
            elif t< s+1:
                e = (-1-mp.mpf(-3)**0.5)/2
            elif t<2-s:
                e=1
                t = t-1
            elif t<3-s:
                e = (-1+mp.mpf(-3)**0.5)/2
                t = t
            else:
                e = 1



            C = e*((27*v+(729*v**2+108)**0.5  )/2)**mp.mpf('1/3')
            return -mp.mpf('1/3')*(C-3/C)
        return E3v2

    else:
        print('variant needs to be either 1 or 2')
        raise


def mpE4(r,maxsteps=1000):

    @mp.memoize
    def comp_bounds(r):
        roots = array(mp.polyroots([1,-2,1,0,-r**2/16],maxsteps=maxsteps))
        roots = array([mp.re(root) for  root in roots])
        x1    = min(roots)
        return x1

    def gamma(t):
        if isinstance(t,(list,ndarray,mp.matrix)):
            return array([gamma(t) for t in t])
        y1 = lambda x: -(x**2-x+1/2)+mp.sqrt(x**2-x+1/4+r**2/16)
        s = lambda x: x+1j*y1(x)**0.5

        x1 = comp_bounds(r)
        l = -2*x1+1
        
        t = t - int(t)

        if t<0:
            t += 1
        t = 4*t
        k = int(t)
        t = t- int(t)
        t = l*t+x1

        if k ==0:
            return s(t)**0.5
        if k == 1:
            return (s(1-t)**0.5).conjugate()
        if k == 2:
            return -s(t)**0.5
        if k ==3:
            return -(s(1-t)**0.5).conjugate()
    return gamma
