# Introduction
This repository provides the Python library `ComCheb` for computing complex Chebyshev polynomials associated with a given contour or a Jordan domain, with the goal of reproducing and exploring results from the recently published preprint [1]. These polynomials are of great interest in approximation theory, numerical conformal mapping, and spectral methods on complex domains.
To facilitate experimentation, the library also includes a collection of sample contours, all of which are presented and studied in [1]. The library includes two complementary implementations: one based on NumPy for fast and efficient double-precision computations, and another using mpmath, which supports arbitrary-precision arithmetic for high-accuracy calculations and validation.
This project is intended for researchers, students, and practitioners working in numerical analysis, complex approximation, and computational mathematics.

Let us clarify the terminology and present the general problem addressed by this project. Given any compact set $K\subset \mathbb C$ we define the Chebyshev norm as $\|f\|_{K,\infty}:=\sup_{z\in K}|f(z)|$. Assume that $K$ consists of infinitely many points, we then define the $n$-th Chebyshev polynomial to be the unique minimizer
of
$$\underset{p\in M_n}{\mathrm{minimize}}\ \ \|p\|_{K,\infty} \tag{1}, $$
where $M_n$ denotes the set of all complex monic polynomials of degree $n$. The aim of the project is to compute the Chebyshev polynomials in the special case where $K$ is either a contour or a domain whose boundary can be described by a contour. It is worth mentioning that the latter case can be reduced to the former, as polynomials are entire functions and therefore obtain their absolute maximum at the boundary of the given domain. In order to be precise, we define a contour $\Gamma$ to be the image of a continuous function $\gamma :[0,1]\to\mathbb C$.


The computational approach provided by this project follows the algorithm introduced by P.T.P. Tang [2], which was further refined by B. Fischer and J. Modersitzki in [3]. The procedure designed by Tang can be interpreted as a complex variant of the Remez algorithm, which computes the best uniform polynomial approximation for a given continuous function on any finite interval. Although Tang's algorithm is applicable in a more general context (i.e. computing best polynomial approximations of arbitrary continuous functions) this library is confined to the special case of computing the Chebyshev polynomials for any given contour. 

Now let us briefly break down the key concepts of Tang's algorithm (for a more detailed description see [2,3]). The procedure exploits a dual representation of (1). In fact, let the contour $\Gamma$ be characterized by $\gamma:[0,1]\to\mathbb C$, then it holds that
$$ 
\min_{p\in M_n}\|p\|_{\Gamma,\infty} = \max_{\substack{\mathbf{\alpha}\in [0,2\pi)^{2n+1} \\ \mathbf t\in[0,1]^{2n+1}}}\left|\sum_{j=1}^{2n+1}r_j(\mathbf t,\mathbf \alpha) \mathrm{Re} \left( e^{-i\alpha_j}\gamma^n(t_j)\right)\right|=:\max_{\substack{\mathbf{\alpha}\in [0,2\pi)^{2n+1} \\ \mathbf t\in[0,1]^{2n+1}}} h(\mathbf t,\mathbf \alpha),\tag{2}
$$
where $r(\mathbf t,\mathbf \alpha)$ is chosen to satisfy
$$\sum_{j=1}^{2n+1}r_j(\mathbf t,\mathbf \alpha)  = 1 ,\quad \text{ and }\quad\sum_{j=1}^{2n+1}r_j(\mathbf t,\mathbf \alpha) \mathrm{Re} \left( e^{-i\alpha_j}\gamma^k(t_j)\right)=0,\quad k=0,1,\dots,n-1.$$
Writing this in terms of matrices we have $A(\mathbf t,\mathbf \alpha)r =(1,0,\dots,0)\in\mathbb R^{2n+1}$ for some $\mathbf t$ and $\mathbf \alpha$ depending matrix $A(\mathbf t,\mathbf \alpha)$. Instead of solving (1) we aim to solve the right-hand side of (2), and recreate the $n$-th Chebyshev  polynomial $T_n^\Gamma$  from the optimal parameters $\mathbf t^*$ and $\mathbf \alpha^*$. If $\mathbb A$ denotes the set of all pairs $(\mathbf t,\mathbf \alpha)\in [0,1]^{2n+1}\times[0,2\pi)^{2n+1}$ such that $A(\mathbf t,\mathbf \alpha)$ is invertible, then there is a known continuous mapping $\colon \mathbb A \to M_n$, $(\mathbf t,\mathbf \alpha)\mapsto p_{\mathbf t,\mathbf\alpha}$ such that $ p_{\mathbf t*,\mathbf\alpha^*}=T_n^\Gamma$. Hence it remains to solve the right hand side of (2). Similar to the classical Remez algorithm, Tang's algorithm solves (2) in an iterative manner. The algorithm constructs a sequence of parameter pairs $(\mathbf t^{(\nu)},\mathbf \alpha^{(\nu)})$ such that $h(\mathbf t^{(\nu)},\mathbf \alpha^{(\nu)})$ is increasing in $\nu$. This sequence usually converges fast towards the unique solution $(\mathbf t^*,\mathbf \alpha^*)$. This iterative routine will be executed until $h(\mathbf t^{(\nu)},\mathbf \alpha^{(\nu)})$ is positive and
$$ \frac{\| p_{\mathbf t^{(\nu)},\mathbf \alpha^{(\nu)}}  \|_{\Gamma,\infty}-h(\mathbf t^{(\nu)},\mathbf \alpha^{(\nu)})}{h(\mathbf t^{(\nu)},\mathbf \alpha^{(\nu)})}<\varepsilon \tag{3} $$
for any predefined accuracy level $\varepsilon>0$. Once (3) is achieved, the polynomial $p_{\mathbf t^{(\nu)},\mathbf \alpha^{(\nu)}}$ is guaranteed to satisfy
$$ \frac{\| p_{\mathbf t^{(\nu)},\mathbf \alpha^{(\nu)}}\|_{\Gamma,\infty}-\|T_n\|_{\Gamma,\infty}}{\|T_n\|_{\Gamma,\infty}}<\varepsilon, \tag{4}$$
hence roughly the first $-\log_{10}(\varepsilon)$ digits of $\| p_{\mathbf t^{(\nu)},\mathbf \alpha^{(\nu)}}\|_{\Gamma,\infty}$ coincide with those of $\|T_n\|_{\Gamma,\infty}$. 





## How to get started
Currently, an installation via PyPI or similar providers is not supported. The easiest way is to copy the module `ComCheb` into the same folder as your Python or Jupyter files are stored, and import the library by `import ComCheb`. For explanations on how to use this library see the documentary below or `HowToUse.ipynb`, which describes the main functionalities of `ComCheb`.

In order to run the main routines, make sure to install the following modules:
- numpy
- mpmath
- matplotlib
- pandas
- IPython  

For the recreation of all results of [1], the following modules are also required:
- time
- pickle

## Overview of Recreation Folder
This folder contains:
- a copy of `ComCheb`,
- a folder for storing images,
- a folder for storing data,
- multiple jupyter notebooks containing the necessary code.   

The jupyter notebooks contain all the relevant code to fully recreate the results presented in [1]. The notebooks have the following purpose:
- Notebooks starting with `WT_` compute the Widom factors displayed in the Tables 1-3 in [1].
- Notebooks starting with `WA_` compute the 'Widom asymptotics', with respect to Figures 1-3 in [1].
- Notebooks starting with `FC_` compare difference between Chebyshev and Faber polynomials, creating the data needed for Figure 4 in [1].
- The Notebook `TRoots` computes the Chebyshev polynomials together with their roots needed for Figures 5-8 in [1].  

All the notebooks named above will store all relevant information within the data folder, while using the `pickle` format. Finally, the notebook `Results` accesses the data folder and binds all results together, creating all tables and Figures displayed in [1].

## References
[1] L. A. Hübner – O. Rubin. *Computing Chebyshev polynomials using the complex Remez algorithm*. Experimental Mathematics *(to appear)*
[2] P. T. P. Tang. *Chebyshev approximation on the complex plane*. University of California, Berkeley, 1987.  
[3] B. Fischer and Jan Modersitzki. *An algorithm for complex linear approximation based on semi-infinite programming*. Numerical Algorithms, vol. 5, pp. 287–297, 1993.

# Documentation
The structure of this library is displayed below. The routines starting with `mp` are employing the `mpmath` library and offer arbitrary precision arithmetic.


**ComCheb**
- [Contours](#contours)
  - [Polygons](#polygons)
  - [Hypocycloids](#hypocycloids)
  - [Circular Lunes](#circular-lunes)
  - [Lemniscate](#lemniscates)
  - [Special A](#special-a)
  - [Special B](#special-b)
- [Remez](#remez)
- [Chebyshev-evaluation](#chebyshev-evaluation)
- [Roots](#roots)

## Contours
All presented functions within this section will create a parameterization $\gamma:[0,1]\to\mathbb C$ depending on its input. That means, they will return an executable (1-periodic) function parameterizing the given contour. Further, functions starting with `mp` will use the module `mpmath` enabling high-precision computation. While using `mp` routines, make sure to feed the function either with string inputs like `'0.1'` or using the `mpf` data format provided by `mpmath`. All returned parameterizations are able to handle `floats`,`lists` and `numpy.ndarrays`, in addition the `mp` routines also handle `mpmath.matrix` inputs.

More detailed descriptions of the presented contours can be found in [1].
### Polygons
The functions `Polygon` and `mpPolygon` will return a parameterization of the regular $m$-polygon. The degree $m$ as well as the radius or the sidelengths can be adjusted.

```python
>>> from ComCheb.Contours import Polygon  
>>> gamma1 = Polygon(4,rad=2)   # Polygon of order 4 with radius 2  
>>> gamma2 = Polygon(5,sl=4)    # Polygon of order 5 with side length 4  

>>> print(gamma1(0.5))  
(-2+2.4492935982947064e-16j)    
>>> print(gamma2([0.1,0.2]))  
[2.22703273+1.61803399j 1.05146222+3.23606798j]
```

The `mpmath` version works analogously. To obtain accurate results, make sure that the input values are either given as strings or as `mp.mpf` format.
```python
>>> from ComCheb.Contours import mpPolygon  
>>> from mpmath import mp
>>> mp.pretty = True              # pretty display of mpmath's numbers  
>>> mp.dps = 30                   # set working precision to 30 digits  
>>> gamma  = mpPolygon(4,rad=2)   # Polygon of order 4 with radius 2  

>>> print(gamma(0.1))             # improper input
(1.19999999999999995559107901499 + 0.800000000000000044408920985006j)
>>> print(gamma('0.1'))           # proper input
(1.2 + 0.8j)  
>>> print(gamma(mp.mpf('0.1')))   # proper input
(1.2 + 0.8j)
```  

### Hypocycloids
The functions `Hypocycloid` and `mpHypocycloid` will return a parameterization of the regular $m$-cusped Hypocycloid. The $m$-Hypocycloid is given by the contour
$$ \mathsf{H}_m^r:= \left\{ re^{2 \pi i\theta}+ \frac{(re)^{-2\pi i(m-1)\theta}}{m-1}\ :\ \theta \in [0,1) \right\}.$$
By default `r=1` is used.

```python
>>> from ComCheb.Contours import Hypocycloid  
>>> gamma1 = Hypocycloid(5)          # 5-cusped Hypocycloid
>>> gamma2 = Hypocycloid(5,1.5)      # 5-cusped Hypocycloid, w.r.t. the level line r = 1.5

>>> print(gamma1([0.1,1/3])) 
[ 0.60676275+0.44083894j -0.625     +0.64951905j]
>>> print(gamma2([0.1,1/3]))
[ 1.17357404+0.85265145j -0.77469136+1.25627142j]
```

Analogously for the `mpmath` variant. Again, make sure to use proper input types, to ensure accurate outputs.

```python
>>> from ComCheb.Contours import mpHypocycloid  
>>> from mpmath import mp
>>> mp.pretty = True                    # pretty display of mpmath's numbers  
>>> mp.dps = 30                         # set working precision to 30 digits  
>>> gamma  = mpHypocycloid(5,'1.5')     # 5-cusped Hypocycloid, w.r.t. the level line r = 1.5 

>>> print(gamma([0.1,1/3]))             # improper input values
[(1.17357403505007801868034181549 + 0.852651446226735759965705828294j)
 (-0.774691358024691187106944610677 + 1.25627141907001910034040635673j)]   
>>> print(gamma(['0.1','1/3']))         # proper input values
[(1.17357403505007805348172193233 + 0.852651446226735712065715428026j)
 (-0.774691358024691358024691358024 + 1.25627141907001902462638855017j)]
```  
### Circular Lunes
The functions `CircLune` and `mpCircLune` will return a parameterization for the $\alpha$-circular lune. The $\alpha$-circular lune is given by 
$$ \mathsf{C}_\alpha^r:= \left\{ \alpha \frac{1+\left(\frac{w-1}{w+1}\right)^\alpha}{1-\left(\frac{w-1}{w+1} \right)^\alpha}\ :\ |w| = r \right\}.$$
By default `r=1` is used, the value $\alpha$ my be chosen from the interval $(0,2]$.

```python
>>> from ComCheb.Contours import CircLune 
>>> gamma1 = CircLune(1/2)          # 0.5-Circular Lune
>>> gamma2 = CircLune(1/2,1.3)      # 0.5-Circular Lune, w.r.t. the level line r = 1.3

>>> print(gamma1([0.1,1/3])) 
[ 0.65062521+0.77692388j -0.42031251+1.06862764j]
>>> print(gamma2([0.1,1/3]))
[ 0.91300251+0.90101979j -0.57749157+1.28748672j]
```

Analogously for the `mpmath` variant. Again, make sure to use proper input types, to ensure accurate outputs.

```python
>>> from ComCheb.Contours import mpCircLune 
>>> from mpmath import mp
>>> mp.pretty = True                         # pretty display of mpmath's numbers  
>>> mp.dps = 30                              # set working precision to 30 digits  
>>> gamma  = mpCircLune('0.5','1.3')         # 0.5-Circular Lune, w.r.t. the level line r = 1.3 

>>> print(gamma([0.1,1/3]))                  # improper input values
[(0.913002506415567641121423908113 + 0.901019786179443762141871336267j)
 (-0.57749156508511440112838059553 + 1.28748672330154726509990439083j)] 
>>> print(gamma(['0.1','1/3']))              # proper input values
[(0.913002506415567661753000440527 + 0.901019786179443722244663619892j)
 (-0.577491565085114515043988502908 + 1.28748672330154718549442819655j)]
```  

### Lemniscates
The functions `Lemniscate` and `mpLemniscate` will return a parameterization for the family of lemniscates given by
$$\mathsf{L}_m^r :=\left\{z\ : \ \left| z^m-1 \right| =r^m\right\},$$
by default `r=1` is used.

```python
>>> from ComCheb.Contours import Lemniscate
>>> gamma1 = Lemniscate(5)          # 5-Lemniscate
>>> gamma2 = Lemniscate(5,1.3)      # 5-Lemniscate, w.r.t. the level line r = 1.3

>>> print(gamma1([0.1,1/3])) 
[1.14869835-2.81349953e-17j 0.23205506+1.09173320e+00j]
>>> print(gamma2([0.1,1/3]))
[1.36350959-5.26204809e-17j 0.19380041+1.32469442e+00j]
```

Analogously for the `mpmath` variant. Again, make sure to use proper input types, to ensure accurate outputs.

```python
>>> from ComCheb.Contours import mpLemniscate 
>>> from mpmath import mp
>>> mp.pretty = True                         # pretty display of mpmath's numbers  
>>> mp.dps = 30                              # set working precision to 30 digits  
>>> gamma  = mpLemniscate(5,'1.3')           # 5-Lemniscate, w.r.t. the level line r = 1.3

>>> print(gamma([0.1,1/3]))                  # improper input values
[(1.36350959328656937836282203999 + 3.74665817815501932087785188762e-17j)
 (0.19380040709718714902421024052 + 1.32469442224040832334398047983j)]
>>> print(gamma(['0.1','1/3']))              # proper input values
[(1.36350959328656937836282203999 - 7.28600183172118583491655094037e-32j)
 (0.193800407097187014877277528596 + 1.32469442224040831562581069932j)]
```  
### Special A
The function `mpA` provides a parameterization for the special Lemniscate
$$\mathsf A^r:= \left\{z\ :\ \left|z^3+z+1 \right|=r \right\}.$$
Currently, only the `mpmath` version is implemented and only for the two values `r=2` and `r=(31/27)**0.5`, these two variants can be accessed by `mpA(1)` and `mpA(2)`, respectively.

```python
>>> from ComCheb.Contours import mpA
>>> from mpmath import mp
>>> mp.pretty = True                         # pretty display of mpmath's numbers  
>>> mp.dps = 30                              # set working precision to 30 digits  
>>> gamma1 = mpA(1)                          # Choose special Lemniscate A1
>>> gamma2 = mpA(2)                          # Choose special Lemniscate A2

>>> print(gamma1(['0.1','1/3']))             # Special Lemniscate A1 with proper input
[(0.804556880352422305011552790866 + 1.10429448146215523261245450916j)
 (-0.341163901914009663684741869855 + 1.16154139999725193608791768725j)]
>>> print(gamma2(['0.1','1/3']))             # Special Lemniscate A2 with proper input
[(0.611136583467896373740063359864 + 1.08805459480006458167410768078j)
 (-0.0355782343631482256325970441582 + 1.00189691699355842647284861416j)]
```  
### Special B
The function `mpB` provides a parameterization for the special Lemniscate
$$\mathsf B^r:= \left\{z\ :\ \left|z^4-z^2 \right|=r \right\}.$$
Currently, only the `mpmath` version is implemented. Note that this function also offers the optional input `maxsteps`, which corresponds to a root finding process within the construction of this contour. If it happens that initialization fails, this value has to be increased. The default value is `maxsteps = 1000`.

```python
>>> from ComCheb.Contours import mpB
>>> from mpmath import mp
>>> mp.pretty = True                         # pretty display of mpmath's numbers  
>>> mp.dps = 30                              # set working precision to 30 digits  
>>> gamma  = mpB('5/4')                      # Choose special Lemniscate B

>>> print(gamma(['0.1','1/3']))              # Special Lemniscate B with proper input
[(0.79666840952708593119645420641 + 0.616140835375106842563006005861j)
 (1.05379442068940989002679349618 - 0.449704781620415922518570400332j)]
``` 
## Remez
The functions `Remez` and `mpRemez` call Tang's adaptation of the Remez algorithm. This algorithm computes the Chebyshev polynomial for any given continuous contour. Both variants work essentially the same, although `mpRemez` offers more optional arguments. 

Note that these computations, especially in high-precision, are very expensive. To optimize efficiency we can exploit symmetries of the given contour. In fact, if the contour $\Gamma$ is conjugate symmetric, that is $z \in \Gamma$ whenever $\overline z\in\Gamma$, then coefficients of the $n$-th Chebyshev polynomial $T_n$ are real. Further, if $\Gamma$ is $m$-rotational symmetric, that means that if $z\in\Gamma$ so are $e^{2\pi ik/m} z$ for $k=1,\dots,m-1$, then $T_n$ is of the form
$$T_n(z)= z^\ell Q_k(z^m) \tag{5}$$
for $n=km+\ell$ and some monic polynomial $Q_k$ of degree $k$ see Lemma 3.1 of [1]. Thus, it is sufficient to use Tang's algorithm to find the polynomial $Q_k$ instead of $T_n$, which significantly speeds up the calculation. In the following, $m$ will denote the rotational symmetric constant.

Both Remez implementation offer the following input data:   
- `gamma` : a parameterization $\gamma:[0,1]\to \mathbb C$ of $\Gamma$. Make sure, that $\gamma$ is able to handle numpy array inputs, and in case of `mpmath` also the input of `mp.mpf`.
- `N` : an integer, which describes the order $n$ of the wanted Chebyshev polynomial $T_n$.
Further, these optional settings are available.
- `t` : an array of $2k+1$ points in $[0,1]$.  The default is `None`, in this case random values are used. The array `t` will be used as initial parameter for $\mathbf t^{(0)}$ according to Section 1.
- `a` : an array of $2k+1$ points in $[0,2\pi)$. The default is `None`, in this case random values are used. The array `a` will be used as initial parameter for $\mathbf \alpha^{(0)}$ according to Section 1.
- `rc` : a boolean, default is `False`. It is short for 'real coefficients'. Set this to  `True`, whenever $\Gamma$ is conjugate symmetric.
- `symmetry` : integer. This value defines the rotational symmetry $m$. Use this, whenever $\Gamma$ is $m$-rotational symmtric. This setting greatly improves the performance.
- `prec` : an float (default 1e-10). This defines the wanted accuracy according to (3) and (4).
- `maxit` : integer, determines the maximum number of iterations for the Remez routine.
- `plot` : bool (default `False`). If `True`, then a plot will be created showing on the left the absolute value of the computed Chebyshev polynomial and on the right the relative error according to (3) with respect to the iteration number.
- `pinfo`: bool (default `False`). This is short for 'print info'. If set to `True` then additional information are printed while the Remez routine is running. Within the printed data `h_p` refers to $\|p_{\mathbf t^{(\nu)},\mathbf \alpha^{(\nu)}}\|_{\Gamma,\infty}$ and `h_D` denotes the value for $h(\mathbf t^{(\nu)},\mathbf \alpha ^{(\nu)})$, both with respect to the notation in Section 1.

Furthermore, the routine `mpRemez` offers the additional parameters `samples` (integer with default `1001`) and `reps` (integer with default `10`). These are used for determining the maximum of $|p_{\mathbf t^{(\nu)},\mathbf \alpha^{(\nu)}}\circ \gamma|$.  We first discretise [0,1] at `samples`-many equidistant points $\{x_0,\dots\}$, then we take the value $x_k$ for which the given function is maximal. We refine the the approximate maximum $x_k$ by discretizing $[x_{k-1},x_{k+1}]$ into 201 equidistant points. We repeat this refining process `reps`-times. It is suggested to set `reps`$\approx-\log_{10}($`prec`) to ensure an accurate result.

Both Remez routines will provide the following output:
- `t` : numpy array containing the reference points $\mathbf t \in [0,1]^{2k+1}$ according to Section 1 applied to the function $Q_k$ as in (5).
- `a` : numpy array containing the angles $\mathbf \alpha \in [0,2\pi)^{2k+1}$ according to Section 1 applied to the  function $Q_k$ as in (5).
- `L` : numpy array containing the coefficients of $Q_k$ as in (5).
- `max_div` : `float` or `mp.mpf` containing the Chebyshev norm of the determined polynomial.
- `rel_err` : `float` or `mp.mpf` describing the achieved relative error according to (3) and (4).

Note, if one wants to evaluate the determined Chebyshev polynomial, it has to be reconstructed using (5). Since the returned coefficients are given with respect to $Q_k$ and not the Chebyshev polynomial $T_n$. The easiest way to do so is to invoke the function `Cheby` described later.

```python
>>> import numpy as np
>>> import ComCheb as CC

>>> m = 4                                # Sets the order of the polygon and symmetry
>>> N = 10                               # Sets the order of the Chebyshev polynomial
>>> gamma = CC.Contours.Polygon(m)       # Define a polygon
>>> np.random.seed(1001)                 # In order to recreate the results
>>> t,a,L,cheb_err,rel_err = CC.Remez(gamma,N,rc=1,symmetry=m,pinfo=True)
Iteration        h_p                     h_D                     relative error
1                3.85229176e-01          1.35110797e-01          1.85120942e+00
2                3.90480082e-01          1.82971941e-01          1.13409816e+00
3                1.87042179e-01          1.85073131e-01          1.06392998e-02
4                1.85905187e-01          1.85312898e-01          3.19615253e-03
5                1.85332296e-01          1.85323051e-01          4.98861458e-05
6                1.85323052e-01          1.85323052e-01          2.16477434e-09
7                1.85323052e-01          1.85323052e-01          4.49305826e-16
```

In a similar manner we might use the `mpmath` variant.

```python
>>> from mpmath import mp
>>> import ComCheb as CC
>>> mp.pretty = True                     # Use mpmath's pretty printing
>>> mp.dps = 50                          # Set mpmath's working precision to 50 digits

>>> m = 4                                # Set the order of the polygon and symmetry
>>> N = 10                               # Set the order of the Chebyshev polynomial
>>> prec = 1e-30                         # Set the wanted accuracy to 1e-30
>>> gamma = CC.Contours.mpPolygon(m)     # Set polynomial contour
>>> t,a,L,cheb_err,rel_err = CC.mpRemez(gamma,N,rc=1,symmetry=m,pinfo=True,reps=30,prec=prec)
Iteration        h_p                     h_D                     relative error
1                1.28219776e+00          -1.53184007e-02         -8.47031091e+01
2                2.87395540e-01           9.72936966e-03          2.85389681e+01
3                3.36255910e-01           4.44786993e-02          6.55993128e+00
4                1.87057321e-01           1.83446734e-01          1.96819357e-02
5                1.85334163e-01           1.85310848e-01          1.25815435e-04
6                1.85323052e-01           1.85323051e-01          6.56003102e-09
7                1.85323052e-01           1.85323052e-01          1.78652523e-17
8                1.85323052e-01           1.85323052e-01          1.32499951e-34
```
## Chebyshev Evaluation
Using the function `Cheby` we obtain an executable function which evaluates the Chebyshev polynomial computed by `Remez` or `mpRemez` according to (5), hence it is necessary to feed the coefficients of $Q_k$, the degree of $n$ of the Chebyshev polynomial as well as the symmetry constant $m$. The function `Cheby` offers the opportunity to either determine only the Chebyshev polynomial $T_n$ or to automatically include the given contour, that means it will compute $T_n\circ \gamma$, while $\gamma$ is a parameterization of the contour. The constructed function is able to handle scalar inputs as well as one-dimensional lists, numpy arrays or mpmath matrices.

```python
>>> import ComCheb as CC

>>> m = 4
>>> N = 10
>>> gamma = CC.Contours.Polygon(m)
>>> _,_,L,_,_ = CC.Remez(gamma,N,rc=1,symmetry=m)

>>> Tn1 = CC.Cheby(N,L,symmetry=m)
>>> Tn2 = CC.Cheby(N,L,symmetry=m,gamma=gamma)

>>> print(Tn1(gamma([0.1,0.5])))
array([0.16867764+7.28032249e-02j, 0.18532305-5.85088766e-16j])
>>> print(Tn2([0.1,0.5]))
array([0.16867764+7.28032249e-02j, 0.18532305-5.85088766e-16j])
```

## Roots
The function `Roots` compute the roots of the Chebyshev polynomial $T_n$, while the input is given according to (5), thus it needs the coefficients of $Q_k$, the order $n$ as well as the symmetry constant $m$. This routine automatically distinguishes between `numpy` and `mpmath` inputs.

```python
>>> import ComCheb as CC

>>> m = 4
>>> N = 10
>>> gamma = CC.Contours.Polygon(m)
>>> _,_,L,_,_ = CC.Remez(gamma,N,rc=1,symmetry=m)
>>> CC.Roots(L,N,m) 
array([ 5.69811814e-01+0.00000000e+00j,  9.43617708e-01+0.00000000e+00j,
        3.26000093e-18+5.69811814e-01j,  5.39861499e-18+9.43617708e-01j,
       -5.69811814e-01+6.52000185e-18j, -9.43617708e-01+1.07972300e-17j,
        2.18509070e-17-5.69811814e-01j,  3.61854603e-17-9.43617708e-01j,
        0.00000000e+00+0.00000000e+00j,  0.00000000e+00+0.00000000e+00j])
```

And similarly for the `mpmath` variant.

```python
>>> from mpmath import mp
>>> import ComCheb as CC
>>> mp.pretty = True                       # Use mpmath's pretty print
>>> mp.dps    = 30                         # Set mpmaths working precision to 30 digits

>>> m = 4
>>> N = 10
>>> gamma = CC.Contours.mpPolygon(m)       # Define a "square"
>>> _,_,L,_,_ = CC.mpRemez(gamma,N,rc=1,symmetry=m,reps =10)
>>> CC.Roots(L,N,m)                                             
array([(0.569811814174935659510738182485 + 0.0j),
       (0.943617707760627741262321245246 + 0.0j),
       (4.83110824650575794968878020126e-32 + 0.569811814174935659510738182485j),
       (8.00039447429160313690780934768e-32 + 0.943617707760627741262321245246j),
       (-0.569811814174935659510738182485 + 9.66221649301151589937756040252e-32j),
       (-0.943617707760627741262321245246 + 1.60007889485832062738156186954e-31j),
       (-1.44933247395172738490663406038e-31 - 0.569811814174935659510738182485j),
       (-2.4001183422874809410723428043e-31 - 0.943617707760627741262321245246j),
       0, 0], dtype=object)
```