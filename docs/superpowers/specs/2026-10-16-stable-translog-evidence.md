# Evidence for the stable_translog design audit

All numbers below were produced on 2026-10-16 on hmg-office. Scripts are reproduced in
full so the reviewer can re-derive them; they need only numpy.

## 1. Baseline tests of the current transition functions

`pixi run -e tests-cpu pytest tests/test_transition_functions.py -v` at revision 94aa213:
see `2026-10-16-stable-translog-baseline-tests.log` (25 passed).

## 2. Divergence in the health-cognition application (translog, unrestricted diagnostics)

Stability diagnostics of two estimation runs *before* the diagnostics tables were
restricted to person-periods with at least one measurement. Columns: SD of the filtered
state and max |state| per factor (P physical, S sensory, M psychological, C cognition),
then explosive persons (|state| > 5) now / cumulative / share / overflow. Linear
transition models in the same runs show max |state| about 3 at every age and zero
explosive persons.

### Run "sensnutr", 1-mix translog, alive loads on physical and cognition
68 & 0.38 & 0.38 & 0.49 & 0.28 & 1.88 & 1.95 & 2.72 & 1.36 & 0 & 0 & 0.000 & 0
70 & 0.45 & 0.47 & 0.54 & 0.30 & 2.17 & 2.24 & 3.30 & 1.37 & 0 & 0 & 0.000 & 0
72 & 0.50 & 0.53 & 0.60 & 0.32 & 2.39 & 2.58 & 3.69 & 1.51 & 0 & 0 & 0.000 & 0
74 & 0.55 & 0.57 & 0.65 & 0.33 & 2.65 & 2.59 & 4.74 & 2.26 & 0 & 0 & 0.000 & 0
76 & 0.57 & 0.60 & 0.68 & 0.35 & 2.56 & 2.52 & 5.80 & 3.47 & 9 & 9 & 0.001 & 0
78 & 0.60 & 0.62 & 0.73 & 0.37 & 2.70 & 2.84 & 7.74 & 6.99 & 24 & 24 & 0.001 & 0
80 & 0.63 & 0.65 & 0.84 & 0.48 & 10.07 & 9.04 & 16.63 & 28.65 & 55 & 55 & 0.003 & 0
82 & 2.31 & 1.78 & 3.86 & 2.66 & 281.15 & 209.73 & 466.99 & 331.21 & 120 & 120 & 0.007 & 0
84 & 75.90 & 4.62 & 812.50 & 228.84 & 9.7e+03 & 581.41 & 1.0e+05 & 2.9e+04 & 209 & 209 & 0.013 & 0
86 & 1.3e+07 & 7.5e+06 & 6.2e+07 & 8.0e+06 & 1.7e+09 & 9.6e+08 & 8.0e+09 & 1.0e+09 & 332 & 332 & 0.020 & 0
88 & 6.6e+16 & 1.1e+17 & 2.8e+17 & 8.0e+15 & 8.4e+18 & 1.4e+19 & 3.6e+19 & 1.0e+18 & 474 & 474 & 0.029 & 0
90 & 1.7e+36 & 2.2e+36 & 6.9e+36 & 5.3e+35 & 2.2e+38 & 2.8e+38 & 8.8e+38 & 6.8e+37 & 623 & 623 & 0.038 & 0
92 & 8.7e+74 & 1.4e+75 & 3.5e+75 & 8.4e+73 & 1.1e+77 & 1.7e+77 & 4.6e+77 & 1.1e+76 & 815 & 815 & 0.049 & 0

### Run "loadtrend", 2-mix translog, alive loads on all four states
68 & 0.38 & 0.37 & 0.61 & 0.28 & 1.93 & 1.91 & 3.30 & 1.35 & 0 & 0 & 0.000 & 0
70 & 0.45 & 0.44 & 0.65 & 0.30 & 2.09 & 2.15 & 3.39 & 1.38 & 0 & 0 & 0.000 & 0
72 & 0.50 & 0.49 & 0.69 & 0.32 & 2.16 & 2.13 & 3.47 & 1.45 & 0 & 0 & 0.000 & 0
74 & 0.54 & 0.52 & 0.72 & 0.33 & 2.28 & 2.28 & 4.65 & 1.65 & 0 & 0 & 0.000 & 0
76 & 0.57 & 0.55 & 0.74 & 0.34 & 2.27 & 2.23 & 5.42 & 1.94 & 1 & 1 & 0.000 & 0
78 & 0.59 & 0.58 & 0.79 & 0.35 & 2.32 & 2.30 & 6.17 & 2.44 & 3 & 3 & 0.000 & 0
80 & 0.62 & 0.60 & 0.83 & 0.36 & 2.39 & 2.77 & 7.53 & 3.25 & 6 & 7 & 0.000 & 0
82 & 0.61 & 0.61 & 0.85 & 0.35 & 2.26 & 4.33 & 13.77 & 4.21 & 11 & 12 & 0.001 & 0
84 & 0.62 & 0.62 & 1.01 & 0.35 & 5.87 & 8.96 & 43.95 & 8.64 & 35 & 36 & 0.002 & 0
86 & 1.03 & 0.91 & 4.35 & 0.66 & 100.71 & 83.76 & 520.40 & 70.10 & 97 & 98 & 0.006 & 0
88 & 1.98 & 124.49 & 907.09 & 25.96 & 220.61 & 1.6e+04 & 1.2e+05 & 3.3e+03 & 175 & 176 & 0.011 & 0
90 & 1.4e+07 & 5.2e+06 & 1.9e+07 & 3.2e+06 & 1.8e+09 & 6.6e+08 & 2.4e+09 & 4.1e+08 & 260 & 261 & 0.016 & 0
92 & 7.5e+15 & 2.6e+15 & 2.2e+16 & 5.2e+15 & 9.6e+17 & 3.4e+17 & 2.8e+18 & 6.6e+17 & 368 & 369 & 0.022 & 0

The data session established for run D that every explosive person sits on rows after
death or after the last interview (forward predictions without measurement updates);
no likelihood-relevant row is affected. After restricting the tables to measured rows,
max |state| is below 4.7 at every age in runs D, E, F.

## 3. Estimated own-state coefficients, run E main translog model (age group 68-73)

Psychological capacity: constant -0.003, own linear 0.769, own square -0.103.
Cognition: constant -0.074, own linear 0.596, own square -0.112.
Full transition tables with standard errors are in the health-cognition build
`bld-2026-10-spec2/tables/no_feedback_to_investments_1mix_alive_ph_cog/pooled/transition_params/`.

## 4. Fixed points and iteration (script and output)

```python
import numpy as np, warnings
warnings.simplefilter("ignore")
np.seterr(all="ignore")
# run E main translog, 68-73, own terms only (others at zero)
cases = {"psychological": (-0.003, 0.769, -0.103), "cognition": (-0.074, 0.596, -0.112)}
for name, (c, rho, b) in cases.items():
    roots = np.roots([b, rho - 1, c])
    print(name, "fixed points", np.sort(roots), "slopes", rho + 2 * b * np.sort(roots))
    for th0 in (-2.0, -2.5, -3.0, 3.0, 5.0):
        th = np.float64(th0)
        path = [th]
        for _ in range(20):
            th = np.float64(c + rho * th + b * th**2)
            path.append(th)
        print(f"  translog start {th0:5.1f}: after 20 = {path[-1]:.3g}")
    s = 3.0
    for th0 in (-2.5, -3.0, 5.0, 50.0):
        th = np.float64(th0)
        for _ in range(200):
            g = s * np.tanh(th / s)
            th = c + rho * th + b * g**2
        print(f"  stable  start {th0:5.1f}: after 200 = {th:.4f}")
    # max deviation stable vs translog on |theta|<=2
    x = np.linspace(-2, 2, 401); g = s*np.tanh(x/s)
    print("  max |g^2 - x^2| on [-2,2]:", np.max(np.abs(g**2 - x**2)), " at |x|=1:", abs((s*np.tanh(1/s))**2-1))
```

Output:

```
psychological fixed points [-2.22965535 -0.0130631 ] slopes [1.228309 0.771691]
  translog start  -2.0: after 20 = -0.192
  translog start  -2.5: after 20 = -1.1e+113
  translog start  -3.0: after 20 = -inf
  translog start   3.0: after 20 = -0.00772
  translog start   5.0: after 20 = -0.00793
  stable  start  -2.5: after 200 = -0.0131
  stable  start  -3.0: after 200 = -0.0131
  stable  start   5.0: after 200 = -0.0131
  stable  start  50.0: after 200 = -0.0131
  max |g^2 - x^2| on [-2,2]: 0.9432763475045336  at |x|=1: 0.06966603644422964
cognition fixed points [-3.4135887  -0.19355416] slopes [1.36064387 0.63935613]
  translog start  -2.0: after 20 = -0.194
  translog start  -2.5: after 20 = -0.196
  translog start  -3.0: after 20 = -0.201
  translog start   3.0: after 20 = -0.193
  translog start   5.0: after 20 = -0.194
  stable  start  -2.5: after 200 = -0.1935
  stable  start  -3.0: after 200 = -0.1935
  stable  start   5.0: after 200 = -0.1935
  stable  start  50.0: after 200 = -0.1935
  max |g^2 - x^2| on [-2,2]: 0.9432763475045336  at |x|=1: 0.06966603644422964
```

## 5. Candidate clipping maps: fidelity, excursion, fixed points (scripts and output)

```python
import numpy as np
np.seterr(all="ignore")
def tanh_g(x, s): return s*np.tanh(x/s)
def alg_g(x, s, p): return x/(1+np.abs(x/s)**p)**(1/p)
c, rho, b = -0.003, 0.769, -0.103   # psychological 68-73, run E
print("name        s  p | rel.err g^2 at |x|=1,2,2.5 | sup|g| | max excursion from start -3,-5,-50 (200 it) | periods to |theta|<0.1 from -3")
for name, fn, s, p in [("tanh",tanh_g,3,None),("tanh",tanh_g,5,None),("tanh",tanh_g,10,None),
                       ("alg",alg_g,3,4),("alg",alg_g,4,4),("alg",alg_g,4,8),("alg",alg_g,5,8)]:
    g = (lambda x: fn(x,s)) if p is None else (lambda x: fn(x,s,p))
    errs = [abs(g(x)**2-x**2)/x**2 for x in (1,2,2.5)]
    exc=[]; 
    for th0 in (-3,-5,-50):
        th=np.float64(th0); m=0; n=None
        for t in range(200):
            th = c+rho*th+b*g(th)**2; m=max(m,abs(th))
            if n is None and abs(th)<0.1: n=t+1
        exc.append((m,n))
    print(f"{name:6} {s:4} {str(p):>4} | {errs[0]:.3f} {errs[1]:.3f} {errs[2]:.3f} | {s:4} | " + " ".join(f"{m:7.2f}" for m,_ in exc) + f" | {exc[0][1]}")
```

Output:

```
name        s  p | rel.err g^2 at |x|=1,2,2.5 | sup|g| | max excursion from start -3,-5,-50 (200 it) | periods to |theta|<0.1 from -3
tanh      3 None | 0.070 0.236 0.330 |    3 |    2.85    4.65   39.38 | 26
tanh      5 None | 0.026 0.098 0.146 |    5 |   10.51   10.51   41.03 | None
tanh     10 None | 0.007 0.026 0.040 |   10 |   44.58   44.58   48.75 | None
alg       3    4 | 0.006 0.086 0.179 |    3 |    2.97    4.72   39.38 | 44
alg       4    4 | 0.002 0.030 0.069 |    4 |    6.74    6.74   40.10 | None
alg       4    8 | 0.000 0.001 0.006 |    4 |    7.13    7.13   40.10 | None
alg       5    8 | 0.000 0.000 0.001 |    5 |   11.16   11.16   41.03 | None
```

`None` in the last column means the iteration never returned to within 0.1 of the
translog fixed point from a start at -3: it converged to a second, spurious attractor.

```python
import numpy as np
np.seterr(all="ignore")
c, rho, b = -0.003, 0.769, -0.103
def fps(f, lo=-60, hi=60, n=600001):
    x=np.linspace(lo,hi,n); d=f(x)-x; idx=np.where(np.sign(d[:-1])!=np.sign(d[1:]))[0]
    out=[]
    for i in idx:
        r=x[i]; h=1e-5; slope=(f(r+h)-f(r-h))/(2*h); out.append((round(float(r),3), "stable" if abs(slope)<1 else "unstable"))
    return out
G={
 "translog": lambda x: x,
 "tanh s=3": lambda x: 3*np.tanh(x/3),
 "alg s=3 p=4": lambda x: x/(1+np.abs(x/3)**4)**0.25,
 "alg s=4 p=8": lambda x: x/(1+np.abs(x/4)**8)**0.125,
 "alg s=2.5 p=8": lambda x: x/(1+np.abs(x/2.5)**8)**0.125,
}
for k,g in G.items():
    print(f"{k:14}", fps(lambda x: c+rho*x+b*g(x)**2))
# decaying nonlinearity x^2 exp(-(x/s)^2)
for s in (3,4):
    h=lambda x: x**2*np.exp(-(x/s)**2)
    print(f"decay s={s}    ", fps(lambda x: c+rho*x+b*h(x)), " rel.err at 2:", abs(h(2.0)-4)/4)
```

Output:

```
translog       [(-2.23, 'unstable'), (-0.013, 'stable')]
tanh s=3       [(-0.013, 'stable')]
alg s=3 p=4    [(-0.013, 'stable')]
alg s=4 p=8    [(-7.13, 'stable'), (-2.235, 'unstable'), (-0.013, 'stable')]
alg s=2.5 p=8  [(-0.013, 'stable')]
decay s=3     [(-0.013, 'stable')]  rel.err at 2: 0.3588196115700454
decay s=4     [(-0.013, 'stable')]  rel.err at 2: 0.22119921692859512
```

```python
import numpy as np
np.seterr(all="ignore")
c, rho, b = -0.003, 0.769, -0.103
def fps(f, lo=-60, hi=60, n=600001):
    x=np.linspace(lo,hi,n); d=f(x)-x; idx=np.where(np.sign(d[:-1])!=np.sign(d[1:]))[0]
    return [(round(float(x[i]),3), "stable" if abs((f(x[i]+1e-5)-f(x[i]-1e-5))/2e-5)<1 else "unstable") for i in idx]
for s,p in [(3,8),(3,6),(3.5,8),(2.5,8),(3,4)]:
    g=lambda x: x/(1+np.abs(x/s)**p)**(1/p)
    errs=[abs(g(x)**2-x**2)/x**2 for x in (1,2,2.5,3)]
    print(f"alg s={s} p={p}: fixed points {fps(lambda x: c+rho*x+b*g(x)**2)}  rel.err g^2 at 1,2,2.5,3 = {[round(e,3) for e in errs]}")
```

Output:

```
alg s=3 p=8: fixed points [(-3.915, 'stable'), (-2.292, 'unstable'), (-0.013, 'stable')]  rel.err g^2 at 1,2,2.5,3 = [np.float64(0.0), np.float64(0.01), np.float64(0.051), np.float64(0.159)]
alg s=3 p=6: fixed points [(-3.709, 'stable'), (-2.419, 'unstable'), (-0.013, 'stable')]  rel.err g^2 at 1,2,2.5,3 = [np.float64(0.0), np.float64(0.028), np.float64(0.092), np.float64(0.206)]
alg s=3.5 p=8: fixed points [(-5.436, 'stable'), (-2.246, 'unstable'), (-0.013, 'stable')]  rel.err g^2 at 1,2,2.5,3 = [np.float64(0.0), np.float64(0.003), np.float64(0.016), np.float64(0.062)]
alg s=2.5 p=8: fixed points [(-0.013, 'stable')]  rel.err g^2 at 1,2,2.5,3 = [np.float64(0.0), np.float64(0.038), np.float64(0.159), np.float64(0.341)]
alg s=3 p=4: fixed points [(-0.013, 'stable')]  rel.err g^2 at 1,2,2.5,3 = [np.float64(0.006), np.float64(0.086), np.float64(0.179), np.float64(0.293)]
```
