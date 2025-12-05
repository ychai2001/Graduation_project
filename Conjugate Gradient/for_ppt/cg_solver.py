import numpy as np
import numba as nb
import cupy as cp
import cupyx

def make_spmv(A):
    A = A.tocsr()
    row_ptr = A.indptr
    col_ind = A.indices
    data    = A.data.astype(np.float32, copy=False)
    n       = A.shape[0]

    def spmv(x, y):
        for i in nb.prange(n):
            s = 0.0
            for k in range(row_ptr[i], row_ptr[i+1]):
                s += data[k] * x[col_ind[k]]
            y[i] = s

    return nb.jit(nopython=True, fastmath=True, parallel=True)(spmv)
    
@nb.njit(parallel=True, fastmath=True)
def saxpby(a, x, b, y):
    n = x.shape[0]
    for i in nb.prange(n):
        y[i] = a * x[i] + b * y[i]

    
def cg_cpu(A, b, x0=None, tol=1e-5, atol=None, itmax=None, verbose=False, callback=None):
    spmv = make_spmv(A)

    bnorm = np.linalg.norm(b)
    if bnorm == 0:
        return b, 0
    if atol is None:
        tol_eff = tol * float(bnorm)
    else:
        tol_eff = max(float(atol), tol * float(bnorm))

    if itmax == None:
        itmax = len(b)*10

    if x0 is None:
        x = np.zeros_like(b)
    else:
        x = x0.copy()

    r  = b - A.dot(x)
    d  = r.copy()
    Ad = np.zeros_like(r)

    r2 = np.dot(r, r)
    r2_old = r2

    for it in range(itmax):

        if np.sqrt(r2) < tol_eff:
            break

        if it > 0:
            beta = r2 / r2_old
            saxpby(1.0, r, beta, d)  # d = r + beta*d

        spmv(d, Ad)  # Ad = A*d

        alpha = r2 / np.dot(d, Ad)

        saxpby(alpha, d, 1.0, x)    # x = x + alpha*d
        saxpby(-alpha, Ad, 1.0, r)  # r = r - alpha*Ad

        r2_old = r2
        r2 = np.dot(r, r)

        if callback is not None:
            callback(x)

        if verbose:
            print(it, np.sqrt(r2))

    return x, it + 1

def cg_cp(A, b, M = None, x0=None, tol=1e-5, itmax=None, atol=None, verbose=False, callback=None):
    assert cupyx.scipy.sparse.isspmatrix(A), "A must be sparse"

    b = b.astype(cp.float32)
    x = cp.zeros_like(b) if x0 is None else x0.astype(cp.float32)

    bnorm = cp.linalg.norm(b)
    if bnorm == 0:
        return b, 0
    if atol is None:
        tol_eff = tol * float(bnorm)
    else:
        tol_eff = max(float(atol), tol * float(bnorm))

    if itmax == None:
        itmax = len(b)*10

    r = b - A @ x
    r0 = cp.linalg.norm(r)

    if r0 == 0 :
        return x, 0, 0.0
    
    if atol is None:
        tol_eff = tol * float(r0)
    else: 
        tol_eff = max(float(atol), tol*float(r0))

    if M is None:
        z = r
    else:
        z = M(r)         

    d = z.copy()
    rz = cp.vdot(r, z).real

    for k in range(itmax):
        Ad = A @ d

        denom = cp.vdot(d, Ad).real  
        if denom == 0:
            r_true = cp.linalg.norm(b - A @ x)
            return x, k, float(r_true)

        alpha = rz / denom
        x += alpha * d
        r -= alpha * Ad
        r_true = cp.linalg.norm(r)

        if M is None:
            z = r
        else:
            z = M(r)

        rz_new = cp.vdot(r, z).real
        beta = rz_new / rz
        d = z + beta * d
        rz = rz_new

        if callback is not None:
            callback(x)

        if r_true <= tol_eff:
            return x, k+1
        
        if verbose:
            print(k+1, f"{r_true:.3e}")


    return x, k+1
