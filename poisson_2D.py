#%%
"""
Implementation adapted from https://github.com/zaman13/Poisson-solver-2D
"""

import h5py
import numpy as np
import pylab as py
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from pathlib import Path


from magtense.utils import plot_magfield

def curl(field):
    field = np.swapaxes(field,1,2)
    Fx_y = np.gradient(field[0], axis=1)
    Fy_x = np.gradient(field[1], axis=0)

    if field.shape[0]== 3:
        Fx_z = np.gradient(field[0], axis=2)
        Fy_z = np.gradient(field[1], axis=2)
        Fz_x = np.gradient(field[2], axis=0)
        Fz_y = np.gradient(field[2], axis=1)
        
        curl_vec = np.stack([Fz_y-Fy_z, Fx_z-Fz_x, Fy_x-Fx_y], axis=0)[:,:,:,1]
    else:
        curl_vec = Fy_x - Fx_y
    
    return curl_vec


def div(field):
    field = np.swapaxes(field,1,2)
    Fx_x = np.gradient(field[0], axis=0)
    Fy_y = np.gradient(field[1], axis=1)

    if field.shape[0]== 3:
        Fz_z = np.gradient(field[2], axis=2)
        # Taking gradients of center layer only
        div = np.stack([Fx_x, Fy_y, Fz_z], axis=0)[:,:,:,1]
    else:                    
        div = np.stack([Fx_x, Fy_y], axis=0)
    
    return div.sum(axis=0), div


def grad(scalar_field):
    F_x = np.gradient(scalar_field, axis=0)
    F_y = np.gradient(scalar_field, axis=1)

    if len(scalar_field.shape) == 3:
        F_z = np.gradient(scalar_field, axis=2)
        grad = np.stack([F_x, F_y, F_z], axis=0)
    else:                    
        grad = np.stack([F_x, F_y], axis=0)
    
    return np.swapaxes(grad,1,2)


def my_contourf(x, y, f, ttl, clrmp='inferno'):
    cnt = py.contourf(x, y, f, 41, cmap=clrmp)
    for c in cnt.collections:
        c.set_edgecolor("face")
    cbar = py.colorbar()
    py.xlabel(r'$x$', fontsize=26)
    py.ylabel(r'$y$', fontsize=26)
    cbar.set_label(ttl, fontsize=26)
    py.xlim([x[0],x[-1]])
    py.ylim([y[0],y[-1]])


def my_scatter(x, y, clr, ttl='', msize=2):
    py.plot(x, y, '.', markersize=msize, color=clr)
    py.xlabel(r'$x$', fontsize=26)
    py.ylabel(r'$y$', fontsize=26)
    py.title(ttl)


def diff_mat_1D(Nx):
    # First derivative (2*dx division is required)
    D_1d = sp.diags([-1, 1], [-1, 1], shape=(Nx,Nx))
    D_1d = sp.lil_matrix(D_1d)
    # 2nd order forward difference (2*dx division is required)
    D_1d[0,[0,1,2]] = [-3, 4, -1]
    # 2nd order backward difference (2*dx division is required)
    D_1d[Nx-1,[Nx-3, Nx-2, Nx-1]] = [1, -4, 3]
    
    # Second derivative (division by dx^2 required)
    D2_1d =  sp.diags([1, -2, 1], [-1, 0, 1], shape = (Nx, Nx))
    D2_1d = sp.lil_matrix(D2_1d)
    # 2nd order forward difference (2*dx division is required)                  
    D2_1d[0,[0,1,2,3]] = [2, -5, 4, -1]
    # 2nd order backward difference (2*dx division is required)
    D2_1d[Nx-1,[Nx-4, Nx-3, Nx-2, Nx-1]] = [-1, 4, -5, 2]
    
    return D_1d, D2_1d


def diff_mat_2D(Nx, Ny):
    # 1D differentiation matrices
    Dx_1d, D2x_1d = diff_mat_1D(Nx)
    Dy_1d, D2y_1d = diff_mat_1D(Ny)

    # Sparse identity matrices
    Ix = sp.eye(Nx)
    Iy = sp.eye(Ny)

    # 2D matrix operators from 1D operators using kronecker product
    # First partial derivatives
    Dx_2d = sp.kron(Iy, Dx_1d)
    Dy_2d = sp.kron(Dy_1d, Ix)
    
    # Second partial derivatives
    D2x_2d = sp.kron(Iy, D2x_1d)
    D2y_2d = sp.kron(D2y_1d, Ix)
    
    # Return compressed Sparse Row format of the sparse matrices
    return Dx_2d.tocsr(), Dy_2d.tocsr(), D2x_2d.tocsr(), D2y_2d.tocsr()


def solve_poisson_msp(field, bnd_dirichlet=False, plot=False):
    _, Nx, Ny = field.shape
    div_field, _ = div((-1) * field)
    source = div_field.flatten()

    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    X, Y = np.meshgrid(x,y)
    Xu = X.ravel()
    Yu = Y.ravel()
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Dirichlet/Neumann boundary conditions at outerwalls
    # Type: 0 = Dirichlet, 1 = Neumann x derivative, 2 = Neumann y derivative
    bnd_ind = []
    bnd_val = []
    bnd_type = []

    # Finding outer boundary regions
    ind_L = np.squeeze(np.where(Xu==x[0]))[1:]
    ind_R = np.squeeze(np.where(Xu==x[-1]))
    ind_B = np.squeeze(np.where(Yu==y[0]))[1:]
    ind_T = np.squeeze(np.where(Yu==y[-1]))

    # x-/y-direction of magnetic field are interchanged
    neg_Hx = np.swapaxes((-1) * field,1,2)[0]
    neg_Hy = np.swapaxes((-1) * field,1,2)[1]

    bnd_ind.append(ind_L)
    bnd_val.append(-neg_Hx[0,1:])
    bnd_type.append(1)

    bnd_ind.append(ind_R)
    bnd_val.append(neg_Hx[-1,:])
    bnd_type.append(1)

    bnd_ind.append(ind_T)
    bnd_val.append(neg_Hy[:,-1])
    bnd_type.append(2)

    bnd_ind.append(ind_B)
    bnd_val.append(-neg_Hy[1:,0])
    bnd_type.append(2)
    
    if bnd_dirichlet:
        bnd_ind.append(0)
        bnd_type.append(0)
        bnd_val.append(0)

    if plot:
        clr_set = ['#eaeee0','#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#42d4f4',
            '#f032e6', '#fabed4', '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000',
            '#aaffc3', '#000075', '#a9a9a9', '#ffffff', '#000000']
        py.close('all')
        py.figure(figsize=(9,7))
        my_scatter(X, Y, clr_set[0], msize=4)

        for m in range(len(bnd_val)):
            my_scatter(Xu[bnd_ind[m]], Yu[bnd_ind[m]], clr_set[len(bnd_val)-m], msize=4)

        py.figure(figsize=(9,7))
        my_contourf(x,y, source.reshape(Ny,Nx), r'$f\,(x,y)$', 'RdBu')

    Dx_2d, Dy_2d, D2x_2d, D2y_2d = diff_mat_2D(Nx, Ny)
    I_sp = sp.eye(Nx * Ny).tocsr()
    # System matrix without boundary conditions
    L_sys = D2x_2d / dx**2 + D2y_2d / dy**2

    # Selectively replace the rows of the system matrix that correspond to boundary value points
    for m in range(len(bnd_val)):
        source[bnd_ind[m]] = bnd_val[m]

        if bnd_type[m] == 0:
            L_sys[bnd_ind[m],:] = I_sp[bnd_ind[m],:]
        if bnd_type[m] == 1:
            L_sys[bnd_ind[m],:] = Dx_2d[bnd_ind[m],:]
        if bnd_type[m] == 2:
            L_sys[bnd_ind[m],:] = Dy_2d[bnd_ind[m],:]

    # Solving
    u = spsolve(L_sys, source).reshape(Ny,Nx)

    # Recovered magnetic field
    vx = -(Dx_2d*u.ravel()).reshape(Ny,Nx)
    vy = -(Dy_2d*u.ravel()).reshape(Ny,Nx)
    rec_field = np.stack([vx, vy], axis=0)

    if plot:
        py.figure(figsize = (12,7))
        my_contourf(x, y, u, r'$u\,(x,y)$')

        py.figure(figsize = (12,7))
        my_contourf(x, y, vx, r'$|-\nabla u\,(x,y)|$', 'bwr')
        py.figure(figsize = (12,7))
        my_contourf(x, y, vy, r'$|-\nabla u\,(x,y)|$', 'bwr')
        # my_contourf(x, y, np.sqrt(vx**2 + vy**2), r'$|-\nabla u\,(x,y)|$','afmhot')
        # py.streamplot(x, y, vx, vy, color='w', density=1.2, linewidth=0.4)

    return u, rec_field

#%%

if __name__ == "__main__":
    path_orig = Path(__file__).parent.resolve() / 'data' 
    # db = h5py.File('/home/spol/Documents/repos/DeepGenerativeModelling/data/magfield_symm_256.h5', mode='r')
    db = h5py.File('data/magfield_symm_val_256.h5')
    field = db['field'][0]
    msp, rec_field = solve_poisson_msp(field, plot=True)
    db.close()
# %%
