import numpy as np
from scipy import interpolate
import sys

def create_coarse_grid_coord(nx,nz,dx,dz,vstart,vend,f,scale=4.,water_nz=0):

    z=np.array(list(map(lambda x: x*dz, np.arange(nz)))) 
    z2= np.tile(np.array([z]).T,(1,nx))

# create a prior model
    v=np.linspace(vstart,vend,(nz-water_nz))
    water_column=np.full(shape=water_nz,fill_value=1500.,dtype=np.float32)
    v=np.concatenate((water_column,v))
    v2= np.tile(np.array([v]).T,(1,nx))

# frequency of the data (Hertz)
    lam=v2/f

    vert_res=( lam/4. )
    horiz_res=( np.sqrt(lam*(z2/2)) )
    vert_res= vert_res*scale 
    horiz_res= horiz_res*scale
    print('Build grid space')
    print('using initial model with v lin. increasing with depth')

    print('Build vertical grid space')
    print('using lambda=v/f')

    Z=np.array([])
    Z=np.append(Z, [0])
    DZ=np.array([])
    DZ=np.append(DZ,vert_res[0,0])
    Z=np.append(Z,Z[0]+DZ[-1])
    i=1

    while (Z[i] < (nz-1)*dz):
          idx = int(Z[i]//dz)
          peso= (Z[i]/dz) % 1.
          DZ=np.append(DZ,((vert_res[idx,0]*(1.- peso))+ (vert_res[idx+1,0]*peso)))
          Z=np.append(Z,Z[i]+DZ[-1])
          i=i+1

    Z=np.delete(Z,-1)

    print('Build horizontal grid space')
    print('using R=sqrt(lambda*z/2)')

    DX=np.array([])
    for i in range(Z.shape[0]):
        lambda_=DZ[i]*4.
        DX=np.append(DX,np.sqrt((lambda_*Z[i])/2.))

    print('Build grid coordinates')

# cut off the surface
    Z=np.delete(Z,0)
    DZ=np.delete(DZ,0)
    DX=np.delete(DX,0)

    v_array=np.interp(Z,z2[:,0],v2[:,0])
    zp=np.array([Z[0],Z[-1]])
# upper bound
    fp=np.array([v_array[0]+v_array[0]*0.2, v_array[-1]+v_array[-1]*0.25])
    u_bound=np.interp(Z,zp,fp)
# lower bound
    fp=np.array([v_array[0]-v_array[0]*0.15, v_array[-1]-v_array[-1]*0.35])
    l_bound=np.interp(Z,zp,fp)

    X=np.array([])
    X=np.append(X, [0])
    NX=int((nx-1)*dx//DX[0])
    NZ=Z.shape[0]

    X=np.array(list(map(lambda x: x*DX[0], np.arange(NX)))) 

    x_coord=np.array([])
    z_coord=np.array([])
    ub=np.array([])
    lb=np.array([])
    ista=0
    for iz in range(NZ):
        NX=int((nx-1)*dx//DX[iz])
        resto=((nx-1)*dx) % DX[iz] 
        for j in range(ista,ista+NX):
            ub=np.append(ub,u_bound[iz])
            lb=np.append(lb,l_bound[iz]) 
        ista=NX        
        x_coord=np.append(x_coord, list(map(lambda x: (x * DX[iz]) + resto/2, np.arange(NX))))
        z_coord=np.append(z_coord, np.tile(Z[iz],(1,NX)))     

    print('this coarse grid has ',len(x_coord),' points')

    return x_coord, z_coord, lb, ub

def coarse2fine(finegrid_nx, finegrid_nz, finegrid_dx, finegrid_dz, coarse_model,x_coord,z_coord):
#
# Coarse to fine from an irregular set of spread out point to a fine
# regular grid
#
# INPUT:
# coarse_model  :  values of the model at positions (x_coord,z_coord)
# x_coord       :  x positions of the irregular set of points
# z_coord       :  z positions of the irregular set of points
#
# by O. Mojica, SENAI CIMATEC, 2019

    finegrid_x=np.array(list(map(lambda x: x*finegrid_dx, np.arange(finegrid_nx)))) # coordinate x coarse grid
    finegrid_z=np.array(list(map(lambda x: x*finegrid_dz, np.arange(finegrid_nz)))) # coordinate z coarse grid

    jump=np.diff(z_coord) # points where there are the jump in depth of the coarse grid
    idx=np.array([0])
    idx=np.append(idx, np.where(jump != 0)[0])
    z_jump=np.array([z_coord[0]])
    idx2=idx+1
    z_jump=np.append(z_jump,z_coord[idx2[1:]])

    fine_model_along_x=np.zeros((idx.shape[0]+2, finegrid_x.shape[0]))

    for i in range(idx.shape[0]-1):
        if i == 0:
           x_at_fixed_depth=np.array([0,finegrid_x[-1]])
           x_at_fixed_depth=np.insert(x_at_fixed_depth,1,x_coord[idx[i]:idx[i+1]+1])

           coarse_model_at_fixed_depth=np.array([coarse_model[idx[i]],coarse_model[idx[i+1]]])
           coarse_model_at_fixed_depth=np.insert(coarse_model_at_fixed_depth,1,coarse_model[idx[i]:idx[i+1]+1])

        else: 
           x_at_fixed_depth=np.array([0,finegrid_x[-1]])
           x_at_fixed_depth=np.insert(x_at_fixed_depth,1,x_coord[idx[i]+1:idx[i+1]+1])

           coarse_model_at_fixed_depth=np.array([coarse_model[idx[i]+1],coarse_model[idx[i+1]]])
           coarse_model_at_fixed_depth=np.insert(coarse_model_at_fixed_depth,1,coarse_model[idx[i]+1:idx[i+1]+1])

        fine_model_along_x[i+1] =np.interp(finegrid_x,x_at_fixed_depth,coarse_model_at_fixed_depth)

# last section after the last jump
    x_at_fixed_depth=np.array([0,finegrid_x[-1]])
    x_at_fixed_depth=np.insert(x_at_fixed_depth,1,x_coord[idx[i+1]+1:])

    coarse_model_at_fixed_depth=np.array([coarse_model[idx[i+1]+1],coarse_model[-1]])
    coarse_model_at_fixed_depth=np.insert(coarse_model_at_fixed_depth,1,coarse_model[idx[i+1]+1:])

    fine_model_along_x[i+2] = np.interp(finegrid_x,x_at_fixed_depth,coarse_model_at_fixed_depth)

# extrapolate the edges from the neighboring areas
    z_jump=np.insert(z_jump,0,0.)
    z_jump=np.append(z_jump,finegrid_z[-1])

    fine_model_along_x[0,:]=fine_model_along_x[1,:]
    fine_model_along_x[-1,:]=fine_model_along_x[-2,:]

# interpolate array in the z direction
    fine_model=np.zeros((finegrid_z.shape[0],finegrid_x.shape[0]))
    for j in range(finegrid_nx):
        for i in range(finegrid_nz):
            fine_model[i,j]=interp2d(z_jump,finegrid_x,fine_model_along_x,(i*finegrid_dz),(j*finegrid_dx))

    return fine_model.T

def interp2d(x,y,array,x0,y0,bounds_error=True,fill_value=None):
    # Bilinear interpolation of array = f(x,y) at (x0,y0)

    if not bounds_error:
       if (fill_value is not None): 
          fill_value_tmp = fill_value
       else:
          fill_value_tmp = 0.

    if(x.size != array.shape[0]): sys.exit('x does not match array')
    if(y.size != array.shape[1]): sys.exit('y does not match array')

    i1 = locate(x,x0)
    i2 = i1 + 1
    j1 = locate(y,y0)
    j2 = j1 + 1

    if(i1==-1):
      if(bounds_error):
        print('("ERROR: Interpolation out of bounds : "%f" in ["%f":"%f"]")'%(x0,x[0],x[-1])) 
        sys.exit()
      else:
        return fill_value_tmp
          
    if(j1==-1):
       if(bounds_error):
         print('("ERROR: Interpolation out of bounds : "%f" in ["%f":"%f"]")'%(y0,y[0],y[-1])) 
         sys.exit()
       else:
         return fill_value_tmp

    norm = 1. / (x[i2] - x[i1]) / (y[j2]-y[j1])

    value= array[i1,j1] * (x[i2]-x0) * (y[j2]-y0) * norm + \
           array[i2,j1] * (x0-x[i1]) * (y[j2]-y0) * norm + \
           array[i1,j2] * (x[i2]-x0) * (y0-y[j1]) * norm + \
           array[i2,j2] * (x0-x[i1]) * (y0-y[j1]) * norm

    return value

def locate(xx,x):
    # Locate a value in a sorted array

    ascnd = bool(xx[-1] >= xx[0])
    jl=0
    ju=len(xx)

    while True:
        if (ju-jl <= 1): break
        jm= (ju+jl)//2
        if (ascnd == bool(x >= xx[jm])):
          jl=jm
        else:
          ju=jm

    if (x == xx[0]):
       return 0
    elif (x == xx[-1]):
       return len(xx)-2
    elif(ascnd and (x > xx[-1] or x < xx[0])):
       return -1
    elif( not ascnd and (x < xx[-1] or x > xx[0])):
       return -1
    else:
       return jl

