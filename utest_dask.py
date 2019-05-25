import numpy
import sys, json
import finetocoarse
from dask_jobqueue import SLURMCluster
from dask.distributed import Client, wait

from examples.seismic import (Model, plot_velocity, AcquisitionGeometry,
                              plot_shotrecord, Receiver, RickerSource,
                              TimeAxis)
from examples.seismic.acoustic import AcousticWaveSolver

from devito import Eq, solve, Operator, TimeFunction
from devito import clear_cache
import time

def find_mem_pos(header_lines):
    for i,line in enumerate(header_lines):
        if('--mem=' in line):
            return i

def my_task(part,param,dobs,x,z,src_coor,rec_coor):
    interpolated_model=finetocoarse.coarse2fine(param['shape'][0],param['shape'][1],
                                                param['spacing'][0],param['spacing'][1],part,x,z)
    interpolated_model[:,0:param['cgrid']['water_samples']]=1500.
    error=0.
    model_part=get_true_model(interpolated_model*(1/1000.),param)
    print (len(dobs[0][0]))
    for i in  range(len(dobs[0][0])):
      dcalc=generate_shotdata_i(numpy.array([src_coor[i]]), rec_coor, param, model_part)
      res=get_value(dcalc,dobs[:,:,i])
      error += res
      clear_cache()
    return (numpy.array(error.data),)

def get_value(dcalc, dobs):
    return  0.5 * numpy.sum((dcalc - dobs)**2.) 

def get_true_model(v,param):
    ''' Define the test phantom; in this case we are using
    a simple circle so we can easily see what is going on.
    '''
    return Model(vp=v,origin=param['origin'], shape=param['shape'],
                 spacing=param['spacing'], space_order=param['space_order'], nbpml=param['nbpml'])

def generate_shotdata_i(src_coordinates,rec_coordinates,param,true_model):
    """ Inversion crime alert! Here the worker is creating the
        'observed' data using the real model. For a real case
        the worker would be reading seismic data from disk.
    """

    # Geometry 
    geometry = AcquisitionGeometry(true_model, rec_coordinates, src_coordinates,
                                   param['t0'], param['tn'], src_type='Ricker',
                                   f0=param['f0'])
    geometry.resample(param['dt'])

    # Set up solver.
    solver = AcousticWaveSolver(true_model, geometry, space_order=param['space_order'])

    # Generate synthetic receiver data from true model.
    true_d, _, _ = solver.forward(m=true_model.m)

    return true_d.data

cluster = SLURMCluster(cores=2,
                     memory="10GB", 
                     queue="standard",
                     interface="ib0", job_extra=['--time=72:00:00 --requeue'])
header_lines = cluster.job_header.split('\n')
mem_pos = find_mem_pos(header_lines)
header_lines = header_lines[:mem_pos] + header_lines[mem_pos+1:]
cluster.job_header = '\n'.join(header_lines)
print(cluster.job_script())
print("Waiting for workers")
cluster.scale(18)
client = Client(cluster)

# Get parameters. 
js=open('parameters.json') 
par=json.load(js)

# Load the data
shots = numpy.fromfile('shots.file', dtype=numpy.float32)
shots = numpy.reshape(shots, (6559, 369, 50))

# Set up source/receiver data and geometry.
src_coordinates = numpy.empty((par['nshots'], len(par['shape'])))
max_offset= (par['nshots']-1)*par['int_btw_shots']

for i in  range(par['nshots']):
  if par['nshots'] > 1: 
    src_coordinates[i,:] = [(i*max_offset/(par['nshots']-1))+par['first_src_xcoor'],par['src_depth']]
  else:
    src_coordinates[i,:] = [par['first_src_xcoor'],par['src_depth']]

rec_coordinates = numpy.empty((par['nreceivers'], len(par['shape'])))
rec_coordinates[:, 0] = numpy.linspace(0., par['shape'][0], num=par['nreceivers'])
rec_coordinates[:, 1] = par['rec_depth'] # at surface

# Compute the x, z coordinates of coarse grid points and lower and upper bounds of search space
x,z,lb,ub=finetocoarse.create_coarse_grid_coord(par['shape'][0],par['shape'][1],par['spacing'][0],
         par['spacing'][1],par['cgrid']['vstart'],par['cgrid']['vend'],par['f0']*2000.,par['cgrid']['scale'], par['cgrid']['water_samples'])

[dobs] = client.scatter([shots], broadcast=True)
params = client.scatter(par, broadcast=True)
[remote_x] = client.scatter([x], broadcast=True)
[remote_z] = client.scatter([z], broadcast=True)
[remote_src] = client.scatter([src_coordinates], broadcast=True)
[remote_rec] = client.scatter([rec_coordinates], broadcast=True)

pop=[]
print(type(pop))
for i in range(360):
 pop.append((ub-lb) * numpy.random.random_sample((len(x))) + lb)

start = time.time()
features=client.map(my_task,pop, param=params, dobs=dobs,x=remote_x,z=remote_z,src_coor=remote_src,rec_coor=remote_rec)
wait(features)
print("Fitness computation took {}".format(time.time() - start))
for fit in features:
 print(fit.result())
