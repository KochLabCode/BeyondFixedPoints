import numpy as np
from scipy import constants
from fp_solver import fokker_planck, boundary


class QuasiPotentialLandscape:
    
    """
    estimation of quasi potential landscape of a given dynamical system (currently
    optimized for two variable systems only). fokker planck solver is adapted from
    gitub repository associated with,
    
    Holubec, V., Kroy, K., and Steffenoni, S. (2019). Physically consistent numerical 
    solver for time-dependent fokkerplanck equations. Physical Review E, 99:032117
    
    https://github.com/johnaparker/fplanck
    
    """
    
    ####### solver settings and parameters
    D=0.02
    drag = 1
    temperature=D*drag/constants.k
    
    ###### spatial grid parameters
    L=1.5 # length of the spatial domain
    h=0.1  # step size
    N=int(np.ceil(L/h)) # number of grid points
    
    def __init__(self, time_point,model,input_params):
        
        '''
        inputs:
            time_point: time point at which the quasi-potential landscape is calculated
            model: model equations
            input_params: includes ctot as well was stimulus amplitudes
        
        
        function 'find_potential' returns the probability Pt (see the Fokker-Planck method)  and
        the spatial grid data.
        '''
        
        self.time_point=time_point
        self.model = model
        self.input_params=input_params
    
    def F(self,u1, u2):        
        y=[u1,u2]       
        return self.model(self.time_point, y,self.input_params)

    def random_pdf(self):
        
        '''
        generates random probability distributionn on a 2D grid.
        '''
    
        def pdf(*args):
            values = np.ones_like(args[0])
    
            for i, arg in enumerate(args):
                values *= np.random.uniform(0,1,np.shape(args)[1:])
            return values
    
        return pdf
    
    def find_potential(self):

        sim = fokker_planck(temperature=self.temperature, drag=self.drag, extent=[self.L, self.L],
                    resolution=self.h, boundary=boundary.reflecting, force=self.F)
    
        ### time-evolved solution
        pdf = self.random_pdf()
        p0 = pdf(*sim.grid)
    
        Tmax=100
        dt=0.01
        Nsteps = int(Tmax/dt)
        Pt = sim.propagate_interval(pdf, Tmax, Nsteps=Nsteps)
        
        return sim.grid,Pt
        
        
