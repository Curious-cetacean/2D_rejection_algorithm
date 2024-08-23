#Functions for the rejection algorithm.
#Conor Davidson 23/08/24

import numpy as np

#Func is a general 2D function from which we generate samples

#1D rejection algorithm
def rejection(N, d, func, **funcparams):
    """For N samples of the distribution function func and x range d, return N length array with
    sampled x positions. **funcparams are the keyword arguments for the general function"""
    
    samples = np.zeros(N) #returned sample array
    
    for i in range(N):
        #print("Particle: %d" % i)
        x = 0.5 #initialise so while loop will run at least once
        prob = 0.0
        rsample = 1.0
        while rsample > prob:
            x = np.random.rand()*d #generate uniform float for the x value
            #print(x)
            prob = func(x, **funcparams) #probability at this x value
            #print(prob)
            rsample = np.random.rand() #sample along this line
            #if rsample is greater than max probabiliyt at f(x), reject x and start again
            
        #x is now below prob
        samples[i] = x
        
    return samples

##Create a 2D version of the rejection algorithm
def rejection_2d(N, minx, Lx, miny, Ly, func, **funcparams):
    """For N samples of the distribution function func and x range Lx and y range Ly, return N length array with
    sampled x, y positions. **funcparams are the keyword arguments for the general function"""
    
    samples = np.zeros((N,2)) #returned sample array with x,y positions
    
    #find the maximum of the func
    max_xpoints = np.linspace(minx, Lx, 1000)
    max_ypoints = np.linspace(miny, Ly, 1000)
    X,Y = np.meshgrid(max_xpoints, max_ypoints)
    z_max = np.max(func(X,Y, **funcparams))
    
    for i in range(N):
        #print("Particle: %d" % i)
        x = 0.5 #initialise so while loop will run at least once
        prob = 0.0
        rsample = 1.0
        k = 0 #iteration counter
        while rsample > prob:
            x = minx + np.random.rand()*(Lx - minx) #generate uniform float for the x value
            y = miny + np.random.rand()*(Ly - miny) #same for y value
            #print(x)
            prob = func(x, y, **funcparams) #probability at this (x,y) value
            #print(prob)
            rsample = np.random.uniform(0.0, 1.0)*z_max #compare random number to prob
            #if rsample is greater than max probabiliyt at f(x), reject (x,y) and start again
            k += 1
            if k > 10000:
                print("While loop reached %d iterations for one sample" % k)
                break
            
        #print("Particle %d created" %i)    
        #pdf(x,y) is now below prob
        samples[i] = (x,y)
        
    return samples

#A vectorised version of the 2D rejection algorithm
def rejection_2d_vectorised(N, minx, Lx, miny, Ly, func, func_max = None,**funcparams):
    """For N samples of the distribution function func and x range Lx and y range Ly, return N length array with
    sampled x, y positions. 
    
    Input parameters:
    N - int, desired number of generated samples
    minx, Lx - float, minimum and maximum values on the x axis respectively
    miny, Ly - float, ditto for the y axis
    func_max - float, predetermined function maximum, if not provided it will be approximated
    using a large function domain, default None
    func - python function, the function from which to generate the samples
    funcparams - tuple, keyword arguments used in the function
    
    
    Vectorised the process using numpy arrays"""
    
    samples = np.zeros((N,2)) #returned sample array with x,y positions
    
    #find the maximum of the func
    if func_max == None: #unknown function maximum
        max_xpoints = np.linspace(minx, Lx, 1000)
        max_ypoints = np.linspace(miny, Ly, 1000)
        X,Y = np.meshgrid(max_xpoints, max_ypoints)
        z_max = np.max(func(X,Y, **funcparams)) #this is needed for the rsample values
        print("Max is: ", z_max)
    else: #predetermined function maximum
        z_max = func_max + 1e-2 #We know the maximum density is achieved by the target density
    
    ncomplete = 0
    k = 0 #iteration counter
    max_iter = 1000
    while (ncomplete < N) and (k < max_iter): #run until N samples have been generated
        #generate random x, y positions from minx to Lx in x and miny to Ly for y
        xarray = np.random.uniform(low = minx, high = Lx, size = (N - ncomplete))
        yarray = np.random.uniform(low = miny, high = Ly, size = (N - ncomplete))
        
        #determine the probabilities at the random x.y positions                           
        probs = func(xarray, yarray, **funcparams) #probabilities at x, y positions
        rsamples = np.random.uniform(0.0, 1.0, size = np.shape(probs))*z_max #probabilities for rejection
        
        comparison_array = rsamples < probs #accept samples with 0 to z_max under prob value
        nold = ncomplete #running number of generated samples
        nnew = np.count_nonzero(comparison_array) #new generated samples, True is a count of 1
        ncomplete = np.count_nonzero(comparison_array) + nold #total number of accepted samples
        
        #index out the accepted x,y positions and save the values
        accepted_x = xarray[np.nonzero(comparison_array)]
        #print(accepted_x)
        accepted_y = yarray[np.nonzero(comparison_array)]
        #print(accepted_y)
        samples[nold:ncomplete,0] = accepted_x
        samples[nold:ncomplete,1] = accepted_y
        k += 1 #iteration counter
        
        if k == max_iter: #reached maximun number of iterations
            print("Reached the maximum number of iterations, with %d generated samples" % (ncomplete))
        
        #print("New number of generated samples is %d" % nnew)
    
    print("Total number of iterations was %d " % k)
    print("Shape of final generated sample array is ", np.shape(samples))
    return np.array(samples)