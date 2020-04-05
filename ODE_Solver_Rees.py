#------------------------------------------------------------------------------------------------------------------------------
# Section 3.0 : Define an ODE solver that will use the previous functions to solve the ODE
#------------------------------------------------------------------------------------------------------------------------------
def ODE_SOLVER_REES():
    import sys
    import math
    import cmath
    import numpy as np
    import matplotlib.pyplot as plt
    import INT_REES
#------------------------------------------------------------------------------------------------------------------------------
# Section 3.1 : Set up the grid in y, allocate memory and define the constants
#------------------------------------------------------------------------------------------------------------------------------

    Levels    = 100                 # Number of levels
    Tolerance = 1e-7                # Tolerance for errors (see Section 3.4)
    beta      = 2.2*(10**-11)       
    g         = 0.0334              # Reduced gravity
    H0        = 60                  # Average thermocline depth (see Tanaka+Hibiya)
    A0        = -1.0                # Constant for n=0
    A2        = -1.0                # Constant for n=2
    c         = np.sqrt(g*H0) 
    C         = np.sqrt((2*beta/c)) 
    
    y  = np.zeros(Levels + 2)       # Allocate memory to y
    dy = 1.0 / float(Levels)        # float/float = float
    
    for n in range(Levels + 2):
        y[n] = n*dy                 # y varies from 0 to 1 in Levels + 2 steps 
       
#-------------------------------------------------------------------------------------------------------------------------------
# Section 3.2 : Set up the values for the wave speed and wave number that we will loop over 
#-------------------------------------------------------------------------------------------------------------------------------
    cs      = np.zeros( (1,1) , dtype = np.complex) # Allocate memory for wave speed
    c_0     = 0.0                                   # Starting point for the wave speed (Take an educated guess)
    d_c_inc = 0.001                                 # Increment that we will add to the previous wave speed after each loop
    d_c     = d_c_inc + 1j * d_c_inc                # Since c is complex we need to make the increment complex too
    k       = 1.0                                   # Prescribed choice of wavenumber
    d_k     = 0.1                                   # Similarly, we increment over the wavenumber after each loop
    
#------------------------------------------------------------------------------------------------------------------------------
# Section 3.3 : Find P and Q for c_0 (first choice of wave speed) and then integrate to find phi with boundary conditions
#------------------------------------------------------------------------------------------------------------------------------    
#------------------------------------------------------------------------------------------------------------------------------
# Section 3.3.1: Begin the iteration towards a solution for wavenumber in the outer loop 
#------------------------------------------------------------------------------------------------------------------------------
# The inner loop (see Section 3.3) involves all the hard work of finding the wave speed for a given wavenumber. The outer loop
# that we start here iterates over a range of wavenumbers so that we can then find (and plot) the growth rate k*c_{i}.
    
    Max_iter_1 = 1 # We will need more than one, but start off with this as a test
    for iter1 in range(Max_iter_1):
#------------------------------------------------------------------------------------------------------------------------------
# Section 3.3.2: Solving for the first choice of wave speed
#------------------------------------------------------------------------------------------------------------------------------
# This section contains the basic structure for how we solve the problem. Begin by calculating P and Q for this choice of wave
# speed. Set the boudary condition and the integrate through the domain to find each phi. Then we calculate the error at the 
# upper boundary.
        
        P, Q = P_Q_Rees(y, c_0 , k)                    # Calculate P and Q for the initial choice of wavespeed

        PHI = np.zeros(Levels + 2, dtype = np.complex) # Allocate memory for phi
        PHI[0] = 0.0                                   # Boundary condition
        PHI[1] = 1.0                                   # Sets the otherwise arbitrary amplitude

        #Integrate up through the domain
        for n in range(1,Levels + 1): # Start at 1 otherwise Press[n-1] starts at Press[-1] # Also up to N_levs + 1 because y[n+1]=y[N_levs+2] which is the top of the boundary
            PHI[n+1] = INT_REES(PHI[n-1],PHI[n],P[n],Q[n],y[n-1],y[n],y[n+1])

        # Calculate the error at the upper boundary
        PHI_NU_BC = PHI[Levels]
        E_0 = PHI[Levels + 1] - PHI_NU_BC   
#------------------------------------------------------------------------------------------------------------------------------
# Section 3.3 : Iterate towards a solution for the ODE over all the wave speed increments
#------------------------------------------------------------------------------------------------------------------------------  
# Now we have the same standard procedure as in the previous section but we are looping over this for different (increasingly
# accurate) wave speeds.

        #Start iterating towards a solution (i.e. increment c forwards)
        c_1 = c_0 + d_c     # The next value for the wave speed is the initial prescribed value plus the increment
        Max_iter_2 = 1      # Number of iterations,  try one as a test but will need more for an accurate c
        for iter2 in range(Max_iter_2):

            P, Q = P_Q_Rees(y,c_1 , k)                     # Calculate P and Q for the new wave speed

            PHI = np.zeros(Levels + 2, dtype = np.complex) # Allocate memory for phi
            PHI[0] = 0.0                                   # Boundary condition
            PHI[1] = 1.0                                   # Sets the otherwise arbitrary amplitude

            #Integrate up through the domain
            for n in range(1,Levels + 1):
                PHI[n+1] = INT_REES(PHI[n-1],PHI[n],P[n],Q[n],y[n-1],y[n],y[n+1]) # For c_1, find all the phi in the domain

            # Calculate the error at the upper boundary
            PHI_NU_BC = PHI[Levels]  
            E_1 = PHI[Levels + 1] - PHI_NU_BC
       
#------------------------------------------------------------------------------------------------------------------------------
# Section 3.4 : Check for convergence and update the wave speed for the next loop
#------------------------------------------------------------------------------------------------------------------------------
# Here we check the convergence of the solution and update c via a shooting method and then store the final value of c in cs.
# If the solution does not converge then we break the loop and stop iterating. 


            if abs(E_1) < Tolerance:    
                if iter_2 == Levels: # i_kappa % Interval_kappa == 0
                    cs[0,0] = c_1                 # Store the final value for c in cs
                    c_0 = c_1        
                    break

                # Calculate the updated values of c using a shooting method
                gradient = (c_1 - c_0)/(E_1 - E_0)  
                c_new = c_1 - gradient*E_1    

                if abs(E_1) < abs(E_0): # If absolute value of the error for c_1 is less than the old error for c_0 then,
                    c_0 = c_1           # Save c_1 if it is better than c_0
                    E_0 = E_1           # Take the least error so that we can be sure that the solution is converging

                c_1 = c_new   # Update c_1 in preperation for the next loop over wave speed

                # If the solution fails to converge for a given k then stop looping
                if abs(E_1) > Tolerance :
                    break                
#------------------------------------------------------------------------------------------------------------------------------
# Section 3.5 : Update the wavenumber in preperation for the next loop
#------------------------------------------------------------------------------------------------------------------------------
        k = k + d_k 
            
ODE_SOLVER_REES()
