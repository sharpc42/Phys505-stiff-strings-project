# tentative stiff springs project

import os

import imageio as imio
import matplotlib.pyplot as plt
import moviepy.editor as mp
import numpy as np

FILENAMES = []
N = 100

# function to handle user input either through
# file or through console line input
def handle_input():
    return 0

# likely Crank-Nicolson numerical icontegration
# for purposes of stability
def integrate():
    L = 1
    h = L/N
    k = 1e-4
    eps = h/1000
    t_end = 10.0 + eps
    frames = t_end / 24
    n = 0
    t = 0.0
    v = 0.9

    # create main and update arrays
    Y = np.zeros(N+3,float)
    Yp = np.zeros(N+3,float)
    dY = np.zeros(N+3,float)
    dYp = np.zeros(N+3,float)

    # pluck string for initial conditions
    # Y[N-1] = 0.1
    # dY[N//2] = -0.1

    # define A, B, and C constant coefficients
    # for use in solving system of equations
    beta = k / (2 * h**4)
    A = beta  # up to multiplicative constant
    B = beta * h*h - 4
    C = 6 - 2*h*h
    print("A",A)
    print("B",B)
    print("C",C)

    # define M-slash and inverse M matrices
    M_slash = np.zeros((N+3, N+3))
    M_slash[0,0] = 1
    M_slash[1,1] = -1/k
    for i in range(2,N+3):
        for j in range(2,N+3):
            if i==j: 
                M_slash[i,j] = C
                if i < N+2 and j < N+2:
                    M_slash[i,j+1] = B
                if i < N+1 and j < N+1:
                    M_slash[i,j+2] = A
                if i>2 and j>2:
                    M_slash[i,j-1] = B
                if i>3 and j>3:
                    M_slash[i,j-2] = A
    M = np.zeros((N+3, N+3))
    M[0,0] = 1
    M[1,1] = -1/k
    for i in range(2,N+3):
        for j in range(2,N+3):
            if i==j: 
                M[i,j] = -C
                if i < N+2 and j < N+2:
                    M[i,j+1] = -B
                if i < N+1 and j < N+1:
                    M[i,j+2] = -A
                if i>2 and j>2:
                    M[i,j-1] = -B
                if i>3 and j>3:
                    M[i,j-2] = -A
    M_inv = np.linalg.inv(M)

    # step through time
    while t < t_end:
        Yp = M_inv.dot(M_slash.dot(Y))
        # enforce boundary conditions
        Yp[0] = 0
        Yp[-1] = 0
        # (how to enforce second derivatives?)
        Y, Yp = Yp, Y
        # don't update boundaries
        # for i in range(1,N):
        #     Yp[i] = Y[i] + h * (v**1 / 1**2) * (Y[i+1] + Y[i-1] - 2*Y[i])
        #     #dYp[i] = dY[i] + h * (v**1 / 1**2) * (Y[i+1] + Y[i-1] - 2*Y[i])
        # Y, Yp = Yp, Y
        # #dY, dYp = dYp, dY
        
        # create plot
        if t_end > n * frames:
            plot_output(Y,n)
            n += 1
        
        t += h

    # final displacement array
    return Y

# plot displacement output for each time step
# as computed by integrator
def plot_output(Y,n):
    fig = plt.figure()
    #img = plt.imshow(Y)
    filename = 'output'+str(n)+'.png'
    FILENAMES.append(filename)

    X = [x for x in range(len(Y))]
    plt.plot(X,Y)
    plt.title('Plot of Displacement at time '+str(n))
    plt.xlabel('x')
    plt.ylabel('Displacement')
    plt.savefig(filename)
    plt.close(fig)

    print('   Created image ' + str(n))

# animate output (prob use AthenaPyPy)
def animate():
    gif_file = 'render.gif'
    anm_file = 'render.mp4'

    images = []
    for file in FILENAMES:
        images.append(imio.imread(file))
    imio.mimsave(gif_file, images)
    gif = mp.VideoFromClip(gif_file, audio=False)
    gif.write_videofile(anm_file)
    os.system('rm ' + gif_file)

# displacement output generates wave form
# which can be used to generate sound file
def generate_sound_file():
    return 0

print("Integrating...")
Y = integrate()
print("Done.\n")

print("Plotting...")
#plot_output(Y)
print("Done.\n")