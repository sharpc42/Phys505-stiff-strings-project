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
    eps = k/1000
    t_end = 10.0 + eps
    frames = t_end / 24
    n = 0
    t = 0.0
    v = 0.9

    # pluck string for initial conditions
    def f(x):
        return 0.1 * np.sin(np.pi*x/N)
    initial_fxn = np.vectorize(f)

    # create main and update arrays for
    # both position and velocity
    Y = initial_fxn(np.linspace(0,N,N))
    Yp = np.zeros(N,float)
    dY = np.zeros(N,float)
    dYp = np.zeros(N,float)
    print("Y",Y)
    print("dY",dY)

    # define A, B, and C constant coefficients
    # for use in solving system of equations
    beta = k / (2 * h**4)
    A = 0.5*k*beta  # up to multiplicative constant
    B = 0.5*k*beta * h*h - 4
    C = 0.5*k*(6 - 2*h*h)
    print("A",A)
    print("B",B)
    print("C",C)

    # define M-slash and inverse M matrices
    M_right = np.zeros((N-2, N-2))
    for i in range(0,N-2):
        for j in range(0,N-2):
            if i==j: 
                M_right[i,j] = 1+C
                if i < N-3 and j < N-3:
                    M_right[i,j+1] = B
                if i < N-4 and j < N-4:
                    M_right[i,j+2] = A
                if i>1 and j>1:
                    M_right[i,j-1] = B
                if i>2 and j>2:
                    M_right[i,j-2] = A
    M_left = np.zeros((N-2, N-2))
    for i in range(0,N-2):
        for j in range(0,N-2):
            if i==j: 
                M_left[i,j] = -C
                if i < N-3 and j < N-3:
                    M_left[i,j+1] = -B
                if i < N-4 and j < N-4:
                    M_left[i,j+2] = -A
                if i>1 and j>1:
                    M_left[i,j-1] = -B
                if i>2 and j>2:
                    M_left[i,j-2] = -A
    V_left = M_left - np.identity(N-2)
    V_right = M_right - np.identity(N-2)

    plot_output(Y,-1)

    # step through time
    outputs = 0
    while t < t_end:
        # compute RHS of position update
        M_right_vec = M_right.dot(Y[1:-1]) + k*dY[1:-1]
        Yp[1:-1] = np.linalg.solve(M_left, M_right_vec)

        # explicitly enforce boundary conditions
        Yp[0] = 0
        Yp[-1] = 0

        # computer RHS of velocity update
        V_right_vec = V_right.dot(Y[1:-1]) + dY[1:-1]
        dYp[1:-1] = np.linalg.solve(V_left,V_right_vec)
        if outputs < 5:
            print("Mrv",M_right_vec[3])
            print("Y",Yp[3])
            print("dY",dYp[3])
            outputs += 1

        # perform update on position and velocity
        Y, Yp = Yp, Y
        dY, dYp = dYp, dY
        
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
    filename = 'output'+str(n+1)+'.png'
    FILENAMES.append(filename)

    X = [x for x in range(len(Y))]
    plt.plot(X,Y)
    plt.title('Plot of Displacement at time '+str(n+1))
    plt.xlabel('x')
    plt.ylabel('Displacement')
    plt.ylim(-0.15,0.15)
    plt.savefig(filename)
    plt.close(fig)

    print('   Created image ' + str(n+1))

# animate output and export MP4 video
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