import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored
import matplotlib as mpl

'''
def quadraticEqGenerator(nSamples):
    x = np.random.default_rng().uniform(-1.0, 1.0, (nSamples,3))
    return x
'''
def quadraticEqGenerator(nSamples):
    return np.random.default_rng().uniform(-1,1,(nSamples,3))

def quadraticEqSolution(coeff):

    a = coeff[:,0:1]
    b = coeff[:,1:2]
    c = coeff[:,2:3]
    
    delta = b**2 - 4*a*c
    delta = delta.reshape(-1,1)
    
    result = np.where(delta>0, np.sqrt(delta), 0.0)
    result = result*np.array([-1,+1])
    result = (-b+result)/(2*a)
    result = np.where(delta>0, result, (None, None))
    result = np.where(np.abs(a)>1E-10, result, np.array((-c/b, -c/b)).reshape(-1,2))  
    return result
'''

def quadraticEqSolution(coeff):
    a = coeff[:,0]
    b = coeff[:,1]
    c = coeff[:,2]
    nSamples = coeff.shape[0]
    # Discriminant
    D = b**2 - 4*a*c
    
    results = np.zeros((nSamples, 3))
    results[:,2] = D
    
    # Masks
    mask_D_neg = D < 0
    mask_D_zero = D == 0
    mask_D_pos = D > 0
    
    # D < 0: without real solutions
    results[mask_D_neg, 0] = np.nan
    results[mask_D_neg, 1] = np.nan
    
    # D == 0: one solution
    results[mask_D_zero, 0] = -b[mask_D_zero] / (2 * a[mask_D_zero])
    results[mask_D_zero, 1] = results[mask_D_zero, 0] 
    
    # D > 0: two solutions
    sqrt_D = np.sqrt(D[mask_D_pos])
    results[mask_D_pos, 0] = (-b[mask_D_pos] + sqrt_D) / (2 * a[mask_D_pos])
    results[mask_D_pos, 1] = (-b[mask_D_pos] - sqrt_D) / (2 * a[mask_D_pos])
    
    # Return results without D columns
    return results[:, :2]
'''

def plotQuadraticEqSolvability(z, interactive=False):
    solutions = quadraticEqSolution(z)

    has_solution = ~np.isnan(solutions[:,0])
    
    if interactive==False:

        fig = plt.figure(figsize=(9,9))
        axis = fig.add_subplot(projection='3d')
        points = axis.scatter(z[:,0], z[:,1], z[:,2], c=has_solution, cmap='RdYlGn')
        cbar = fig.colorbar(points, aspect=4)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["No Solution", "Has Solution"])
        axis.set_xlabel('a')
        axis.set_ylabel('b')
        axis.set_zlabel('c')
        axis.set_title('Quadratic Equation Solvability')
        plt.show()
        
    else:
        
        import plotly.express as px
        has_solution = ~np.isnan(solutions[:500,0])
        fig = px.scatter_3d(x=z[:500,0], y=z[:500,1], z=z[:500,2], color=has_solution, size=[1]*500, labels={'x':'a', 'y':'b', 'z':'c', 'color':'Has Solution'})  
        fig.show()
    
    
    
    
def plotSqEqSolutions(x, y, y_pred):
    
    fig, axes = plt.subplots(1,2, figsize=(12,4))

    pull = (y - y_pred)/y
    pull = pull.flatten()
    threshold = 1E-2
    print(colored("Fraction of events with Y==Y_pred:","blue"),np.mean(np.isclose(y, y_pred)))
    print(colored("Fraction of examples with abs(pull)<0.01:","blue"),np.mean(np.abs(pull)<threshold))
    print(colored("Pull standard deviation:","blue"),pull.std())
    
    axes[0].hist(pull, bins=np.linspace(-1.5,1.5,40), label="(True-Pred)/True");
    axes[0].legend()

    axes[1].axis(False)
    axis = fig.add_subplot(133, projection='3d')

    pull = (y - y_pred)/y
    colors = np.abs(pull)<threshold
    colors = np.sum(colors, axis=1)

    cmapName = plt.rcParams["image.cmap"]
    cmap = mpl.colormaps[cmapName]
    axis.scatter(x[:,0:1], x[:,1:2], x[:,2:3], c = colors);
    axis.scatter((-2), (-2), (-2), label='none correct', marker='o', color=cmap.colors[1])
    axis.scatter((-2), (-2), (-2), label='single correct', marker='o', color=cmap.colors[128])
    axis.scatter((-2), (-2), (-2), label='double correct', marker='o', color=cmap.colors[-1])
    axis.legend(bbox_to_anchor=(1.5,1), loc='upper left')
    axis.set_xlabel("a")
    axis.set_ylabel("b")
    axis.set_zlabel("c");
    axis.set_xlim([-1.1,1.1])
    axis.set_ylim([-1.1,1.1])
    axis.set_zlim([-1.1,1.1])

    plt.subplots_adjust(bottom=0.15, left=0.05, right=0.95, wspace=0.35, hspace=0.0)