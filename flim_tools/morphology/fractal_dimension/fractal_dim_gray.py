import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams["figure.dpi"] = 300
from PIL import Image
import numpy as np
import math

def pad_to_square(im, debug=False):
    """
    Pads image to a square for fractal dimension method

    Parameters
    ----------
    im : ndarray
        image in ndarray.
    debug : bool, optional
        Show debugging information. The default is False.

    Returns
    -------
    im_square : ndarray
        image or roi padded to a square.

    """
    # pad with zeros to dimension
    n_rows, n_cols = im.shape
    pad = "rows" if n_rows == min(n_rows, n_cols) else "cols"
    
    #pad offsets
    # pad axis
    difference = np.abs(n_rows- n_cols)
    before = difference // 2 
    after = int(np.ceil(difference /2))
    if pad == "rows":
        pass
        im_square = np.pad(im,((before,after),(0,0)), 
                            mode="constant", constant_values=0)
    if pad == "cols":
        pass
        im_square = np.pad(im,((0,0),(before,after)),
                            mode="constant", constant_values=0)

    return im_square


def fractal_dimension_gray(roi, intensity,mode="standard", debug=False):
    """
    Computes the fractal dimenison based on the differential box counting method
    

    Parameters
    ----------
    roi : ndarray
        labels image.
    intensity : ndarray
        intensity image .
    mode : str, optional
        Choose the differential box counting method
        "Standard" will keep intensities in place
        "Shifting" will begin box counting method with bottom of 
        first box at the min grayscale value. The default is "standard".
    debug : bool, optional
        Show debugging in formation. The default is False.

    Returns
    -------
    D : float
        fractal dimension.

    """
    image = roi * intensity
    # pad to square if needed
    rows, cols = image.shape
    if rows != cols:
        image = pad_to_square(image, debug=debug)
    
    
    (imM, _) = image.shape
    
    # calculate Nr and r
    Nr = []
    r = []
    if debug:
        print("|\tNr\t|\tr\t|S\t|")
    a = 2
    b = imM//2
    nval = 20
    lnsp = np.linspace(1,math.log(b,a),nval)
    sval  = a**lnsp
	
    for S in sval:#range(2,imM//2,(imM//2-2)//100):
        Ns = differential_box_counting(image, int(S), mode=mode, debug=debug)
        Nr.append(Ns)
        R = S/imM
        # r.append(S) # I think this should be R
        r.append(R)
        if debug:
            print("|%10d\t|%10f\t|%4d\t|"% (Ns,R,S))
	
	
    # calculate log(Nr) and log(1/r)    
    y = np.log(np.array(Nr))
    x = np.log(1/np.array(r))
    (D, b) = np.polyfit(x, y, deg=1)
    
    # search fit error value
    N = len(x)
    Sum = 0
    for i in range(N):
        Sum += (D*x[i] + b - y[i])**2
        
    errorfit = (1/N)*math.sqrt(Sum/(1+D**2))
    
    if debug:
        # figure size 10x5 inches
        plt.figure(1,figsize=(10,5)).canvas.set_window_title('Fractal Dimension Calculate')
        plt.subplots_adjust(left=0.04,right=0.98)
        plt.subplot(121)
        # plt.title(path)
        plt.imshow(image, cmap="gray")
        plt.axis('off')
    
        plt.subplot(122)  
        plt.title(f'Fractal dimension = {D:.3f} \n Fit Error = {errorfit:.3f}')
        
        plt.plot(x, y, 'ro',label='Calculated points')
        plt.plot(x, D*x+b, 'k--', label='Linear fit' )
        plt.legend(loc=4)
        plt.xlabel('log(1/r)')
        plt.ylabel('log(Nr)')
        plt.show()
        
    return D


def differential_box_counting(im, s, mode="standard", debug=False):
    """
    differential box counting method for grayscale images

    Parameters
    ----------
    im : ndarry
        2d image
    s : int
        scaling factor
    mode : string, optional
        determines how the boxes will be calculated, 
        standard differentil box counting method vs 
        shifting box counting method.
        The default is "standard".
    debug : bool, optional
        Show intermediate images. The default is False.

    Returns
    -------
    Ns : int
        number of boxes counted at this scale.
    """
    (width, height) = im.shape
    assert(width == height)
    M = width
    # grid size must be bigger than 2 and least than M/2    
    G = 256 # range for dtype=np.uint8 # better way?
    
    # these lines are for skimage dtype determination
    # they generate a sample image 3x3 to know what
    # the return type is
    if im.shape == (3,3):
        return 1
    
    # error check 
    assert(s >= 2) # scaling factor is min 2
    assert(s <= M//2) # length of image is min 2
        
    ngrid = math.ceil(M / s)
    h = G*(s / M) # box height
    grid = np.zeros((ngrid,ngrid), dtype='int32')
    
    #iterate through larger grid
    for i in range(ngrid):
        for j in range(ngrid):
            maxg = 0
            ming = 255
            #iterate through each pixel in sub-grid, the min error bounds 
            for k in range(i*s, min((i+1)*s, M)):
                for l in range(j*s, min((j+1)*s, M)):
                    if im[k, l] > maxg:
                        maxg = im[k, l]
                    if im[k, l] < ming:
                        ming = im[k, l]
            
            # box counting methods
            if mode == "standard":
                grid[i,j] = math.ceil(maxg/h) - math.ceil(ming/h) + 1
            if mode == "shifting":    
                grid[i,j] = math.ceil((maxg-ming+1)/h)
    
    if debug:
        plt.title(f"mode: {mode} \n scale: {s}")
        plt.imshow(grid)
        plt.show()
    
    Ns = grid.sum()
    
    return Ns


if __name__ == '__main__':
    path = str(input("Enter path to image:"))
    # make image if nothing passed
    if path == "":
        from skimage.morphology import disk
        roi = disk(2)
        roi = roi[:4,:4]
        rng = np.random.default_rng(seed=0)
        random_grid = rng.random(roi.shape)
        intensity = (roi * random_grid * 255).astype(int)
        
    else:
        from pathlib import Path
        # path = Path(r"./sierpienski_triangle.jpg")
        # path = Path(r"./waves.jpg")
        # path = Path(r"./clouds.jpg")
        path = Path(r"./ocean.jpg")
        # path = Path(path)
        intensity = Image.open(path) # Brodatz/D1.gif
        intensity = intensity.convert('L')
        # invert, make into abox, and set as binary with uin8 range
        
        intensity = np.asarray(intensity)
        # image = np.invert(np.asarray(image, dtype=(np.uint8)))
        im_min = min(intensity.shape)
        intensity = intensity[:im_min, :im_min]
        # image = ((image > 100).astype(int) * 255).astype(np.uint8)
        roi = np.ones(intensity.shape)
        
    # image = image[:, :len(image)//2]
    fractal_dim = fractal_dimension_gray(roi, intensity, debug=True)
    print(f"fractal dimension: {fractal_dim}")
   

