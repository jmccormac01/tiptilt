"""
Script to work out the tip-tilt of GOTO CCDs. Uses
a focus sweep to determine how the best focus position
changes across the FOV

To do:
    - Pickle the focus cube after it is run once
    - Add smoothing of the 2D grid of best focus postions
    - Determine the correct orientation and amount of movement to flatten image
    - Centre grid on images, rather than starting in the corner
"""

import glob as g
import sep
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.stats.sigma_clipping import sigma_clipped_stats
from scipy.optimize import curve_fit

def measureHfd(data):
    bkg = sep.Background(data)
    thresh = 3 * bkg.globalrms
    objects = sep.extract(data-bkg, thresh)
    # taken from g-tecs autoFocus.py
    hfr, mask = sep.flux_radius(data,
                                objects['x'],
                                objects['y'],
                                30*np.ones_like(objects['x']),
                                0.5,
                                normflux=objects['cflux'])
    mask = np.logical_and(mask == 0, objects['peak'] < 40000)
    hfd = 2*hfr[mask]
    if hfd.size > 3:
        mean, median, std = sigma_clipped_stats(hfd, sigma=2.5, iters=10)
        return median, std
    return 0.0, 0.0

def measureFwhm(data):
    bkg = sep.Background(data)
    thresh = 3 * bkg.globalrms
    objects = sep.extract(data-bkg, thresh)
    fwhm = 2. * np.sqrt(np.log(2.) * (objects['a']**2+ objects['b']**2))
    mask = np.where(objects['peak'] < 40000)
    fwhm = fwhm[mask]
    if len(fwhm) > 3:
        mean, median, std = sigma_clipped_stats(fwhm, sigma=2.5, iters=10)
        return median, std
    return 0.0, 0.0

def orderImgs(imgs):
    """
    Order the images by increasing focus position
    """
    foc_pos = []
    for img in imgs:
        foc_pos.append(int(fits.open(img)[0].header['FOCPOS']))
    temp = zip(foc_pos, imgs)
    temp = sorted(temp)
    foc_pos, imgs = zip(*temp)
    return imgs, foc_pos

def generateFocusSweepCube(imgs, foc_pos, box_size):
    """
    Generate a cube of focus values
    """
    focus_stack = []
    for i, img in enumerate(imgs):
        print('{} FocPos: {}'.format(img, foc_pos[i]))
        data = fits.open(img)[0].data[46:6177, 65:8240]
        n_box_x = int(data.shape[1]/box_size)
        n_box_y = int(data.shape[0]/box_size)
        focus = np.empty((n_box_x, n_box_y))
        for x in range(0, n_box_x):
            for y in range(0, n_box_y):
                grid = np.array(data[y*box_size:y*box_size+box_size,
                                     x*box_size:x*box_size+box_size]).astype(np.int32).copy(order='C')
                focus[x, y] = measureFwhm(grid)[0]
        focus_stack.append(focus)
    return np.dstack(focus_stack)

def parabola(t, *p):
    """
    Parabolic function - vertex form
    """
    a, h, k = p
    y = np.zeros(t.shape)
    y = a*((t-h)**2)+k
    return y

def estimateParab(pos, med):
    """
    estimate the starting position from the data:
    p0 = [a,h,k]
    a = +ve for U shaped parabola
    h = x pos of minimum
    k = y pos of minium
    """
    return [1., np.average(pos), min(med)]

def fitParabola(pos, med):
    """
    Fit a parabola to the focus sweep data
    """
    p0 = estimateParab(pos, med)
    try:
        coeff, _ = curve_fit(parabola, pos, med, p0)
        xfit = np.linspace(min(pos)-1000, max(pos)+1000)
        yfit = parabola(xfit, *coeff)
        print(coeff)
        fitted_pos = coeff[1]
        fitted_fwhm = coeff[2]
        return xfit, yfit, fitted_pos, fitted_fwhm
    except RuntimeError:
        return 0.0, 0.0, 999, 999

def fitStack(stack, foc_pos):
    """
    Take a focus sweep cube and fit parabolas to the min
    focus areas. Plot the 2D grid showing the results
    """
    x = stack.shape[0]
    y = stack.shape[1]
    best_foc_pos_grid = np.empty((x, y))
    fig, ax = plt.subplots(x, y, sharex=True, sharey=True, figsize=(15, 15))
    for i in range(0, x):
        for j in range(0, y):
            loc = np.where(stack[i, j] == min(stack[i, j]))[0][0]
            llim = loc - 4
            ulim = loc + 4
            if llim < 0:
                llim = 0
            if ulim > len(stack[i, j]):
                ulim = len(stack[i, j])
            y_to_fit = stack[i, j][llim:ulim]
            x_to_fit = foc_pos[llim:ulim]
            para_x, para_y, best_foc_pos, best_fwhm = fitParabola(x_to_fit, y_to_fit)
            # sanity check on min foc
            if best_fwhm - min(stack[i, j]) > 0.25:
                best_foc_pos = foc_pos[loc]
            ax[i, j].plot(foc_pos, stack[i, j], 'k-')
            ax[i, j].plot(para_x, para_y, 'g-')
            ax[i, j].axvline(best_foc_pos, 0, 7, color='r', linewidth=1)
            ax[i, j].set_ylim(1, 7)
            best_foc_pos_grid[i, j] = best_foc_pos
    fig2, ax2 = plt.subplots(1, figsize=(10, 10))
    final_map = ax2.imshow(best_foc_pos_grid, cmap='Greys')
    plt.colorbar(final_map, ax=ax2)
    plt.show()


if __name__ == "__main__":
    box_size = 1000
    imgs = g.glob('data/*.fits')
    imgs, foc_pos = orderImgs(imgs)
    stack = generateFocusSweepCube(imgs, foc_pos, box_size)
    fitStack(stack, foc_pos)

