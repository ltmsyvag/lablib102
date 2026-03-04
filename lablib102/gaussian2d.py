#%%
import numpy as np
from scipy import ndimage

def gaussian_2d_iso(xxyy_coords, A, x0, y0, sxsq_plus_sysq, bbgg):
    xx, yy = xxyy_coords
    g = A * np.exp(-(
        (xx - x0)**2 +
        (yy - y0)**2) / (sxsq_plus_sysq)) + bbgg
    return g.ravel()   # must return 1D array


def initial_guess_gaussian2d(zz, percentile_thrown = 5):
    # estimate background
    bbgg = np.percentile(zz, percentile_thrown)
    A = zz.max() - bbgg

    # subtract background (important!)
    zz_shifted = zz-bbgg
    zz_shifted[zz_shifted <0] = 0

    # centroid and 2nd moments
    y0, x0 = ndimage.center_of_mass(zz_shifted)
    yy, xx = np.indices(zz_shifted.shape)
    sigma_x_sq = ((xx - x0)**2*zz_shifted).sum()/zz_shifted.sum()
    sigma_y_sq = ((yy - y0)**2*zz_shifted).sum()/zz_shifted.sum()
    sxsq_plus_sysq = sigma_x_sq + sigma_y_sq

    return A, x0, y0, sxsq_plus_sysq, bbgg
if __name__ == '__main__':
    # ---- Generate synthetic data ----
    nx, ny = 50, 30
    x = np.arange(nx)
    y = np.arange(ny)
    xx, yy = np.meshgrid(x, y)

    true_params = (5, 20, 25, 50, 0.5)  # A, x0, y0, sxy, bg
    zz = gaussian_2d_iso((xx, yy), *true_params).reshape(ny, nx)

    # add noise
    zz_noisy = zz + 0.3 * np.random.normal(size=zz.shape)
    
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit

    fig, ax = plt.subplots()
    ax.imshow(zz_noisy)

    yy, xx = np.indices(zz_noisy.shape)
    popt, _ = curve_fit(gaussian_2d_iso, (xx, yy), zz_noisy.ravel(), p0=initial_guess_gaussian2d(zz_noisy))

    zz_fit = gaussian_2d_iso((xx,yy), *popt)
    zz_fit = zz_fit.reshape(zz_noisy.shape)
    fig, ax = plt.subplots()
    # ax.imshow(zz_fit)
    ax.contourf(zz_fit)
    ax.set_box_aspect(ny/nx)
    ax.invert_yaxis()