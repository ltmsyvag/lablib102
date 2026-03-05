#%%
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import pandas as pd
from scipy import ndimage
from .gaussian2d import gaussian_2d_iso, initial_guess_gaussian2d
from scipy.optimize import curve_fit
import lablib102
class ArrayFrame:
    def __init__(self, path):
        self.path = path
        self.imgarr = np.array(Image.open(path).convert('L'))
        """
            三个点的位置:
               1**********2
              ************
             ************
            3***********
        """
        self.total_mask = None
        self.rects123 = None, None, None, None, None, None
        self.nsites_x, self.nsites_y = None, None
        self.rect_side = None
        self._low_edges = None
        self.df = None
        ## single stats
        self.rects_pixel_mean = None
        self.bg_pixel_mean = None
        self.total_pixel_mean = None
        self.std_of_rect_means = None
        self.centroid_of_sites = None # id_X, id_Y
        # misc
        self.percentile_thrown = 50 # 找质心时丢掉的背景, 以及拟合 2d 高斯时的背景 initial guess
        self._arr_rect_mean_normed = None
        self.popt = None # A, x0, y0, sxsq_plus_sysq, bg
    def define_rects(self, x1, y1, x2, y2, x3, y3, 
                     nsites_x, nsites_y, rect_side, 
                     figsize = (6.4, 4.8), vmax = None, save_path = None,
                     fit_gaussian = False):
        vecx = np.array([x2 - x1, y2 - y1])/(nsites_x-1)
        vecy = np.array([x3 - x1, y3 - y1])/(nsites_y-1)
        grid_points_float = np.array([np.array([x1, y1]) + nx*vecx + ny*vecy 
                        for ny in range(nsites_y)
                        for nx in range(nsites_x)
                        ])
        grid_points_int = np.round(grid_points_float).astype(int)
        rect_side = round(rect_side)
        rect_sums = []
        low_edges = []
        total_mask = np.zeros(self.imgarr.shape, dtype=bool)
        for x, y in grid_points_int:
            x_low_edge = x - (rect_side - 1) // 2 # 保证 rect_side == 1 时, grid point 对应 rect 左上角
            y_low_edge = y - (rect_side - 1) // 2
            low_edges.append((x_low_edge, y_low_edge))
            this_block_slice = (slice(y_low_edge, y_low_edge+rect_side), 
                           slice(x_low_edge, x_low_edge+rect_side))
            
            total_mask[this_block_slice] = True
            this_rect_sum = self.imgarr[this_block_slice].sum()
            rect_sums.append(this_rect_sum)


        self.rects123 = np.round([x1, y1, x2, y2, x3, y3]).astype(int)
        self.nsites_x = nsites_x
        self.nsites_y = nsites_y
        self.rect_side = rect_side
        self._low_edges = low_edges
        arr_sums = np.array(rect_sums).reshape(nsites_y, nsites_x)
        self.total_mask = total_mask
        ## useful stats
        self.rects_pixel_mean = self.imgarr[total_mask].mean()
        self.bg_pixel_mean = self.imgarr[~total_mask].mean()
        self.total_pixel_mean = self.imgarr.mean()
        pixel_means_by_rects = (arr_sums/(rect_side**2)).flatten()
        self.std_of_rect_means = pixel_means_by_rects.std()
        ## dataframe
        lst_id2d = [(id1d//self.nsites_x, id1d%self.nsites_x) for id1d in range(self.nsites_x*self.nsites_y)]
        self.df = pd.DataFrame(lst_id2d, columns=['id_y', 'id_x'])
        self.df['rect_sum'] = arr_sums.flatten()
        self.df['rect_mean'] = pixel_means_by_rects
        self.df['rect_mean_normed'] = self.df['rect_mean']/self.rects_pixel_mean
        self.df[['frame_coord_x', 'frame_coord_y']] = grid_points_int
        
        ## centroid
        x0, y0 = self.get_site_centroid(arr_sums)
        self._update_radial_distance(x0, y0, 'r_from_centroid')

        ## 2d gaussian fit
        self.popt = None
        self._arr_rect_mean_normed = self.df['rect_mean_normed'].values.reshape(self.nsites_y, self.nsites_x)
        if fit_gaussian:
            yy, xx = np.indices(self._arr_rect_mean_normed.shape)
            self.popt, _ = curve_fit(gaussian_2d_iso,
                                (xx, yy), 
                                self._arr_rect_mean_normed.ravel(),
                                p0 = initial_guess_gaussian2d(
                                    self._arr_rect_mean_normed, 
                                    percentile_thrown=self.percentile_thrown)
                                )
            _, x0, y0, _, _ = self.popt
            self._update_radial_distance(x0, y0, 'r_from_gaussian_peak')

        self.visualize_rects(figsize=figsize, vmax=vmax, save_path=save_path)
    def visualize_gaussian_fit(self):
        if self.popt is not None:
            data_shape = self._arr_rect_mean_normed.shape
            yy, xx = np.indices(data_shape)
            zz_fit = gaussian_2d_iso((xx, yy), *self.popt)
            zz_fit = zz_fit.reshape(data_shape)
            fig = plt.figure(figsize = (12,6))
            ax1 = fig.add_subplot(1,2,1)
            ax2 = fig.add_subplot(1,2,2, projection='3d')
            im = ax1.contourf(zz_fit)
            ax1.set_box_aspect(self.nsites_y/self.nsites_x)
            fig.colorbar(im, ax = ax1)
            fig.suptitle(f'A, x0, y0, sxsq_plus_sysq, bg = {self.popt}')
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            ax1.invert_yaxis()
            ax1.set_title('gaussian fit')
            yy, xx = np.indices(zz_fit.shape)
            ax2.plot_wireframe(xx, yy, zz_fit, color = 'k',
                               rstride = 2, cstride = 2,
                                label = 'gaussian fit', alpha = 1)
            ax2.plot_surface(xx, yy, self._arr_rect_mean_normed, cmap = 'jet',
                               alpha=0.3, label = 'data')
            ax2.set_xlabel('x')
            ax2.set_ylabel('y')
            ax2.invert_yaxis()
            ax2.legend()
        else:
            raise ValueError('no fit stored!')
    def _update_radial_distance(self, x0, y0, col):
        # self.centroid_of_sites = x0, y0
        self.df[col] = np.sqrt(
            (self.df['id_x'] - x0)**2
            + (self.df['id_y'] - y0)**2)
    def set_manual_origin(self, x0, y0):
        self._has_rects()
        self._update_radial_distance(x0, y0, 'r_from_manual_origin')
    def get_site_centroid(self, zz: np.ndarray):
        bbgg = np.percentile(zz, self.percentile_thrown) # bg substraction, kinda important
        zz_shifted = zz-bbgg
        zz_shifted[zz_shifted < 0] = 0
        y0, x0 = ndimage.center_of_mass(zz_shifted)
        self.centroid_of_sites = x0, y0
        return x0, y0
    def _has_rects(self):
        if self.rects123 is None:
            raise ValueError("Please call define_rects first to define rects.")
    def visualize_single_rect(self, ix, iy, vmax = None):
        self._has_rects()
        id1d = iy*self.nsites_x + ix
        x_low_edge, y_low_edge = self._low_edges[id1d]
        block_slice = (slice(y_low_edge, y_low_edge+self.rect_side),
                       slice(x_low_edge, x_low_edge+self.rect_side))
        site_arr = self.imgarr[block_slice]
        fig, ax = plt.subplots()
        im = ax.imshow(site_arr, vmax=vmax)
        ax.set_title(f'site {ix, iy}')
        fig.colorbar(im, ax=ax)
    def show_bmp(self, figsize=(6.4, 4.8), vmax=None, save_path=None):
        fig, ax = plt.subplots(figsize=figsize)
        extent = 0, self.imgarr.shape[1],  self.imgarr.shape[0], 0
        ax.imshow(self.imgarr, extent=extent, vmax=vmax)
        if save_path is not None:
            fig.savefig(save_path, dpi=600)
    def visualize_rects(self, figsize = (6.4, 4.8), vmax = None, save_path = None):
        self._has_rects()
        fig, ax = plt.subplots(figsize = figsize)
        extent = 0, self.imgarr.shape[1],  self.imgarr.shape[0], 0
        ax.imshow(self.imgarr, extent=extent, vmax = vmax)
        for x_low_edge, y_low_edge in self._low_edges:
            ax.add_patch(Rectangle((x_low_edge, y_low_edge),
                                self.rect_side, self.rect_side, 
                                fill=False, edgecolor='red', linewidth=0.5))
        x1, y1, x2, y2, x3, y3 = self.rects123
        ax.set_title(f'x1 {x1}, y1 {y1}, x2 {x2}, y2 {y2}, x3 {x3}, y3 {y3}, nx/ny {self.nsites_x}/{self.nsites_y}, rect_side {self.rect_side}\n{lablib102.__version__}',)
        if save_path is not None:
            fig.savefig(save_path, dpi=600)
    def visualize_site_homogeneity(self):
        self._has_rects()
        fig, ax = plt.subplots()
        im = ax.imshow(self._arr_rect_mean_normed)
        ax.add_patch(Circle(
            self.centroid_of_sites, radius=2, color='black', fill = False,
            label = 'centroid of sites'))
        if self.popt is not None:
            A, x0, y0, sxsq_plus_sysq, bg = self.popt
            ax.add_patch(
                Circle(
                (x0, y0), radius = 2, 
                color = 'red', fill = False, ls = ':',
                  label = 'gaussian peak'))
            ax.set_title(f'x0, y0, D4$\sigma$\n{x0:.2f}, {y0:.2f}, {np.sqrt(2*sxsq_plus_sysq):.1f}', loc = 'right')
        fig.colorbar(im, ax=ax)
        ax.legend(loc = (0, 1))
    def rects_hist(self):
        self._has_rects()
        fig, ax = plt.subplots()
        hist_heights, _, _ = ax.hist(self.df['rect_mean_normed'], bins=30, label = 'single ROI pixel mean')
        ax.axvline(self.total_pixel_mean/self.rects_pixel_mean, color='red', linestyle='dashed', label='total Pixel Mean')
        ax.axvline(self.rects_pixel_mean/self.rects_pixel_mean, color='k', linestyle='dashed', label='Pixel Mean of all ROIs')
        ax.axvline(self.bg_pixel_mean/self.rects_pixel_mean, color='blue', linestyle='dashed', label='Pixel Mean of bg (ROI subtracted)')
        ax.errorbar(1, hist_heights.max()/2, xerr=self.std_of_rect_means/self.rects_pixel_mean,
                    fmt='o', color='black', capsize=20, label = f'hist std = {self.std_of_rect_means/self.rects_pixel_mean:.2f}')
        ax.set_xlabel('relative mean intensity per pixel')
        ax.set_ylabel('frequency')
        ax.legend(loc = (1,0))