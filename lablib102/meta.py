#%%
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd

class ArrayFrame:
    def __init__(self, path):
        self.path = path
        assert path.endswith('.bmp'), "Only .bmp files are supported"
        self.imgarr = np.array(Image.open(path))
        self.total_mask = None
        
        """
            三个点的位置:
               1**********2
              ************
             ************
            3***********
        """
        self.rects123 = None, None, None, None, None, None
        self.n_sites_x, self.n_sites_y = None, None
        self.rect_side = None
        self._low_edges = None
        self.arr_sums = None
        self.df = None
        ## single stats
        self.rects_pixel_mean = None
        self.bg_pixel_mean = None
        self.total_pixel_mean = None
        # self.pixel_means_by_rects = None
        self.std_of_rect_means = None
        self.centroid_id = None # id_X, id_Y

    def define_rects(self, x1, y1, x2, y2, x3, y3, nsites_x, nsites_y, rect_side, figsize = (6.4, 4.8), vmax = None, save_path = None):
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
        self.n_sites_x = nsites_x
        self.n_sites_y = nsites_y
        self.rect_side = rect_side
        self._low_edges = low_edges
        self.arr_sums = np.array(rect_sums).reshape(nsites_y, nsites_x)
        self.total_mask = total_mask
        ## useful stats
        self.rects_pixel_mean = self.imgarr[total_mask].mean()
        self.bg_pixel_mean = self.imgarr[~total_mask].mean()
        self.total_pixel_mean = self.imgarr.mean()
        pixel_means_by_rects = (self.arr_sums/(rect_side**2)).flatten()
        self.std_of_rect_means = pixel_means_by_rects.std()
        ## dataframe
        lst_id2d = [(id1d//self.n_sites_x, id1d%self.n_sites_x) for id1d in range(self.n_sites_x*self.n_sites_y)]
        self.df = pd.DataFrame(lst_id2d, columns=['id_y', 'id_x'])
        self.df['rect_sum'] = self.arr_sums.flatten()
        self.df['rect_mean'] = pixel_means_by_rects
        self.df['rect_mean_normed'] = self.df['rect_mean']/self.rects_pixel_mean
        # self.df['rect_sum_normed'] = self.df['rect_sum']/self.df['rect_sum'].max()
        self.df[['frame_coord_x', 'frame_coord_y']] = grid_points_int
        # derived data
        self.centroid_id = ((self.df['id_x']*self.df['rect_sum']).sum()/self.df['rect_sum'].sum(), 
                            (self.df['id_y']*self.df['rect_sum']).sum()/self.df['rect_sum'].sum())
        
        self.df['r_to_centroid'] = np.sqrt((self.df['id_x'] - self.centroid_id[0])**2 + (self.df['id_y'] - self.centroid_id[1])**2)
        self.visualize_rects(figsize=figsize, vmax=vmax, save_path=save_path)
    def _rects_check(self):
        if self.rects123 is None:
            raise ValueError("Please call define_rects first to define rects.")
    def visialize_single_rect(self, ix, iy, vmax = None):
        id1d = iy*self.n_sites_x + ix
        x_low_edge, y_low_edge = self._low_edges[id1d]
        block_slice = (slice(y_low_edge, y_low_edge+self.rect_side),
                       slice(x_low_edge, x_low_edge+self.rect_side))
        site_arr = self.imgarr[block_slice]
        fig, ax = plt.subplots()
        im = ax.imshow(site_arr, vmax=vmax)
        fig.colorbar(im, ax=ax)
    def visualize_bmp(self, figsize=(6.4, 4.8), vmax=None, save_path=None):
        fig, ax = plt.subplots(figsize=figsize)
        extent = 0, self.imgarr.shape[1],  self.imgarr.shape[0], 0
        ax.imshow(self.imgarr, extent=extent, vmax=vmax)
        if save_path is not None:
            fig.savefig(save_path, dpi=600)
    def visualize_rects(self, figsize = (6.4, 4.8), vmax = None, save_path = None):
        self._rects_check()
        fig, ax = plt.subplots(figsize = figsize)
        extent = 0, self.imgarr.shape[1],  self.imgarr.shape[0], 0
        ax.imshow(self.imgarr, extent=extent, vmax = vmax)
        for x_low_edge, y_low_edge in self._low_edges:
            ax.add_patch(Rectangle((x_low_edge, y_low_edge),
                                self.rect_side, self.rect_side, 
                                fill=False, edgecolor='red', linewidth=0.5))
        if save_path is not None:
            fig.savefig(save_path, dpi=600)
    def visualize_site_homogeneity(self):
        self._rects_check()
        fig, ax = plt.subplots()
        im = ax.imshow(self.arr_sums/self.arr_sums.max())
        fig.colorbar(im, ax=ax)
    def rects_hist(self):
        self._rects_check()
        fig, ax = plt.subplots()
        ax.hist(self.df['rect_mean']/self.rects_pixel_mean, bins=30, label = 'single ROI pixel mean')
        ax.axvline(self.total_pixel_mean/self.rects_pixel_mean, color='red', linestyle='dashed', label='total Pixel Mean')
        ax.axvline(self.rects_pixel_mean/self.rects_pixel_mean, color='k', linestyle='dashed', label='Pixel Mean of all ROIs')
        ax.axvline(self.bg_pixel_mean/self.rects_pixel_mean, color='blue', linestyle='dashed', label='Pixel Mean of bg (ROI subtracted)')
        ax.errorbar(1, 100, xerr=self.std_of_rect_means/self.rects_pixel_mean,
                    fmt='o', color='black', capsize=20, label = f'hist std = {self.std_of_rect_means/self.rects_pixel_mean:.2f}')
        ax.set_xlabel('relative mean intensity per pixel')
        ax.set_ylabel('frequency')
        ax.legend(loc = (1,0))
