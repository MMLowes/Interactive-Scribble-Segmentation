# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 12:59:04 2022

@author: lowes
"""
import numpy as np
import torch
import os
import cv2

def overlay_masks(im, masks, alpha=0.5):
    colors = np.load(os.path.join(os.path.dirname(__file__), 'pascal_map.npy'))/255.
    
    if isinstance(masks, np.ndarray):
        masks = [masks]

    assert len(colors) >= len(masks), 'Not enough colors'

    ov = im.copy()
    im = im.astype(np.float32)
    total_ma = np.zeros([im.shape[0], im.shape[1]])
    i = 1
    for ma in masks:
        ma = ma > 0.9
        fg = im * alpha+np.ones(im.shape) * (1 - alpha) * colors[i, :3]   # np.array([0,0,255])/255.0
        i = i + 1
        ov[ma == 1] = fg[ma == 1]
        total_ma += ma

        # [-2:] is s trick to be compatible both with opencv 2 and 3
        contours = cv2.findContours(ma.copy().astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        cv2.drawContours(ov, contours[0], -1, (0.0, 0.0, 0.0), 1)
    ov[total_ma == 0] = im[total_ma == 0]

    return ov

class points_to_skelet():
    def __init__(self, linewidth=2,
                        linefade=1,
                        pointwidth=2,
                        pointfade=3,
                        interp_array=None,
                        shape = (256,256)):
        super().__init__()
        self.shape = shape
        self.linewidth = linewidth
        self.linefade = linefade
        self.pointwidth = pointwidth
        self.pointfade = pointfade
        
        if isinstance(interp_array, int):
            if interp_array<2:
                self.interp_array = None
            else:
                self.interp_array = np.linspace(-0.5+0.5/interp_array,
                                                 0.5-0.5/interp_array,
                                                 interp_array).tolist()
        elif isinstance(interp_array, list):
            pass
        elif isinstance(interp_array, np.ndarray):
            self.interp_array = interp_array.tolist()
        else:
            self.interp_array = None
            
    def __call__(self,points,connectivity):
        points = torch.from_numpy(points.copy())
        connectivity = torch.from_numpy(connectivity.copy()).long()
        im = torch.zeros(self.shape)
        im = self._render_points(im, points)
        im = self._render_lines(im, points, connectivity)
        
        return im
    
    def _render_points(self,im,points):
        """
        Function used to add points to a single image effeciently.
        
        Parameters
        ----------
        im : torch.tensor
            Tensor of shape (H,W) with the image.
        points : torch.tensor
            Coordinates of points to be rendered in image coordinates. If
            n points are rendered then the shape should be (n,2).
    
        Returns
        -------
        im : torch.tensor
            Tensor of shape (H,W) with the image that has had points 
            rendered.
        """
        assert points.shape[1]==2
        n = points.shape[0]
        
        h_min = torch.floor(points[:,0]-self.pointwidth).type(torch.int)
        h_max = torch.ceil(points[:,0]+self.pointwidth).type(torch.int)+1
        w_min = torch.floor(points[:,1]-self.pointwidth).type(torch.int)
        w_max = torch.ceil(points[:,1]+self.pointwidth).type(torch.int)+1
        
        h_min[h_min<0] = 0
        h_max[h_max>im.shape[0]] = im.shape[0]
        w_min[w_min<0] = 0
        w_max[w_max>im.shape[1]] = im.shape[1]

        
        for i in range(n):
            H_idx,W_idx = torch.meshgrid(torch.arange(h_min[i],h_max[i]),
                                         torch.arange(w_min[i],w_max[i]),
                                         indexing='ij')
            dist = ((H_idx-points[i,0])**2+(W_idx-points[i,1])**2)**0.5

            im[h_min[i]:h_max[i],w_min[i]:w_max[i]] = 1-(
                                        (1-im[h_min[i]:h_max[i],w_min[i]:w_max[i]])*
                                        (1-self.render_dist_func(dist,
                                        width=self.pointwidth,
                                        fade=self.pointfade)))
        return im
    
    def _render_lines(self,im,line_points,connectivity):
        """
        Function used to add lines between points to a single image 
        effeciently.
        
        Parameters
        ----------
        im : torch.tensor
            Tensor of shape (H,W) with the image.
        line_points : torch.tensor
            Coordinates of endpoints of lines in image coordinates.
        connectivity : torch.tensor
            Tensor containing indices of line_points between which lines
            should be rendered. Shape of (n,2). For example if
            connectivity = torch.tensor([[0,2]]) then a line will be drawn
            between point number 0 and 2 in line_points.
    
        Returns
        -------
        im : torch.tensor
            Tensor of shape (H,W) with the image that has had lines 
            rendered.
        """
        
        assert line_points.shape[1]==2
        assert connectivity.shape[1]==2
        
        n_points = line_points.shape[0]
        
        assert n_points>=2
        assert connectivity.max()<=n_points
        assert connectivity.min()>=0
        
        h_min = torch.floor(line_points[:,0]-self.linewidth).type(torch.int)
        h_max = torch.ceil(line_points[:,0]+self.linewidth).type(torch.int)+1
        w_min = torch.floor(line_points[:,1]-self.linewidth).type(torch.int)
        w_max = torch.ceil(line_points[:,1]+self.linewidth).type(torch.int)+1
        
        h_min[h_min<0] = 0
        h_max[h_max>im.shape[0]] = im.shape[0]
        w_min[w_min<0] = 0
        w_max[w_max>im.shape[1]] = im.shape[1]
        
        for idx in connectivity:
            h_min_i = h_min[idx].min()
            h_max_i = h_max[idx].max()
            w_min_i = w_min[idx].min()
            w_max_i = w_max[idx].max()
            H_idx,W_idx = torch.meshgrid(torch.arange(h_min_i,h_max_i),
                                         torch.arange(w_min_i,w_max_i),
                                         indexing='ij')
            if self.interp_array is None:
                rendered = self.render_dist_func(
                                        self.dist_point_to_line(line_points[idx[0]],
                                        line_points[idx[1]],
                                        H_idx,
                                        W_idx),
                                                width=self.linewidth,
                                                fade=self.linefade)
            else:
                dist = torch.zeros(len(self.interp_array)**2,h_max_i-h_min_i,w_max_i-w_min_i)
                k = 0
                for h_t in self.interp_array:
                    for w_t in self.interp_array:
                        dist[k] = self.dist_point_to_line(line_points[idx[0]],
                                                          line_points[idx[1]],
                                                          H_idx+h_t,
                                                          W_idx+w_t)
                        k += 1
                rendered = self.render_dist_func(dist,
                                                 width=self.linewidth,
                                                 fade=self.linefade).mean(0)
            
            im[h_min_i:h_max_i,w_min_i:w_max_i] = 1-(
                                        (1-im[h_min_i:h_max_i,w_min_i:w_max_i])*
                                        (1-rendered))
        return im
    
    def dist_point_to_line(self,p1,p2,H_idx,W_idx,line_segment=True):
        """
        Calculates the distance for all points in (H_idx,W_idx) to the line
        that intersects p1 and p2.

        Parameters
        ----------
        p1 : torch.tensor
            Tensor of point one with 2 elements representing [h,w] coordinates.
        p2 : torch.tensor
            Tensor of point one with 2 elements representing [h,w] coordinates.
        H_idx : torch.tensor
            Tensor containing h coordinates that the distance is compute wrt.
        W_idx : torch.tensor
            Tensor containing w coordinates that the distance is compute wrt.
        line_segment : boolean, optional
            Should the line be infinite (False) or should it end at the 
            supplied points (True). The default is True.

        Returns
        -------
        dist : torch.tensor
            Distance of each point in (H_idx,W_idx) wrt. the line.
        """
        if all(p1==p2):
            return torch.ones_like(H_idx)*(H_idx.max()+W_idx.max())
        a = p1[1]-p2[1]
        b = p2[0]-p1[0]
        c = p1[0]*p2[1]-p1[1]*p2[0]
        div_constant = (a**2+b**2)**0.5
        a /= div_constant
        b /= div_constant
        c /= div_constant
        dist = torch.abs(H_idx*a+W_idx*b+c)
        if line_segment:
            p3 = (p1+p2)*0.5
            p4 = p3+torch.tensor([-1,1])*(p2-p1)[[1,0]]
            half_length = ((p3-p2)**2).sum()**0.5
            dist2 = self.dist_point_to_line(p3,p4,H_idx,W_idx,line_segment=False)-half_length
            non_lin_mask = dist2>0
            dist[non_lin_mask] = (dist[non_lin_mask]**2+dist2[non_lin_mask]**2)**0.5
        return dist
    
    def render_dist_func(self,dist,width=2,fade=1,mode="linear"):
        """
        Render intensities from a distance field.
        
        Parameters
        ----------
        dist : torch.tensor
            Distance field image of shape (h,w).
        width : float>=0
            Width of the rendered pixels, meaning all pixels with a distance of 
            less than width are nonzero
        fade : float>=0
            Fade width of the nonzero part of the rendered pixels. For example
            if width=2 and fade=1 then all pixels with a distance of 0<=x<=1
            will get pixel value 1, and all pixels with x>=0 will get pixel 
            value 0. Since the fade has width 1 then the interval of the fade
            1<=x<=2 has this length. In the example it fades linearly, e.g.
            x=1.6 will have intensity 0.4.
        mode : one of ["linear"]
            Which mode of fading from 0 to 1 is used. The default is "linear".

        Returns
        -------
        im: torch.tensor
            Intensities returned by the function taken on the distance field.

        """
        if mode.lower()=="linear":
            if fade>0:
                return torch.clamp((-1/fade)*dist + width/fade,min=0,max=1)
            else:
                return (dist<=width).type(torch.float)
        else:
            raise ValueError("Did not recognize render dist func mode: "+mode)