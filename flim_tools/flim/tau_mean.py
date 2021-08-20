# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 10:34:36 2021

@author: Nabiki
"""

def tau_mean(a1, t1, a2, t2):


    # ADD tm values
    # a1 = ma.masked_array(tifffile.imread(im_set['a1']), mask=m_roi)
    # t1 = ma.masked_array(tifffile.imread(im_set['t1']), mask=m_roi)
    # a2 = ma.masked_array(tifffile.imread(im_set['a2']), mask=m_roi)
    # t2 = ma.masked_array(tifffile.imread(im_set['t2']), mask=m_roi)
    
    
    list_roi_features_header.append("tm_mean")
    list_roi_features.append(np.mean(a1*t1 + a2*t2))
    
    list_roi_features_header.append("tm_stdev")
    list_roi_features.append(np.std(a1*t1 + a2*t2))

    return tau_mean