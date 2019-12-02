#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from numpy import zeros, ones, empty, nan, shape
from numpy import isnan, nanmean, nanmax, nanmin
import numpy.ma as ma
from netCDF4 import MFDataset, Dataset, num2date, date2num, date2index
import os
import matplotlib
import matplotlib.mlab as ml
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, cm, maskoceans
from scipy.interpolate import griddata
import math
import copy

from CMIP5lib import *


def func_latlon_regrid(lat_n_regrid, lon_n_regrid, lat_min_regrid, lat_max_regrid, lon_min_regrid, lon_max_regrid): 
    # This function centers lat at [... -1.5, -0.5, +0.5, +1.5 ...] - No specific equator lat cell
    ###lat_n_regrid, lon_n_regrid = 180, 360 # Number of Lat and Lon elements in the regridded data
    ###lon_min_regrid, lon_max_regrid = 0, 360 # Min and Max value of Lon in the regridded data
    ###lat_min_regrid, lat_max_regrid = -90, 90 # Min and Max value of Lat in the regridded data
    ####creating arrays of regridded lats and lons ###
    #### Latitude Bounds ####
    Lat_regrid_1D=zeros ((lat_n_regrid));
    Lat_bound_regrid = zeros ((lat_n_regrid,2)); Lat_bound_regrid[0,0]=-90;  Lat_bound_regrid[0,1]=Lat_bound_regrid[0,0] + (180/lat_n_regrid); Lat_regrid_1D[0]=(Lat_bound_regrid[0,0]+Lat_bound_regrid[0,1])/2
    for ii in range(1,lat_n_regrid):
        Lat_bound_regrid[ii,0]=Lat_bound_regrid[ii-1,1]
        Lat_bound_regrid[ii,1]=Lat_bound_regrid[ii,0] +  (180/lat_n_regrid)
        Lat_regrid_1D[ii]=(Lat_bound_regrid[ii,0]+Lat_bound_regrid[ii,1])/2
    #### Longitude Bounds ####
    Lon_regrid_1D=zeros ((lon_n_regrid));
    Lon_bound_regrid = zeros ((lon_n_regrid,2)); Lon_bound_regrid[0,0]=0;  Lon_bound_regrid[0,1]=Lon_bound_regrid[0,0] + (360/lon_n_regrid); Lon_regrid_1D[0]=(Lon_bound_regrid[0,0]+Lon_bound_regrid[0,1])/2
    for ii in range(1,lon_n_regrid):
        Lon_bound_regrid[ii,0]=Lon_bound_regrid[ii-1,1]
        Lon_bound_regrid[ii,1]=Lon_bound_regrid[ii,0] +  (360/lon_n_regrid)
        Lon_regrid_1D[ii]=(Lon_bound_regrid[ii,0]+Lon_bound_regrid[ii,1])/2
    
    return Lat_regrid_1D, Lon_regrid_1D, Lat_bound_regrid, Lon_bound_regrid

def func_latlon_regrid_eq(lat_n_regrid, lon_n_regrid, lat_min_regrid, lat_max_regrid, lon_min_regrid, lon_max_regrid): 
    # This function centers lat at [... -2, -1, 0, +1, +2 ...] - Creates an equator lat cell (lat=0)
    ###lat_n_regrid, lon_n_regrid = 180, 360 # Number of Lat and Lon elements in the regridded data
    ###lon_min_regrid, lon_max_regrid = 0, 360 # Min and Max value of Lon in the regridded data
    ###lat_min_regrid, lat_max_regrid = -90, 90 # Min and Max value of Lat in the regridded data
    ####creating arrays of regridded lats and lons ###
    #### Latitude Bounds ####
    Lat_regrid_1D=zeros ((lat_n_regrid+1));
    Lat_bound_regrid = zeros ((lat_n_regrid+1,2)); Lat_bound_regrid[0,0]=-90;  Lat_bound_regrid[0,1]=Lat_bound_regrid[0,0] + ( (180/lat_n_regrid) /2 ); Lat_regrid_1D[0]=(Lat_bound_regrid[0,0]+Lat_bound_regrid[0,1])/2
    for ii in range(1,lat_n_regrid+1):
        Lat_bound_regrid[ii,0]=Lat_bound_regrid[ii-1,1]
        Lat_bound_regrid[ii,1]=Lat_bound_regrid[ii,0] +  (180/lat_n_regrid)
        Lat_regrid_1D[ii]=(Lat_bound_regrid[ii,0]+Lat_bound_regrid[ii,1])/2
    Lat_bound_regrid[-1,1]=90;  Lat_regrid_1D[-1]=(Lat_bound_regrid[-1,0]+Lat_bound_regrid[-1,1])/2
    #### Longitude Bounds ####
    Lon_regrid_1D=zeros ((lon_n_regrid));
    Lon_bound_regrid = zeros ((lon_n_regrid,2)); Lon_bound_regrid[0,0]=0;  Lon_bound_regrid[0,1]=Lon_bound_regrid[0,0] + (360/lon_n_regrid); Lon_regrid_1D[0]=(Lon_bound_regrid[0,0]+Lon_bound_regrid[0,1])/2
    for ii in range(1,lon_n_regrid):
        Lon_bound_regrid[ii,0]=Lon_bound_regrid[ii-1,1]
        Lon_bound_regrid[ii,1]=Lon_bound_regrid[ii,0] +  (360/lon_n_regrid)
        Lon_regrid_1D[ii]=(Lon_bound_regrid[ii,0]+Lon_bound_regrid[ii,1])/2
    
    return Lat_regrid_1D, Lon_regrid_1D, Lat_bound_regrid, Lon_bound_regrid

def func_oceanlandmask(Lat_regrid_2D, Lon_regrid_2D):
    lat_n_regrid, lon_n_regrid =Lat_regrid_2D.shape[0], Lat_regrid_2D.shape[1]
    Ocean_Land_mask = empty ((lat_n_regrid, lon_n_regrid)) * nan
    ocean_mask= maskoceans(Lon_regrid_2D-180, Lat_regrid_2D, Ocean_Land_mask)
    for ii in range(lat_n_regrid):
        for jj in range(lon_n_regrid):
            if ma.is_masked(ocean_mask[ii,jj]):
                Ocean_Land_mask[ii,jj]=1 # Land_Ocean_mask=1 means grid cell is ocean (not on land)
            else:
                Ocean_Land_mask[ii,jj]=0 # Land_Ocean_mask=0 means grid cell is land
    land_mask2 = copy.deepcopy ( Ocean_Land_mask ) # The created land_mask's longitude is from -180-180 - following lines transfer it to 0-360
    Ocean_Land_mask=empty((Lat_regrid_2D.shape[0], Lat_regrid_2D.shape[1])) *nan
    Ocean_Land_mask[:,0:int(Ocean_Land_mask.shape[1]/2)]=land_mask2[:,int(Ocean_Land_mask.shape[1]/2):]
    Ocean_Land_mask[:,int(Ocean_Land_mask.shape[1]/2):]=land_mask2[:,0:int(Ocean_Land_mask.shape[1]/2)]
    
    return Ocean_Land_mask # 1= ocean, 0= land

def func_oceanindex (Lat_regrid_2D, Lon_regrid_2D):
    
    Ocean_Land_mask = func_oceanlandmask(Lat_regrid_2D, Lon_regrid_2D) # 1= ocean, 0= land
    
    #directory= '/data1/home/basadieh/behzadcodes/behzadlibrary/'
    directory = os.path.dirname(os.path.realpath(__file__)) # Gets the directory where the code is located - The gx3v5_OceanIndex.nc should be placed in the same directory
    file_name='gx3v5_OceanIndex.nc'
    dset_n = Dataset(directory+'/'+file_name)
    
    REGION_MASK=np.asarray(dset_n.variables['REGION_MASK'][:])
    TLAT=np.asarray(dset_n.variables['TLAT'][:])
    TLONG=np.asarray(dset_n.variables['TLONG'][:])
    
    REGION_MASK_regrid = func_regrid(REGION_MASK, TLAT, TLONG, Lat_regrid_2D, Lon_regrid_2D)
    Ocean_Index = copy.deepcopy(REGION_MASK_regrid)    
    for tt in range(0,6): # Smoothing the coastal gridcells - If a cell in the regrid has fallen on land but in Ocean_Land_mask it's in ocean, a neighboring Ocean_Index value will be assigned to it
        for ii in range(Ocean_Index.shape[0]):
            for jj in range(Ocean_Index.shape[1]):
                
                if Ocean_Index[ii,jj] == 0 and Ocean_Land_mask[ii,jj] == 1:
                    if ii>2 and jj>2:
                        Ocean_Index[ii,jj] = np.max(Ocean_Index[ii-1:ii+2,jj-1:jj+2])

    Ocean_Index[ np.where( np.logical_and(   np.logical_and( 0 <= Lon_regrid_2D , Lon_regrid_2D < 20 ) , np.logical_and( Lat_regrid_2D <= -30 , Ocean_Index != 0 )   ) ) ] = 6 ## Assigning Atlantic South of 30S to Atlantic Ocean Index
    Ocean_Index[ np.where( np.logical_and(   np.logical_and( 290 <= Lon_regrid_2D , Lon_regrid_2D <= 360 ) , np.logical_and( Lat_regrid_2D <= -30 , Ocean_Index != 0 )   ) ) ] = 6   
    Ocean_Index[ np.where( np.logical_and(   np.logical_and( 20 <= Lon_regrid_2D , Lon_regrid_2D < 150 ) , np.logical_and( Lat_regrid_2D <= -30 , Ocean_Index != 0 )   ) ) ] = 3 ## Assigning Pacifi South of 30S to Atlantic Ocean Index   
    Ocean_Index[ np.where( np.logical_and(   np.logical_and( 150 <= Lon_regrid_2D , Lon_regrid_2D < 290 ) , np.logical_and( Lat_regrid_2D <= -30 , Ocean_Index != 0 )   ) ) ] = 2 ## Assigning Pacifi South of 30S to Atlantic Ocean Index
    
    return Ocean_Index # [0=land] [2=Pacific] [3=Indian Ocean] [6=Atlantic] [10=Arctic] [8=Baffin Bay (west of Greenland)] [9=Norwegian Sea (east of Greenland)] [11=Hudson Bay (Canada)] 
                       # [-7=Mediterranean] [-12=Baltic Sea] [-13=Black Sea] [-5=Red Sea] [-4=Persian Gulf] [-14=Caspian Sea]

def func_atlantic_mask(Lat_regrid_2D, Lon_regrid_2D):
    lat_n_regrid, lon_n_regrid =Lat_regrid_2D.shape[0], Lat_regrid_2D.shape[1]
    Atlantic_mask = empty ((lat_n_regrid, lon_n_regrid)) * nan       
####     This is Atlantic Mask
    [Li,Lj] = np.where(np.logical_or(
                       np.logical_or((Lat_regrid_2D<=12) & (Lon_regrid_2D>=0) & (20>=Lon_regrid_2D),(Lat_regrid_2D<=12) & (290<=Lon_regrid_2D)& (Lon_regrid_2D<=360)),#### SH indeces
                       np.logical_or((Lat_regrid_2D>12) & (Lon_regrid_2D>=0) & (20>=Lon_regrid_2D),(Lat_regrid_2D>12) & (270<=Lon_regrid_2D) & (Lon_regrid_2D<=360))))#### NH indeces####
    Atlantic_mask[Li,Lj]=1
    return Atlantic_mask    

    
def func_regrid(Data_orig, Lat_orig, Lon_orig, Lat_regrid_2D, Lon_regrid_2D):    
    
    Lon_orig[Lon_orig < 0] +=360
    
    if np.ndim(Lon_orig)==1: # If the GCM grid is not curvlinear
        Lon_orig,Lat_orig=np.meshgrid(Lon_orig, Lat_orig)
        
    lon_vec = np.asarray(Lon_orig)
    lat_vec = np.asarray(Lat_orig)
    lon_vec = lon_vec.flatten()
    lat_vec = lat_vec.flatten()
    coords=np.squeeze(np.dstack((lon_vec,lat_vec)))

    Data_orig=np.squeeze(Data_orig)
    if Data_orig.ndim==2:#this is for 2d regridding
        data_vec = np.asarray(Data_orig)
        if np.ndim(data_vec)>1:
            data_vec = data_vec.flatten()
        Data_regrid = griddata(coords, data_vec, (Lon_regrid_2D, Lat_regrid_2D), method='nearest')
        return np.asarray(Data_regrid)
    if Data_orig.ndim==3:#this is for 3d regridding
        Data_regrid=[]
        for d in range(len(Data_orig)):
            z = np.asarray(Data_orig[d,:,:])
            if np.ndim(z)>1:
                z = z.flatten()
            zi = griddata(coords, z, (Lon_regrid_2D, Lat_regrid_2D), method='nearest')
            Data_regrid.append(zi)
        return np.asarray(Data_regrid)

    
def func_corr_3d(Var_y, Var_x):  
    from scipy import stats
    
    lat_n=Var_x.shape[1]
    lon_n=Var_x.shape[2]   
    Corr_xy = empty((lat_n,lon_n)) * nan
    Pvalue_xy = empty((lat_n,lon_n)) * nan  
    
    Var_x_mean=np.nanmean(Var_x,axis=0)
    
    for ii in range(0,lat_n):
        for jj in range(0,lon_n):
            if np.logical_not(np.isnan(Var_x_mean[ii,jj])): # If this grid cell has data and is non NaN
                xx=Var_x[:,ii,jj] ; yy=Var_y[:,ii,jj] # Creating a time series of the variabel for that grid cell                
                yy=yy[np.logical_not(np.isnan(xx))] ; xx=xx[np.logical_not(np.isnan(xx))] # Excluding any NaN values in the time series 
                xx=xx[np.logical_not(np.isnan(yy))] ; yy=yy[np.logical_not(np.isnan(yy))] # Excluding any NaN values in the time series 
                corr_ij, p_value_ij = stats.pearsonr(xx, yy)
                
                Corr_xy [ii,jj] = corr_ij
                Pvalue_xy [ii,jj] = p_value_ij  
                
    return Corr_xy, Pvalue_xy             

def func_detrend_3d(Var_orig):  
    from scipy import stats
    
    lat_n=Var_orig.shape[1]
    lon_n=Var_orig.shape[2] 
    time_n=Var_orig.shape[0]    
    detrend_mat = empty((time_n,lat_n,lon_n)) * nan
    
    Var_orig_mean=np.nanmean(Var_orig,axis=0)
    
    for ii in range(0,lat_n):
        for jj in range(0,lon_n):
            if np.logical_not(np.isnan(Var_orig_mean[ii,jj])): # If this grid cell has data and is non NaN
                
                yy=Var_orig[:,ii,jj]
                xx=np.asarray(range(1,time_n+1))              
                yy=yy[np.logical_not(np.isnan(xx))] ; xx=xx[np.logical_not(np.isnan(xx))] # Excluding any NaN values in the time series 
                xx=xx[np.logical_not(np.isnan(yy))] ; yy=yy[np.logical_not(np.isnan(yy))] # Excluding any NaN values in the time series 
               
                m_slope, b_intercept, r_val, p_val, std_err = stats.linregress(xx, yy)
                
                xx=np.asarray(range(1,time_n+1))                  
                detrend_mat[:,ii,jj] = m_slope*xx

    Var_detrend=Var_orig-detrend_mat
                
    return Var_detrend    
    

def func_plot(Plot_Var, Lat_img, Lon_img, bounds_max, Var_plot_unit, Plot_title, proj, lon_00):
    
    #Plot_Var [Plot_Var > 1E19] =nan
    # create figure
    fig = plt.figure()
    # create an Axes at an arbitrary location, which makes a list of [left, bottom, width, height] values in 0-1 relative figure coordinates:
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    # create Basemap instance.
    m = Basemap(projection=proj, lat_0=0, lon_0=lon_00)
    m.drawcoastlines(linewidth=1.25)
    m.fillcontinents(color='0.95')
    #m.drawmapboundary(fill_color='0.9')
    # cacluate colorbar ranges
    bounds = np.arange(0, bounds_max, bounds_max/40)
    norm = matplotlib.colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    # create a pseudo-color plot over the map
    im1 = m.pcolormesh(Lon_img, Lat_img, Plot_Var, norm=norm, shading='flat', cmap=plt.cm.jet, latlon=True)
    m.drawparallels(np.arange(-90.,90.001,30.),labels=[True,False,False,False], linewidth=0.01) # labels = [left,right,top,bottom]
    m.drawmeridians(np.arange(0.,360.,30.),labels=[False,False,False,True], linewidth=0.01) # labels = [left,right,top,bottom]
    # add colorbar
    cbar = m.colorbar(im1,"right", size="3%", pad="2%", extend='max') # extend='both' will extend the colorbar in both sides (upper side and down side)
    cbar.set_label(Var_plot_unit)
    #set title
    ax.set_title(Plot_title)
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized() # Maximizes the plot window to save figures in full
    plt.show()
    #fig.savefig(dir_figs+str(GCM)+'_Phydiat.png', format='png', dpi=300, transparent=True)
    #plt.close()

def func_plot_save(Plot_Var, Lat_img, Lon_img, bounds_max, Var_plot_unit, Plot_title, proj, lon_00, Plot_save_dir):
    
    #Plot_Var [Plot_Var > 1E19] =nan
    # create figure
    fig = plt.figure()
    # create an Axes at an arbitrary location, which makes a list of [left, bottom, width, height] values in 0-1 relative figure coordinates:
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    # create Basemap instance.
    m = Basemap(projection=proj, lat_0=0, lon_0=lon_00)
    m.drawcoastlines(linewidth=1.25)
    m.fillcontinents(color='0.95')
    #m.drawmapboundary(fill_color='0.9')
    # cacluate colorbar ranges
    bounds = np.arange(0, bounds_max, bounds_max/40)
    norm = matplotlib.colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    # create a pseudo-color plot over the map
    im1 = m.pcolormesh(Lon_img, Lat_img, Plot_Var, norm=norm, shading='flat', cmap=plt.cm.jet, latlon=True)
    m.drawparallels(np.arange(-90.,90.001,30.),labels=[True,False,False,False], linewidth=0.01) # labels = [left,right,top,bottom]
    m.drawmeridians(np.arange(0.,360.,30.),labels=[False,False,False,True], linewidth=0.01) # labels = [left,right,top,bottom]
    # add colorbar
    cbar = m.colorbar(im1,"right", size="3%", pad="2%", extend='max') # extend='both' will extend the colorbar in both sides (upper side and down side)
    cbar.set_label(Var_plot_unit)
    #set title
    ax.set_title(Plot_title)
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized() # Maximizes the plot window to save figures in full
    plt.show()
    fig.savefig(Plot_save_dir, format='png', dpi=300, transparent=True)
    #plt.close()

def func_plot_bounds_save(Plot_Var, Lat_img, Lon_img, bounds, Var_plot_unit, Plot_title, proj, lon_00, Plot_save_dir):
    
    #Plot_Var [Plot_Var > 1E19] =nan
    # create figure
    fig = plt.figure()
    # create an Axes at an arbitrary location, which makes a list of [left, bottom, width, height] values in 0-1 relative figure coordinates:
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    # create Basemap instance.
    m = Basemap(projection=proj, lat_0=0, lon_0=lon_00)
    m.drawcoastlines(linewidth=1.25)
    m.fillcontinents(color='0.95')
    #m.drawmapboundary(fill_color='0.9')
    # cacluate colorbar ranges
    #bounds = np.arange(0, bounds_max, bounds_max/40)
    norm = matplotlib.colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    # create a pseudo-color plot over the map
    im1 = m.pcolormesh(Lon_img, Lat_img, Plot_Var, norm=norm, shading='flat', cmap=plt.cm.jet, latlon=True)
    m.drawparallels(np.arange(-90.,90.001,30.),labels=[True,False,False,False], linewidth=0.01) # labels = [left,right,top,bottom]
    m.drawmeridians(np.arange(0.,360.,30.),labels=[False,False,False,True], linewidth=0.01) # labels = [left,right,top,bottom]
    # add colorbar
    cbar = m.colorbar(im1,"right", size="3%", pad="2%", extend='both') # extend='both' will extend the colorbar in both sides (upper side and down side)
    cbar.set_label(Var_plot_unit)
    #set title
    ax.set_title(Plot_title)
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized() # Maximizes the plot window to save figures in full
    plt.show()
    fig.savefig(Plot_save_dir, format='png', dpi=300, transparent=True)
    #plt.close()


def func_MLD(*args):
    import seawater as sw
    # args :
    # 0) netcdf_read object of thetao; (Temperature))
    # 1) netcdf_read object of so; (Salinity)
    # 2) month;
    # 3) 0/1/2, 0=SH, Weddell Sea, 1=NH, Labrador Sea, 2=NH, Norwegian Sea (SH below 50S; NH above 50N)
    # 4) year start
    # 5) year_end
    # 6) netcdf_read object of area
    # 7) lat grid numbers
    # 8) lon grid numbers
    if args[3]==0:
        depth_MLD_tr=2000 # Weddell Sea - [30E-60W]
    elif args[3]==1:
        depth_MLD_tr=1000 # Labrador Sea  - [60W-30W]
    elif args[3]==2:
        depth_MLD_tr=2000 # Norwegian Sea
    elif args[3]==3:
        depth_MLD_tr=1000 # Labrador Sea  - [60W-40W]   
    else:
        depth_MLD_tr=2000
        
    start,end = args[0].find_time(args[0].times, args[4], args[5])
    deep_conv_area=[]
#    areacello=args[6]
#    lon,lat,areacello=interpolate_2_custom_grid(args[0].x,args[0].y,areacello, args[7], args[8])
    if args[6]=='-':
        Lon_orig=args[0].x
        Lat_orig=args[0].y
        if np.ndim(Lon_orig)==1: # If the GCM grid is not curvlinear
            Lon_orig,Lat_orig=np.meshgrid(Lon_orig, Lat_orig)
        lat_n=Lat_orig.shape[0] # Number of Lat elements in the data
        lon_n=Lon_orig.shape[1] # Number of Lon elements in the data
        earth_R = 6378e3 # Earth Radius - Unit is kilometer (km)
        GridCell_Areas = zeros ((lat_n, lon_n )) # A = 2*pi*R^2 |sin(lat1)-sin(lat2)| |lon1-lon2|/360 = (pi/180)R^2 |lon1-lon2| |sin(lat1)-sin(lat2)| 
        for ii in range(1,lat_n-1):
            for jj in range(1,lon_n-1):
                GridCell_Areas [ii,jj] = math.fabs( (earth_R**2) * (math.pi/180) * np.absolute( (Lon_orig[ii,jj-1]+Lon_orig[ii,jj])/2  -  (Lon_orig[ii,jj]+Lon_orig[ii,jj+1])/2 )  * np.absolute( math.sin(math.radians( ( Lat_orig[ii-1,jj]+Lat_orig[ii,jj])/2 )) - math.sin(math.radians( Lat_orig[ii,jj]+Lat_orig[ii+1,jj])/2  )) )                  
        for ii in range(1,lat_n-1):
            for jj in range(2,lon_n-2):
                if GridCell_Areas [ii,jj] > GridCell_Areas [ii,jj-1]*3:
                    GridCell_Areas [ii,jj]=GridCell_Areas [ii,jj-1]
                if GridCell_Areas [ii,jj] > GridCell_Areas [ii,jj+1]*3:
                    GridCell_Areas [ii,jj]=GridCell_Areas [ii,jj+1]
        GridCell_Areas[0,:]=GridCell_Areas[1,:]; GridCell_Areas[-1,:]=GridCell_Areas[-2,:]
        GridCell_Areas[:,0]=GridCell_Areas[:,1]; GridCell_Areas[:,-1]=GridCell_Areas[:,-2]
        areacello=GridCell_Areas
    else:
        areacello=args[6]   
    
    Lat_regrid_1D, Lon_regrid_1D, Lat_bound_regrid, Lon_bound_regrid = func_latlon_regrid_eq(args[7], args[8], -90, 90, 0, 360)
    lon, lat = np.meshgrid(Lon_regrid_1D, Lat_regrid_1D)
    areacello = func_regrid(areacello, args[0].y, args[0].x, lat, lon)
    
    data_plot=np.full([args[5]-args[4]+1,len(lon),len(lon[0])], np.nan)
    for t in range(int((end+1-start)/12)):
        #print (int((end+1-start)/12))
        #print(start+12*t,start+12*t+11)
        print('MLD calc - Year: ', args[4]+t)
        data_thetao_extracted=args[0].extract_data(args[0].variable,start+12*t+args[2]-1,start+12*t+args[2]-1)
        data_so_extracted=args[1].extract_data(args[1].variable,start+12*t+args[2]-1,start+12*t+args[2]-1)
        data_thetao_extracted=np.squeeze(data_thetao_extracted)
        data_so_extracted=np.squeeze(data_so_extracted)
        data_dens=sw.dens0(data_so_extracted, data_thetao_extracted-273.15)
        depth10m_shalow=0
        depth10m_deep=0
        depth_array=np.asarray(args[0].lvl)
        
        for k in range(len(args[0].lvl[:])):
            if args[0].lvl[k]<=10:
                depth10m_shalow=k
        for k in range(len(args[0].lvl[:])):        
            if args[0].lvl[k]>=10:
                depth10m_deep+=k
                break
        
        interpol_x = [args[0].lvl[depth10m_shalow], args[0].lvl[depth10m_deep]]
##        lon=args[0].x
##        lat=args[0].y
        data_i=data_dens
#        lon,lat,data_i=interpolate_2_custom_grid(args[0].x,args[0].y,data_dens, args[7], args[8])
        data_i = func_regrid(data_dens, args[0].y, args[0].x, lat, lon)
        #lon,lat,data_i=interpolate_2_reg_grid(args[0].x,args[0].y,data_dens)
        data_i[data_i>100000]=np.nan
        
        if (int(args[3])==int(0)):# Weddell Sea
            [ii,jj] = np.where(lat<=-50)###indeces####
        elif (int(args[3])==int(1)):# Labrador Sea  - [60W-30W]
            [ii,jj] = np.where(lat>=50)###indeces####
        elif (int(args[3])==int(2)):# Norwegian Sea
            [ii,jj] = np.where(lat>=58)###indeces####
        elif (int(args[3])==int(3)):# Labrador Sea  - [60W-40W]  
            [ii,jj] = np.where(lat>=50)###indeces####            
        else:
            print(args[3])
            print('invalid input for hemisphere option')
            break
        area=0
        for k in range(len(ii)):
            if not(str(data_i[0,ii[k],jj[k]])=='nan'):
                dummy=100
                #MLD=0
                interpol_dens = [data_i[depth10m_shalow,ii[k],jj[k]], data_i[depth10m_deep,ii[k],jj[k]]]
                p_10m_dens = np.interp(10, interpol_x, interpol_dens)
                for d in range(len(data_i)):
                    if not(str(data_i[0,ii[k],jj[k]])=='nan'):
                        p_dens = data_i[d,ii[k],jj[k]]
                        if abs(p_dens-p_10m_dens-0.03)<dummy:
                            dummy=abs(p_dens-p_10m_dens-0.03)
                            MLD=d
                if MLD==0:
                    MLD+=1
                    p_dens_interpol = [data_i[MLD-1,ii[k],jj[k]]-p_10m_dens,data_i[MLD,ii[k],jj[k]]-p_10m_dens,data_i[MLD+1,ii[k],jj[k]]-p_10m_dens]
                    depth_levels = [depth_array[MLD-1],depth_array[MLD],depth_array[MLD+1]]
                ##elif MLD==49:
                elif MLD==len(data_i)-1: # If MLD is the last layer                   
                    MLD-=1
                    p_dens_interpol = [data_i[MLD-1,ii[k],jj[k]]-p_10m_dens,data_i[MLD,ii[k],jj[k]]-p_10m_dens,data_i[MLD+1,ii[k],jj[k]]-p_10m_dens]
                    depth_levels = [depth_array[MLD-1],depth_array[MLD],depth_array[MLD+1]]
                else:
                    p_dens_interpol = [data_i[MLD-1,ii[k],jj[k]]-p_10m_dens,data_i[MLD,ii[k],jj[k]]-p_10m_dens,data_i[MLD+1,ii[k],jj[k]]-p_10m_dens]
                    depth_levels = [depth_array[MLD-1],depth_array[MLD],depth_array[MLD+1]]
                interpol_z=np.interp(0.03,p_dens_interpol,depth_levels)
                if interpol_z>=depth_MLD_tr:
                #y1+=float(interpol_z)
                    area+=areacello[ii[k],jj[k]]
                    data_plot[t,ii[k],jj[k]]=float(interpol_z)     
        deep_conv_area.append(area)
    deep_conv_area=np.asarray(deep_conv_area)
##    st=np.nanstd(deep_conv_area)
##    st_area=area/st
##    st2 = np.where(st_area>=1.9)
##    average_MLD=np.nanmean(data_plot[st2],axis=0)
    average_MLD=np.nanmean(data_plot,axis=0)
    if args[3]==0: # SH, Weddell Sea
        indeces = np.where(np.logical_or((lon<=30) & (average_MLD>depth_MLD_tr), (lon>=300) &(average_MLD>depth_MLD_tr)))
    elif args[3]==1: # NH, Labrador Sea  - [60W-30W]
        indeces = np.where(np.logical_and((lon>=30) & (average_MLD>depth_MLD_tr), (lon<=330) &(average_MLD>depth_MLD_tr)))
    elif args[3]==2: # NH, Norwegian Sea
        indeces = np.where(np.logical_or((lon<=30) & (average_MLD>depth_MLD_tr), (lon>=345) &(average_MLD>depth_MLD_tr)))
    elif args[3]==3: # NH, Labrador Sea  - [60W-40W]
        indeces = np.where(np.logical_and((lon>=30) & (average_MLD>depth_MLD_tr), (lon<=320) &(average_MLD>depth_MLD_tr)))
    else: ### This should never be the case though ###
        indeces = np.where(np.logical_and((lon>=30) & (average_MLD>depth_MLD_tr), (lon<=330) &(average_MLD>depth_MLD_tr)))        
        
    return deep_conv_area, data_plot, lon, lat, indeces


def func_time_depth_plot(*args):
    # args :
    # 0) netcdf_read object;
    # 1) year start
    # 2) year_end
    # 3) indices
    # 4) lat
    # 5) lon
    # 6) depth for Convection Index
    start,end = args[0].find_time(args[0].times, args[1], args[2])
    [ii,jj]=args[3]
    region=[]
    
    Lat_regrid_1D, Lon_regrid_1D, Lat_bound_regrid, Lon_bound_regrid = func_latlon_regrid_eq(args[4], args[5], -90, 90, 0, 360)
    lon, lat = np.meshgrid(Lon_regrid_1D, Lat_regrid_1D)     
    
    for t in range(int((end+1-start)/12)):
        data=args[0].extract_data(args[0].variable,start+12*t,start+12*t+11)
        data=np.asarray(data)
        data[data>100000]=np.nan
        data=np.nanmean(data,axis=0)
        data=np.squeeze(data)
        #lon,lat,data_i=interpolate_2_custom_grid(args[0].x,args[0].y,data, args[4], args[5])
        data_i = func_regrid(data, args[0].y, args[0].x, lat, lon)
        data_i=data_i[:,ii,jj]
        
        print('time_depth_plot calc - Year: ', args[1]+t)
        region.append(np.nanmean(data_i,axis=1))

    a=args[0].find_depth(args[0].lvl, args[6], args[6])
    print(a[0])
    region=np.asarray(region)
    convection_index=region[:,a[0]]
    return region,convection_index


def func_stream(*args):
    ##### order of args :
    ##### 0) data (netcdf_read object from CMIP5lib.py),
    ##### 1) start year,
    ##### 2) end year,
    ##### 3) Mask,
    ##### 4) lat
    ##### 5) lon    
    depths=args[0].extract_depth() # Units: 'm'
    ### these are upper and lower depths of each cell in an ocean grid
    depths_b=args[0].extract_depth_bounds()
    ### calculate the depth of each cell in an ocean grid
    depths_r=depths_b[:,1]-depths_b[:,0]
    ### find timeindeces
    start,end = args[0].find_time(args[0].times, args[1], args[2])
    transport_lon_final=[] # SUM(V_y * dX * dZ) over longitudes, for all years
    transport_final=[]
    transport_0_1000=[]
    transport_1000_2000=[]
    transport_2000_3000=[]
    transport_3000_4000=[]
    transport_4000_5000=[]
    transport_5000_below=[]
    
    Depth_indx=np.zeros((5,4)) # Rows: Depths of 1000, 2000, 3000, 4000 and 5000 meters
                               # Columns: 0=row number, 1=depths, 2=depths lower range, 3=depths upper range    
    for dd in range(depths.shape[0]):
        if (depths_b[dd,0] < 1000) and (depths_b[dd,1] > 1000):
            Depth_indx[0,0]=dd; Depth_indx[0,1]=depths[dd]
            Depth_indx[0,2]=depths_b[dd,0]; Depth_indx[0,3]=depths_b[dd,1]
        if (depths_b[dd,0] < 2000) and (depths_b[dd,1] > 2000):
            Depth_indx[1,0]=dd; Depth_indx[1,1]=depths[dd]
            Depth_indx[1,2]=depths_b[dd,0]; Depth_indx[1,3]=depths_b[dd,1]
        if (depths_b[dd,0] < 3000) and (depths_b[dd,1] > 3000):
            Depth_indx[2,0]=dd; Depth_indx[2,1]=depths[dd]
            Depth_indx[2,2]=depths_b[dd,0]; Depth_indx[2,3]=depths_b[dd,1]
        if (depths_b[dd,0] < 4000) and (depths_b[dd,1] > 4000):
            Depth_indx[3,0]=dd; Depth_indx[3,1]=depths[dd]
            Depth_indx[3,2]=depths_b[dd,0]; Depth_indx[3,3]=depths_b[dd,1]
        if (depths_b[dd,0] < 5000) and (depths_b[dd,1] > 5000):
            Depth_indx[4,0]=dd; Depth_indx[4,1]=depths[dd]
            Depth_indx[4,2]=depths_b[dd,0]; Depth_indx[4,3]=depths_b[dd,1]

    Lat_regrid_1D, Lon_regrid_1D, Lat_bound_regrid, Lon_bound_regrid = func_latlon_regrid_eq(args[4], args[5], -90, 90, 0, 360)
    lon, lat = np.meshgrid(Lon_regrid_1D, Lat_regrid_1D)

    ### we averaging velocities over the year for monthly data
    for t in range(int((end+1-start)/12)):
        print ('Stream calc - Year: ', t)
        data_extracted=args[0].extract_data(args[0].variable,start+12*t,start+12*t+11)
        data=np.squeeze(data_extracted)
        data=np.mean(data, axis=0)
        
#        lon,lat,data_i=interpolate_2_reg_grid(args[0].x,args[0].y,data)
        data_i = func_regrid(data, args[0].y, args[0].x, lat, lon)
        
        ##### converting 1e+20 to nan ######
        data_i[data_i>1000]=np.nan
        if t==0: 
            data_depth=np.full([len(lon),len(lon[0])], np.nan)
            data_depth_ranges=np.full([len(data_i),len(data_i[0]),len(data_i[0][0])], np.nan)
        [ii,jj] = args[3]
        for k in range(len(ii)):
            #### I calculate the depth by looking how many nans is in the depth column
            if sum(~np.isnan(data_i[:,ii[k],jj[k]]))>0:
                if t==0:
                    data_depth[ii[k],jj[k]]=depths[sum(~np.isnan(data_i[:,ii[k],jj[k]]))-1]
                    for l in range(sum(~np.isnan(data_i[:,ii[k],jj[k]]))):
                        data_depth_ranges[l,ii[k],jj[k]]=depths_r[l]
                        
        #### calculating volume transport
        #### first multiplying by 111km*cos(lat)
        mul_by_lat=data_i*(np.cos(np.deg2rad(lat))*111321) # V_y * dX [Unit: m/s * m]
        #### second multiplying by depth
        transport=mul_by_lat*data_depth_ranges/1000000 # V_y * dX * dZ [Unit: m/s * m * m * 1e-6 = Sverdrup]
        #### calculating integral over dz
        transport_lon=np.nansum(transport,axis=2) # SUM(V_y * dX * dZ) over longitudes [Unit: m3/s * 1e-6 = Sverdrup]
        #### calculating cum integral over dz
        #stream=np.nancumsum(transport_lon,axis=0)
        transport_lon_final.append(transport_lon) # SUM(V_y * dX * dZ) over longitudes, for all years
        transport_0_1000.append(np.nanmean(transport[0:int(Depth_indx[0,0])+1,:,:],axis=0))
        transport_1000_2000.append(np.nanmean(transport[int(Depth_indx[0,0])+1:int(Depth_indx[1,0])+1,:,:],axis=0))        
        transport_2000_3000.append(np.nanmean(transport[int(Depth_indx[1,0])+1:int(Depth_indx[2,0])+1,:,:],axis=0))
        transport_3000_4000.append(np.nanmean(transport[int(Depth_indx[2,0])+1:int(Depth_indx[3,0])+1,:,:],axis=0))
        transport_4000_5000.append(np.nanmean(transport[int(Depth_indx[3,0])+1:int(Depth_indx[4,0])+1,:,:],axis=0))
        if depths.shape[0] - (int(Depth_indx[4,0])+1) ==1:
            transport_5000_below.append(transport[int(Depth_indx[4,0])+1,:,:])
        elif depths.shape[0] - (int(Depth_indx[4,0])+1) >=2:
            transport_5000_below.append(np.nanmean(transport[int(Depth_indx[4,0])+1:,:,:],axis=0))

    
        #transport_final.append(transport)
    transport_lon_final=np.asarray(transport_lon_final)

    transport_0_1000=np.asarray(transport_0_1000)
    transport_1000_2000=np.asarray(transport_1000_2000)      
    transport_2000_3000=np.asarray(transport_2000_3000)
    transport_3000_4000=np.asarray(transport_3000_4000)
    transport_4000_5000=np.asarray(transport_4000_5000)
    transport_5000_below=np.asarray(transport_5000_below)

    transport_0_1000_mean=np.nanmean(transport_0_1000,axis=0)
    transport_2000_3000_mean=np.nanmean(transport_2000_3000,axis=0)

    maxvals=np.nanmax(transport_0_1000_mean,axis=1)
    minvals=np.nanmin(transport_2000_3000_mean,axis=1)

    ind_max = np.array([np.argwhere(transport_0_1000_mean == [x]) for x in maxvals])
    ind_max=np.concatenate(ind_max).astype(None)
    ind_max=np.concatenate(ind_max).astype(None)
    ii_max=ind_max[0::2].astype(int)
    jj_max=ind_max[1::2].astype(int)
    ind_min = np.array([np.argwhere(transport_2000_3000_mean == [x]) for x in minvals])
    ind_min=np.concatenate(ind_min).astype(None)
    ind_min=np.concatenate(ind_min).astype(None)
    ii_min=ind_min[0::2].astype(int)
    jj_min=ind_min[1::2].astype(int)

    return transport_lon_final, transport_0_1000, transport_1000_2000, transport_2000_3000, transport_3000_4000, transport_4000_5000, transport_5000_below, data_depth, Depth_indx, lon, lat, ii_max,jj_max, ii_min, jj_min, transport_0_1000_mean, transport_2000_3000_mean

def func_ENSO(*args):
    ny, nx = 181, 360
    xmin, xmax = 0, 359
    ymin, ymax = -90, 90
    xi = np.linspace(xmin, xmax, nx)
    yi = np.linspace(ymin, ymax, ny)
    lon, lat = np.meshgrid(xi, yi)
    # ENSO index based on 190 220 -5 5 rectangele
    ##### order of args :
    ##### 0) data (netcdf_read object from CMIP5lib.py),
    ##### 1) start year,
    ##### 2) end year,
    ##### 3) plot option to check the depth and mask  = 1/0
    start,end = args[0].find_time(args[0].times, args[1], args[2])
    [ii,jj] = np.where(np.logical_and((lat<=5) &(lon>=190), (220>=lon) &(lat>=-5)))### NH indeces####
    
    Lat_regrid_1D, Lon_regrid_1D, Lat_bound_regrid, Lon_bound_regrid = func_latlon_regrid_eq(180, 360, -90, 90, 0, 360)
    lon, lat = np.meshgrid(Lon_regrid_1D, Lat_regrid_1D)   
    
    ENSO=[]
    for i in range(int((end+1-start)/12)):
        
        print ('ENSO calc - Year: ', i)
        data_extracted=args[0].extract_data(args[0].variable,start+12*i,start+12*i+11, 0, 1)
        data=np.squeeze(data_extracted)
        data=np.mean(data, axis=0)
        #lon,lat,data_i=interpolate_2_reg_grid(args[0].x,args[0].y,data)
        data_i = func_regrid(data, args[0].y, args[0].x, lat, lon)
        if i==0:
            data_plot=np.full([len(data_i),len(data_i[0])], np.nan)
        average=[]
        for k in range(len(ii)):
            if i==0:
                data_plot[ii[k],jj[k]]=data_i[ii[k],jj[k]]
            average.append(data_i[ii[k],jj[k]])
        average=np.asarray(average)
        average=np.nanmean(average)
        ENSO.append(average)
    ENSO=np.asarray(ENSO)
    ENSO=ENSO-np.nanmean(ENSO)

    return ENSO, data_plot, lon, lat

def func_NAO(*args):
    ##### NAO index based on 80W 30E 10N 85N rectangele
    ##### order of args :
    ##### 0) data (netcdf_read object from CMIP5lib.py),
    ##### 1) start year,
    ##### 2) end year,
    ##### 3) plot option to check the depth and mask  = 1/0
    NAO=[]
    start,end = args[0].find_time(args[0].times, args[1], args[2])
    for i in range(int((end+1-start)/12)):
        print ('NAO calc - Year: ', i)
        data_extracted=args[0].extract_data(args[0].variable,start+12*i,start+12*i+11, 0, 1)
        data=np.squeeze(data_extracted)
        data=np.mean(data, axis=0)
        #lon,lat,data_i=interpolate_2_reg_grid(args[0].y,args[0].x,data)
        data_i=data
        data_EOF=[]
        if i==0:
            #### Here I don not interpolate because of high memory cost of eigenvectors/values computation
            #### Get memory error for 180x360 grid
            #### Probably need to write additional coarse interpolation function which would interpolate on e.g. 90x180 grid
            xi = args[0].x
            yi = args[0].y
            if np.nanmin(xi<0):
                xi[xi < 0] += 360
            if np.ndim(xi)>1:
                lon=xi
                lat=yi
            else:
                lon, lat = np.meshgrid(xi, yi)
            [ii,jj] = np.where(np.logical_or((lat<=85) &(lon>=280)&(lat>=10), (30>=lon)&(lat<=85)&(lat>=10) ))### NH indeces####
            
            lat_f=[];
            lon_f=[];
            data_plot=np.full([len(data_i),len(data_i[0])], np.nan)
        for k in range(len(ii)):
            # Following Wang et al., 2017
            NAO_i=data_i[ii[k],jj[k]]*np.sqrt(np.cos(np.deg2rad(lat[ii[k],jj[k]])))
            data_EOF.append(NAO_i)
            if i==0:
                data_plot[ii[k],jj[k]]=data_i[ii[k],jj[k]]
                lat_f.append(lat[ii[k],jj[k]])
                lon_f.append(lon[ii[k],jj[k]])

        NAO.append(data_EOF)
    C=np.cov(np.transpose(NAO))
    C= np.array(C, dtype=np.float32)
    eigval,eigvec=np.linalg.eig(C)
    eigval=np.real(eigval)
    eigvec=np.real(eigvec)    
    time_series_NAO=np.dot(np.transpose(eigvec[:,0]),np.transpose(NAO))
    lat=np.unique(lat_f)
    lon=np.unique(lon_f)
    lon, lat = np.meshgrid(lon, lat)
    spatial_pattern_NAO=np.reshape(eigvec[:,0],(len(lat),len(lat[0])))
    time_series_NAO=(time_series_NAO-np.nanmean(time_series_NAO))/np.std(time_series_NAO)

    return time_series_NAO, spatial_pattern_NAO, lon, lat


def func_EOF (Calc_Var, Calc_Lat):
#%% Example :
#Calc_Var = data_set_regrid [:,10:61,300:]
#Calc_Lat = Lat_regrid_2D [10:61,300:]
#Calc_Lon = Lon_regrid_2D [10:61,300:]
#
#EOF_spatial_pattern, EOF_time_series, EOF_variance_prcnt = func_EOF (Calc_Var, Calc_Lat)
#%%
    
    EOF_all=[]
    for i in range(Calc_Var.shape[0]):
        
        print ('EOF calc - Year: ', i)
        data_i=Calc_Var[i,:,:]
        data_i=np.squeeze(data_i)        

        data_EOF=[]
        if i==0:
            [lat_ii,lon_jj] = np.where(~np.isnan(data_i))

        for kk in range(len(lat_ii)):
            EOF_i=data_i[lat_ii[kk],lon_jj[kk]]*np.sqrt(np.cos(np.deg2rad(Calc_Lat[lat_ii[kk],lon_jj[kk]])))
            data_EOF.append(EOF_i)
    
        EOF_all.append(data_EOF)    
    
    EOF_all=np.asarray(EOF_all)
    
    C=np.cov(np.transpose(EOF_all))
    #C= np.array(C, dtype=np.float32)
    eigval,eigvec=np.linalg.eig(C)
    eigval=np.real(eigval)
    eigvec=np.real(eigvec)
    
    EOF_spatial_pattern = empty((10,Calc_Var.shape[1],Calc_Var.shape[2]))*nan # Stores first 10 EOFs for spatial pattern map
    for ss in range(EOF_spatial_pattern.shape[0]):
        for kk in range(len(lat_ii)):
            EOF_spatial_pattern[ss,lat_ii[kk],lon_jj[kk]] = eigvec[kk,ss]

    EOF_time_series = empty((10,Calc_Var.shape[0]))*nan # Stores first 10 EOFs times series
    for ss in range(EOF_time_series.shape[0]):
        EOF_time_series[ss,:] = np.dot(np.transpose(eigvec[:,ss]),np.transpose(EOF_all))
        
    EOF_variance_prcnt = empty((10))*nan # Stores first 10 EOFs variance percentage
    for ss in range(EOF_variance_prcnt.shape[0]):
        EOF_variance_prcnt[ss]=( eigval[ss]/np.nansum(eigval,axis=0) ) * 100        

    return EOF_spatial_pattern, EOF_time_series, EOF_variance_prcnt





    
  
def func_regrid_old(GCM, Data_orig, Lat_orig, Lon_orig, Lat_regrid_2D, Lon_regrid_2D, curvilinear, regrid_method):

    data_vec=[] # Data to be regrided, converted into a vector, excluding nan/missing arrays
    lon_vec=[] # Lon_orig converted to a vector
    lat_vec=[]
    for ii in range(Data_orig.shape[0]): # Loop over rows
        for jj in range(Data_orig.shape[1]): # Loop over columns
            # creating variables, used to calculate averages for each lat lon grid point
            if Data_orig[ii,jj] < 1e19: # Values equal to 1e20 are empty arrays # Only selecting grids with available data
                data_vec.append(Data_orig[ii,jj])
                # appending lat lons, which will be used in regriding the data
                # check for curvelinear coordinates
                if curvilinear==1:
                    # check if the GCM is GFDL, as GFDL lons start at -270 and go to 90
                    if GCM[:4]=='GFDL':
                        if Lon_orig[ii,jj]<=0:
                            # converting -270 - 90 lon range to 0 - 360 lon range
                            lon_vec.append(Lon_orig[ii,jj]+360)
                        else:  
                            lon_vec.append(Lon_orig[ii,jj])
                    else:
                        lon_vec.append(Lon_orig[ii,jj])
                    lat_vec.append(Lat_orig[ii,jj])
                else:
                    lon_vec.append(Lon_orig[jj])
                    lat_vec.append(Lat_orig[ii]) 
                    
    # converting to numpy arrays, as regriding does not work with lists
    lon_vec = np.asarray(lon_vec)
    lat_vec = np.asarray(lat_vec)
    data_vec = np.asarray(data_vec)
    lon_lat_vec=zeros((lon_vec.shape[0],2)) # Combining lat and lon vectors into one vector
    lon_lat_vec[:,0]=lon_vec
    lon_lat_vec[:,1]=lat_vec
    
    # converting the old grid to a new, using nearest neighbour interpolation
    #Data_regrid = ml.griddata(lon_vec, lat_vec, data_vec, Lon_regrid, Lat_regrid, interp='linear')
    Data_regrid = griddata(lon_lat_vec, data_vec, (Lon_regrid_2D, Lat_regrid_2D), method=regrid_method) # Options: method='linear' , method='cubic', method='nearest'
    
    return Data_regrid
    
    
    
    
    
    