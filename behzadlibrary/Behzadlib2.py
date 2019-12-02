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
from Behzadlib import func_latlon_regrid, func_latlon_regrid_eq, func_oceanlandmask, func_oceanindex, func_regrid

#%%
def func_barotropicstream_Vx_DyIntegrated(*args):
    ##### order of args :
    ##### 0) data (netcdf_read object from CMIP5lib.py),
    ##### 1) start year,
    ##### 2) end year,
    ##### 3) Higher bound of lat
    ##### 4) Lower bound of lat
    ##### 5) lat
    ##### 6) lon   
#%% Example :
#GCM = 'GFDL-ESM2G'; year_start=1; year_end=10
#t_frequency='Omon'  ; variable='uo'# vo = sea_water_x_velocity - units: m s-1
#dir_data_in1 = ('/data5/scratch/CMIP5/CMIP5_models/ocean_physics/') # Directory to raed raw data from
#dir_data_in2=(dir_data_in1+ GCM + '/piControl/mo/')
#dset_uo = netcdf_read (dir_data_in2+str(variable)+'_'+str(t_frequency)+'_'+str(GCM)+'*12.nc',str(GCM), variable) # Specifies the netCDF dataset (consisting of multiple files) from which the data will be read
#Lat_refrence_H = -30 # Higher bound of lat
#Lat_refrence_L = -90 # Lower bound of lat
#Stream_function_Barotropic_SO_allyears = func_barotropicstream_Vx_DyIntegrated(dset_uo, year_start, year_end, Lat_refrence_H, Lat_refrence_L, 180, 360)
#%%      
    depths=args[0].extract_depth() # Units: 'm'
    ### these are upper and lower depths of each cell in an ocean grid
    depths_b=args[0].extract_depth_bounds()
    ### calculate the depth of each cell in an ocean grid
    depths_r=depths_b[:,1]-depths_b[:,0]
    ### find timeindeces
    start,end = args[0].find_time(args[0].times, args[1], args[2])
    transport_depth_lat_sum_allyears=[] # SUM(V_x * dZ * dY) [Unit: m3/s * 1e-6 = Sverdrup], for all years

    Lat_regrid_1D, Lon_regrid_1D, Lat_bound_regrid, Lon_bound_regrid = func_latlon_regrid_eq(args[5], args[6], -90, 90, 0, 360)
    lon, lat = np.meshgrid(Lon_regrid_1D, Lat_regrid_1D)

    ### we averaging velocities over the year for monthly data
    for t in range(int((end+1-start)/12)):
        print ('BarotropicStream calc - Year: ', t+1)
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

        [ii,jj] = np.where(np.logical_and((lat<=args[3]) , (lat>=args[4])))  
        for k in range(len(ii)):
            #### I calculate the depth by looking how many nans is in the depth column
            if sum(~np.isnan(data_i[:,ii[k],jj[k]]))>0:
                if t==0:
                    data_depth[ii[k],jj[k]]=depths[sum(~np.isnan(data_i[:,ii[k],jj[k]]))-1]
                    for l in range(sum(~np.isnan(data_i[:,ii[k],jj[k]]))):
                        data_depth_ranges[l,ii[k],jj[k]]=depths_r[l]
        

        #### multiplying by depth
        transport_depth=data_i*data_depth_ranges # V_x * dZ [Unit: m/s * m ]
        #### calculating integral over dz
        transport_depth=np.nansum(transport_depth,axis=0) # SUM(V_x * dZ) over depth [Unit: m2/s ]  
        #### multiplying by latitude (dY)
        transport_depth_lat = transport_depth *111321 # SUM(V_x * dZ * dY) [Unit: m/s * m * m]
        transport_depth_lat = transport_depth_lat /1000000 # SUM(V_x * dZ * dY) [Unit: m3/s * 1e-6 = Sverdrup]
        
        transport_depth_lat_sum=zeros((transport_depth.shape[0], transport_depth.shape[1]))
        #transport_depth_lat_sum[0,:]=transport_depth[0,:]
        for ii in range(transport_depth.shape[0]-2,0,-1):
            help_transport = transport_depth_lat[ii,:];  help_transport[ np.isnan(help_transport) ] =0
            transport_depth_lat_sum[ii,:]=transport_depth_lat_sum[ii+1,:]+help_transport
            #transport_depth_lat_sum[ii,:]= np.nansum( transport_depth_lat_sum[ii-1,:],transport_depth_lat_sum[ii,:])  
        
#        transport_depth_lat_sum_final=zeros((transport_depth_lat_sum.shape[0], transport_depth_lat_sum.shape[1]))
#        aaa=np.nansum(transport_depth_lat, axis=0)
#        for ii in range(0, transport_depth.shape[0]):
#            transport_depth_lat_sum_final[ii,:] =  (transport_depth_lat_sum[ii,:] - aaa) * -1

        transport_depth_lat_sum_allyears.append(transport_depth_lat_sum) # SUM(V_x * dZ * dY) [Unit: m3/s * 1e-6 = Sverdrup], for all years
    
        #transport_final.append(transport)
    transport_depth_lat_sum_allyears=np.asarray(transport_depth_lat_sum_allyears)


    return transport_depth_lat_sum_allyears
#%%

def func_barotropicstream_Vy_DxIntegrated(*args):
    ##### order of args :
    ##### 0) data (netcdf_read object from CMIP5lib.py),
    ##### 1) start year,
    ##### 2) end year,
    ##### 3) Higher bound of lat
    ##### 4) Lower bound of lat
    ##### 5) lat
    ##### 6) lon    
#%% Example :
#GCM = 'GFDL-ESM2G'; year_start=1; year_end=10
#t_frequency='Omon'  ; variable='vo'# vo = sea_water_y_velocity - units: m s-1
#dir_data_in1 = ('/data5/scratch/CMIP5/CMIP5_models/ocean_physics/') # Directory to raed raw data from
#dir_data_in2=(dir_data_in1+ GCM + '/piControl/mo/')
#dset_vo = netcdf_read (dir_data_in2+str(variable)+'_'+str(t_frequency)+'_'+str(GCM)+'*12.nc',str(GCM), variable) # Specifies the netCDF dataset (consisting of multiple files) from which the data will be read
#Lat_refrence_H = 90 # Higher bound of lat
#Lat_refrence_L = 0 # Lower bound of lat
#Stream_function_Barotropic_NAtl_VyDx_allyears = func_barotropicstream_Vy_DxIntegrated(dset_vo, year_start, year_end, Lat_refrence_H, Lat_refrence_L, 180, 360)
#%%          
    Lat_regrid_1D, Lon_regrid_1D, Lat_bound_regrid, Lon_bound_regrid = func_latlon_regrid_eq(180, 360, -90, 90, 0, 360)
    Lon_regrid_2D, Lat_regrid_2D = np.meshgrid(Lon_regrid_1D, Lat_regrid_1D)
    Ocean_Land_mask = func_oceanlandmask(Lat_regrid_2D, Lon_regrid_2D) # 1= ocean, 0= land
    Ocean_Index = func_oceanindex (Lat_regrid_2D, Lon_regrid_2D) # [0=land] [2=Pacific] [3=Indian Ocean] [6=Atlantic] [10=Arctic] [8=Baffin Bay (west of Greenland)] [9=Norwegian Sea (east of Greenland)] [11=Hudson Bay (Canada)] 
    Ocean_Index[Ocean_Index==8]=6; Ocean_Index[Ocean_Index==9]=6; # Converting the left and right seas of Greenland to Atlantix
    
    depths=args[0].extract_depth() # Units: 'm'
    ### these are upper and lower depths of each cell in an ocean grid
    depths_b=args[0].extract_depth_bounds()
    ### calculate the depth of each cell in an ocean grid
    depths_r=depths_b[:,1]-depths_b[:,0]
    ### find timeindeces
    start,end = args[0].find_time(args[0].times, args[1], args[2])
    transport_depth_lon_sum_allyears=[] # SUM(V_y * dZ * dX) [Unit: m3/s * 1e-6 = Sverdrup], for all years

    Lat_regrid_1D, Lon_regrid_1D, Lat_bound_regrid, Lon_bound_regrid = func_latlon_regrid_eq(args[5], args[6], -90, 90, 0, 360)
    lon, lat = np.meshgrid(Lon_regrid_1D, Lat_regrid_1D)

    ### we averaging velocities over the year for monthly data
    for t in range(int((end+1-start)/12)):
        print ('BarotropicStream calc - Year: ', t+1)
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

        [ii,jj] = np.where(np.logical_and((lat<=args[3]) , (lat>=args[4])))  
        for k in range(len(ii)):
            #### I calculate the depth by looking how many nans is in the depth column
            if sum(~np.isnan(data_i[:,ii[k],jj[k]]))>0:
                if t==0:
                    data_depth[ii[k],jj[k]]=depths[sum(~np.isnan(data_i[:,ii[k],jj[k]]))-1]
                    for l in range(sum(~np.isnan(data_i[:,ii[k],jj[k]]))):
                        data_depth_ranges[l,ii[k],jj[k]]=depths_r[l]
        

        #### multiplying by depth
        transport_depth=data_i*data_depth_ranges # V_y * dZ [Unit: m/s * m ]
        #### calculating integral over dz
        transport_depth=np.nansum(transport_depth,axis=0) # SUM(V_y * dZ) over depth [Unit: m2/s ]  
        #### multiplying by latitude (dY)
        transport_depth_lon = transport_depth *(np.cos(np.deg2rad(lat))*111321) # SUM(V_y * dZ * dX) [Unit: m/s * m * m]
        transport_depth_lon = transport_depth_lon /1000000 # SUM(V_y * dZ * dX) [Unit: m3/s * 1e-6 = Sverdrup]
        
        transport_depth_lon_atl=copy.deepcopy(transport_depth_lon) # Atlantic Ocean Only
        transport_depth_lon_atl [ Ocean_Index!=6 ] = nan
        
        transport_depth_lon_pac=copy.deepcopy(transport_depth_lon) # Pacific Ocean Only
        transport_depth_lon_pac [ Ocean_Index!=2 ] = nan       
        
        transport_depth_lon_ind=copy.deepcopy(transport_depth_lon) # Indian Ocean Only
        transport_depth_lon_ind [ Ocean_Index!=3 ] = nan              
        
        transport_depth_lon_atl_sum=zeros((transport_depth.shape[0], transport_depth.shape[1]))
        transport_counter=zeros((transport_depth.shape[0]))
        #transport_depth_lat_sum[0,:]=transport_depth[0,:]
        for jj in range(40, -100,-1):
            help_transport = transport_depth_lon_atl[:,jj];  help_transport[ np.isnan(help_transport) ] =0
            transport_counter = transport_counter + help_transport
            transport_depth_lon_atl_sum[:,jj]=transport_counter

        transport_depth_lon_pac_sum=zeros((transport_depth.shape[0], transport_depth.shape[1]))
        transport_counter=zeros((transport_depth.shape[0]))
        #transport_depth_lat_sum[0,:]=transport_depth[0,:]
        for jj in range(-70, -270,-1):
            help_transport = transport_depth_lon_pac[:,jj];  help_transport[ np.isnan(help_transport) ] =0
            transport_counter = transport_counter + help_transport
            transport_depth_lon_pac_sum[:,jj]=transport_counter

        transport_depth_lon_ind_sum=zeros((transport_depth.shape[0], transport_depth.shape[1]))
        transport_counter=zeros((transport_depth.shape[0]))
        #transport_depth_lat_sum[0,:]=transport_depth[0,:]
        for jj in range(-250, -330,-1):
            help_transport = transport_depth_lon_ind[:,jj];  help_transport[ np.isnan(help_transport) ] =0
            transport_counter = transport_counter + help_transport
            transport_depth_lon_ind_sum[:,jj]=transport_counter
       
        transport_depth_lon_sum=zeros((transport_depth.shape[0], transport_depth.shape[1]))
        for ii in range (0,transport_depth.shape[0]):
            for jj in range (0,transport_depth.shape[1]):
                if Ocean_Index[ii,jj]==6:
                    transport_depth_lon_sum[ii,jj]=transport_depth_lon_atl_sum[ii,jj]
                if Ocean_Index[ii,jj]==2:
                    transport_depth_lon_sum[ii,jj]=transport_depth_lon_pac_sum[ii,jj]
                if Ocean_Index[ii,jj]==3:
                    transport_depth_lon_sum[ii,jj]=transport_depth_lon_ind_sum[ii,jj]                   

        transport_depth_lon_sum_allyears.append(transport_depth_lon_sum) # SUM(V_y * dZ * dX) [Unit: m3/s * 1e-6 = Sverdrup], for all years
    
        #transport_final.append(transport)
    transport_depth_lon_sum_allyears=np.asarray(transport_depth_lon_sum_allyears)


    return transport_depth_lon_sum_allyears
#%%

def func_MLD_AllYears_annual_ave(*args):
    import seawater as sw
    # args :
    # 0) netcdf_read object of thetao; (Temperature))
    # 1) netcdf_read object of so; (Salinity)
    # 2) year start
    # 3) year_end
    # 4) lat grid numbers
    # 5) lon grid numbers
#%% Example :
#t_frequency='Omon'
#variable_thetao='thetao' # Sea Water Temperature
#variable_so='so' # Sea Water Salinity
#dir_data_in1 = ('/data5/scratch/CMIP5/CMIP5_models/ocean_physics/') # Directory to raed raw data from
#dir_data_in2=(dir_data_in1+ GCM + '/piControl/mo/')
#dset_thetao = netcdf_read (dir_data_in2+str(variable_thetao)+'_'+str(t_frequency)+'_'+str(GCM)+'*12.nc',str(GCM), variable_thetao) # Specifies the netCDF dataset (consisting of multiple files) from which the data will be read
#dset_so = netcdf_read (dir_data_in2+str(variable_so)+'_'+str(t_frequency)+'_'+str(GCM)+'*12.nc',str(GCM), variable_so)
#MLD_years_monthlyave = func_MLD_AllYears_months_ave(dset_thetao, dset_so, year_start, year_end, 180, 360)    
#%%           
    start,end = args[0].find_time(args[0].times, args[2], args[3])

    Lat_regrid_1D, Lon_regrid_1D, Lat_bound_regrid, Lon_bound_regrid = func_latlon_regrid_eq(args[4], args[5], -90, 90, 0, 360)
    lon, lat = np.meshgrid(Lon_regrid_1D, Lat_regrid_1D)
    
    MLD_years=np.full([args[3]-args[2]+1,len(lon),len(lon[0])], np.nan)
    for t in range(int((end+1-start)/12)):
        #print (int((end+1-start)/12))
        #print(start+12*t,start+12*t+11)
        print('MLD calc - Year: ', args[2]+t)
        data_thetao_extracted=args[0].extract_data(args[0].variable,start+12*t+1-1,start+12*t+12-1)
        data_so_extracted=args[1].extract_data(args[1].variable,start+12*t+1-1,start+12*t+12-1)
        data_thetao_extracted=np.nanmean(data_thetao_extracted, axis=0)
        data_so_extracted=np.nanmean(data_so_extracted, axis=0)
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
        data_i=data_dens
        data_i = func_regrid(data_dens, args[0].y, args[0].x, lat, lon)
        data_i[data_i>100000]=np.nan
        
        [ii,jj] = np.where(lat>=-2000)###indeces####

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

                MLD_years[t,ii[k],jj[k]]=float(interpol_z)     

    return MLD_years
#%%

def func_MLD_AllYears_months_ave(*args):
    import seawater as sw
    # args :
    # 0) netcdf_read object of thetao; (Temperature))
    # 1) netcdf_read object of so; (Salinity)
    # 2) year start
    # 3) year_end
    # 4) lat grid numbers
    # 5) lon grid numbers
#%% Example :
#t_frequency='Omon'
#variable_thetao='thetao' # Sea Water Temperature
#variable_so='so' # Sea Water Salinity
#dir_data_in1 = ('/data5/scratch/CMIP5/CMIP5_models/ocean_physics/') # Directory to raed raw data from
#dir_data_in2=(dir_data_in1+ GCM + '/piControl/mo/')
#dset_thetao = netcdf_read (dir_data_in2+str(variable_thetao)+'_'+str(t_frequency)+'_'+str(GCM)+'*12.nc',str(GCM), variable_thetao) # Specifies the netCDF dataset (consisting of multiple files) from which the data will be read
#dset_so = netcdf_read (dir_data_in2+str(variable_so)+'_'+str(t_frequency)+'_'+str(GCM)+'*12.nc',str(GCM), variable_so)
#MLD_years_monthlyave = func_MLD_AllYears_months_ave(dset_thetao, dset_so, year_start, year_end, 180, 360)    
#%%           
    start,end = args[0].find_time(args[0].times, args[2], args[3])

    Lat_regrid_1D, Lon_regrid_1D, Lat_bound_regrid, Lon_bound_regrid = func_latlon_regrid_eq(args[4], args[5], -90, 90, 0, 360)
    lon, lat = np.meshgrid(Lon_regrid_1D, Lat_regrid_1D)
    
    MLD_years=np.full([args[3]-args[2]+1,len(lon),len(lon[0])], np.nan)
    for t in range(int((end+1-start)/12)):
        print('MLD calc - Year: ', args[2]+t)
        data_thetao_extracted=args[0].extract_data(args[0].variable,start+12*t+1-1,start+12*t+12-1)
        data_so_extracted=args[1].extract_data(args[1].variable,start+12*t+1-1,start+12*t+12-1)
        
        MLD_months=np.full([12,len(lon),len(lon[0])], np.nan)
        for mm in range(0,12):
            
            data_thetao_extracted_m=data_thetao_extracted[mm,:,:,:]
            data_so_extracted_m=data_so_extracted[mm,:,:,:]
            
            data_thetao_extracted_m=np.squeeze(data_thetao_extracted_m)
            data_so_extracted_m=np.squeeze(data_so_extracted_m)
            
            data_dens=sw.dens0(data_so_extracted_m, data_thetao_extracted_m-273.15)
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
            data_i=data_dens
            data_i = func_regrid(data_dens, args[0].y, args[0].x, lat, lon)
            data_i[data_i>100000]=np.nan
            
            [ii,jj] = np.where(lat>=-2000)###indeces####
    
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
    
                    MLD_months[mm,ii[k],jj[k]]=float(interpol_z) 
            
        MLD_months_ave=np.nanmean(MLD_months, axis=0)
            
        MLD_years[t,:,:]=MLD_months_ave

    return MLD_years
#%%
    


def func_MLD_AllYears_1month(*args):
    import seawater as sw
    # args :
    # 0) netcdf_read object of thetao; (Temperature))
    # 1) netcdf_read object of so; (Salinity)
    # 2) month number of year;
    # 3) year start
    # 4) year_end
    # 5) lat grid numbers
    # 6) lon grid numbers
#%% Example :
#t_frequency='Omon'
#variable_thetao='thetao' # Sea Water Temperature
#variable_so='so' # Sea Water Salinity
#dir_data_in1 = ('/data5/scratch/CMIP5/CMIP5_models/ocean_physics/') # Directory to raed raw data from
#dir_data_in2=(dir_data_in1+ GCM + '/piControl/mo/')
#dset_thetao = netcdf_read (dir_data_in2+str(variable_thetao)+'_'+str(t_frequency)+'_'+str(GCM)+'*12.nc',str(GCM), variable_thetao) # Specifies the netCDF dataset (consisting of multiple files) from which the data will be read
#dset_so = netcdf_read (dir_data_in2+str(variable_so)+'_'+str(t_frequency)+'_'+str(GCM)+'*12.nc',str(GCM), variable_so)
#month_no=9 # args[2] in main code # month of the year, 9=September, 3=March
#MLD_years_september = func_MLD_AllYears_1month(dset_thetao, dset_so, month_no, year_start, year_end, 180, 360)
#month_no=3 # args[2] in main code # month of the year, 9=September, 3=March
#MLD_years_march = func_MLD_AllYears_1month(dset_thetao, dset_so, month_no, year_start, year_end, 180, 360)
#%%           
    start,end = args[0].find_time(args[0].times, args[3], args[4])

    Lat_regrid_1D, Lon_regrid_1D, Lat_bound_regrid, Lon_bound_regrid = func_latlon_regrid_eq(args[5], args[6], -90, 90, 0, 360)
    lon, lat = np.meshgrid(Lon_regrid_1D, Lat_regrid_1D)
    
    MLD_years=np.full([args[4]-args[3]+1,len(lon),len(lon[0])], np.nan)
    for t in range(int((end+1-start)/12)):
        print('MLD calc - Year: ', args[3]+t)
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
        data_i=data_dens
        data_i = func_regrid(data_dens, args[0].y, args[0].x, lat, lon)
        data_i[data_i>100000]=np.nan
        
        [ii,jj] = np.where(lat>=-2000)###indeces####

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

                MLD_years[t,ii[k],jj[k]]=float(interpol_z)     

    return MLD_years
#%%
   
    
def func_MLD_Allmonths_bio(*args):
    import seawater as sw
    # args :
    # 0) netcdf_read object of thetao; (Temperature))
    # 1) netcdf_read object of so; (Salinity)
    # 2) year start
    # 3) year_end
    # 4) lat grid numbers
    # 5) lon grid numbers
#%% Example :
#t_frequency='Omon'
#variable_thetao='thetao' # Sea Water Temperature
#variable_so='so' # Sea Water Salinity
#dir_data_in1 = ('/data5/scratch/CMIP5/CMIP5_models/ocean_physics/') # Directory to raed raw data from
#dir_data_in2=(dir_data_in1+ GCM + '/piControl/mo/')
#dset_thetao = netcdf_read (dir_data_in2+str(variable_thetao)+'_'+str(t_frequency)+'_'+str(GCM)+'*12.nc',str(GCM), variable_thetao) # Specifies the netCDF dataset (consisting of multiple files) from which the data will be read
#dset_so = netcdf_read (dir_data_in2+str(variable_so)+'_'+str(t_frequency)+'_'+str(GCM)+'*12.nc',str(GCM), variable_so)
#MLD_years_monthlyave = func_MLD_AllYears_months_ave(dset_thetao, dset_so, year_start, year_end, 180, 360)    
#%%           
    start,end = args[0].find_time(args[0].times, args[2], args[3])

    Lat_regrid_1D, Lon_regrid_1D, Lat_bound_regrid, Lon_bound_regrid = func_latlon_regrid(args[4], args[5], -90, 90, 0, 360)
    lon, lat = np.meshgrid(Lon_regrid_1D, Lat_regrid_1D)
    
    MLD_Allmonths=np.full([ (args[3]-args[2]+1) * 12,len(lon),len(lon[0])], np.nan)
    for t in range(int((end+1-start)/12)):
        print('MLD calc - Year: ', args[2]+t)
        data_thetao_extracted=args[0].extract_data(args[0].variable,start+12*t+1-1,start+12*t+12-1)
        data_so_extracted=args[1].extract_data(args[1].variable,start+12*t+1-1,start+12*t+12-1)
        
        MLD_months=np.full([12,len(lon),len(lon[0])], np.nan)
        for mm in range(0,12):
            print('**********************', mm)
            data_thetao_extracted_m=data_thetao_extracted[mm,:,:,:]
            data_so_extracted_m=data_so_extracted[mm,:,:,:]
            
            data_thetao_extracted_m=np.squeeze(data_thetao_extracted_m)
            data_so_extracted_m=np.squeeze(data_so_extracted_m)
            
            data_dens=sw.dens0(data_so_extracted_m, data_thetao_extracted_m-273.15)
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
            data_i=data_dens
            data_i = func_regrid(data_dens, args[0].y, args[0].x, lat, lon)
            data_i[data_i>100000]=np.nan
            
            [ii,jj] = np.where(lat>=-2000)###indeces####
    
            for k in range(len(ii)):
                if not(str(data_i[0,ii[k],jj[k]])=='nan'):
                    dummy=100
                    #MLD=0
                    interpol_dens = [data_i[depth10m_shalow,ii[k],jj[k]], data_i[depth10m_deep,ii[k],jj[k]]]
                    p_10m_dens = np.interp(10, interpol_x, interpol_dens)
                    for d in range(len(data_i)):
                        if not(str(data_i[0,ii[k],jj[k]])=='nan'):
                            p_dens = data_i[d,ii[k],jj[k]]
                            if abs(p_dens-p_10m_dens-0.125)<dummy:
                                dummy=abs(p_dens-p_10m_dens-0.125)
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
    
                    MLD_months[mm,ii[k],jj[k]]=float(interpol_z) 
            
#        MLD_Allmonths.append(MLD_months)
        
        MLD_Allmonths[12*t+1-1:12*t+12-1+1,:,:]=MLD_months
        MLD_Allmonths=np.asarray(MLD_Allmonths)

    return MLD_Allmonths
#%%
    

def func_MLD_Allmonths_bio2(SST_4D, SO_4D, Depths):
    import seawater as sw

    start=0
    end=SST_4D.shape[0]

    MLD_Allmonths=np.full([ SST_4D.shape[0],SST_4D.shape[2],SST_4D.shape[3]], np.nan)
    for t in range(int((end+1-start)/12)):
        print('MLD calc - Year: ', t)
        data_thetao_extracted=SST_4D[12*t+1-1:12*t+12-1+1,:,:,:]
        data_so_extracted=SO_4D[12*t+1-1:12*t+12-1+1,:,:,:]
        
        MLD_months=np.full([12,SST_4D.shape[2],SST_4D.shape[3]], np.nan)
        for mm in range(0,12):
            print('**********************', mm)
            data_thetao_extracted_m=data_thetao_extracted[mm,:,:,:]
            data_so_extracted_m=data_so_extracted[mm,:,:,:]
            
            data_thetao_extracted_m=np.squeeze(data_thetao_extracted_m)
            data_so_extracted_m=np.squeeze(data_so_extracted_m)
            
            data_dens=sw.dens0(data_so_extracted_m, data_thetao_extracted_m-273.15)
            depth10m_shalow=0
            depth10m_deep=0
            depth_array=np.asarray(Depths)
            
            for k in range(len(Depths)):
                if Depths[k]<=10:
                    depth10m_shalow=k
            for k in range(len(Depths)):        
                if Depths[k]>=10:
                    depth10m_deep+=k
                    break
            
            interpol_x = [Depths[depth10m_shalow], Depths[depth10m_deep]]
            data_i=data_dens
            data_i[data_i>100000]=np.nan
            
            data_dens_ave=np.nanmean(data_dens,0)
            [ii,jj] = np.where( data_dens_ave > -10000 )###indeces####
    
            for k in range(len(ii)):
                if not(str(data_i[0,ii[k],jj[k]])=='nan'):
                    dummy=100
                    #MLD=0
                    interpol_dens = [data_i[depth10m_shalow,ii[k],jj[k]], data_i[depth10m_deep,ii[k],jj[k]]]
                    p_10m_dens = np.interp(10, interpol_x, interpol_dens)
                    for d in range(len(data_i)):
                        if not(str(data_i[0,ii[k],jj[k]])=='nan'):
                            p_dens = data_i[d,ii[k],jj[k]]
                            if abs(p_dens-p_10m_dens-0.125)<dummy:
                                dummy=abs(p_dens-p_10m_dens-0.125)
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
    
                    MLD_months[mm,ii[k],jj[k]]=float(interpol_z) 
            
#        MLD_Allmonths.append(MLD_months)
        
        MLD_Allmonths[12*t+1-1:12*t+12-1+1,:,:]=MLD_months
        MLD_Allmonths=np.asarray(MLD_Allmonths)

    return MLD_Allmonths
#%%






































    