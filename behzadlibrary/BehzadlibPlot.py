
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

"""
###############################################################################
## To make the lines transparent in plots, add the following:

for l in fig.gca().lines:
    l.set_alpha(0.7)
###############################################################################
## To add a horizontal line to the plot, add the following:

l = plt.axhline(y=0, color='darkcyan')
###############################################################################  
## To add a text to the plot, add the following:

plt.text(0, -1, 'The Text', horizontalalignment='center', verticalalignment='center', fontsize=20, color='k')  
###############################################################################  
"""


def func_plotmap_contourf(P_Var, P_Lon, P_Lat, P_range, P_title, P_unit, P_cmap, P_proj, P_lon0, P_latN, P_latS, P_c_fill):

### P_Var= Plotting variable, 2D(lat,lon) || P_Lon=Longitude, 2D || P_range=range of plotted values, can be vector or number || P_title=Plot title || P_unit=Plot colorbar unit
### P_cmap= plt.cm.seismic , plt.cm.jet || P_proj= 'cyl', 'npstere', 'spstere' || P_lon0=middle longitude of plot || P_latN=upper lat bound of plot || P_latS=lower lat bound of plot || P_c_fill= 'fill' fills the continets with grey color
    
#%% Example :
    
#Plot_Var = np.nanmean(Wind_Curl,axis=0)
#cmap_limit=np.nanmax(np.abs( np.nanpercentile(Plot_Var, 99)))
#Plot_range=np.linspace(-cmap_limit,cmap_limit,101) ### Or:  Plot_range=100
#Plot_unit='(N/m3)'; Plot_title= 'Wind Curl (1E-7 N/m3) - (contour lines = Tau_x) - '+str(year_start)+'-'+str(year_end)+' - '+str(GCM)
#fig, m = func_plotmap_contourf(Plot_Var, Lon_regrid_2D, Lat_regrid_2D, Plot_range, Plot_title, Plot_unit, plt.cm.seismic, 'cyl', 210., 80., -80., 'fill')
#fig.savefig(dir_figs+'name.png', format='png', dpi=300, transparent=True, bbox_inches='tight')
#%%  
    fig=plt.figure()
    
    if P_proj=='npstere':
        m = Basemap( projection='npstere',lon_0=0, boundinglat=30)
        m.drawparallels(np.arange(-90,90,20), labels=[1,1,0,1], linewidth=1, color='k', fontsize=20)
        m.drawmeridians(np.arange(0,360,30), labels=[1,1,0,1], linewidth=1, color='k', fontsize=20)
    elif P_proj=='spstere':
        m = Basemap( projection='spstere',lon_0=180, boundinglat=-30)
        m.drawparallels(np.arange(-90,90,20), labels=[1,1,0,1], linewidth=1, color='k', fontsize=20)
        m.drawmeridians(np.arange(0,360,30), labels=[1,1,0,1], linewidth=1, color='k', fontsize=20)        
    else:
         m = Basemap( projection=P_proj, lon_0=P_lon0, llcrnrlon=P_lon0-180, llcrnrlat=P_latS, urcrnrlon=P_lon0+180, urcrnrlat=P_latN)
         m.drawparallels(np.arange(P_latS, P_latN+0.001, 40.),labels=[True,False,False,False], linewidth=0.01, color='k', fontsize=20) # labels = [left,right,top,bottom] # Latitutes
         m.drawmeridians(np.arange(P_lon0-180,P_lon0+180,60.),labels=[False,False,False,True], linewidth=0.01, color='k', fontsize=20) # labels = [left,right,top,bottom] # Longitudes        
    if P_c_fill=='fill':
        m.fillcontinents(color='0.8')
    m.drawcoastlines(linewidth=1.0, linestyle='solid', color='k', antialiased=1, ax=None, zorder=None)
    im=m.contourf(P_Lon, P_Lat, P_Var,P_range,latlon=True, cmap=P_cmap, extend='both')
    if P_proj=='npstere' or P_proj=='spstere':
        cbar = m.colorbar(im,"right", size="4%", pad="14%")
    else:
        cbar = m.colorbar(im,"right", size="3%", pad="2%")
    cbar.ax.tick_params(labelsize=20) 
    cbar.set_label(P_unit)
    plt.show()
    plt.title(P_title, fontsize=18)
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized() # Maximizes the plot window to save figures in full
    
    #m = Basemap( projection='cyl',lon_0=210., llcrnrlon=30.,llcrnrlat=-80.,urcrnrlon=390.,urcrnrlat=80.)    
    #m.drawparallels(np.arange(-90.,90.001,30.),labels=[True,False,False,False], linewidth=0.01, color='k', fontsize=20) # labels = [left,right,top,bottom] # Latitutes
    #m.drawmeridians(np.arange(0.,360.,60.),labels=[False,False,False,True], linewidth=0.01, color='k', fontsize=18) # labels = [left,right,top,bottom] # Longitudes
    #plt.close()
    
    return fig, m


def func_plot_laggedmaps(P_Var, P_Lon, P_Lat, P_peaktime, P_lag, P_lag_period, P_anomalies, P_range, P_title, P_cmap, P_proj, P_lon0, P_latN, P_latS, P_c_fill):

### P_Var= Plotting variable, 3D(time,lat,lon) || P_Lon=Longitude, 2D || P_Lat=Latitude, 2D || P_peaktime=peak time of the lagged maps || P_lag=total lag time || P_lag_period=lag periods
### P_range=range of plotted values, can be vector or number || P_anomalies= if 'yes' the maps would be annomalies with respect to average of the +/- lag time || P_cmap= plt.cm.seismic , plt.cm.jet 
### P_proj= 'cyl', 'npstere', 'spstere' || P_lon0=middle longitude of plot || P_latN=upper lat bound of plot || P_latS=lower lat bound of plot || P_c_fill= 'fill' fills the continets with grey color

#%% Example :
    
#P_Var=copy.deepcopy(Temp_allyears); P_Var=P_Var - 273.15
#P_Var=P_Var[:,0,:,:]
#P_peaktime=387  ; P_lag=40; P_lag_period=5
#P_title='Sea Surface Temperature annomalies during the WS convection at year 387 (degree C) - '+str(GCM)
#fig = func_plot_laggedmaps(P_Var, Lon_regrid_2D, Lat_regrid_2D, P_peaktime, P_lag, P_lag_period, 'yes', 100, P_title, plt.cm.seismic, 'mill', 210., 80., -80., 'fill')
#fig.savefig(dir_figs+'name.png', format='png', dpi=300, transparent=True, bbox_inches='tight')
#%%  
    
    n_t=np.int( P_lag/P_lag_period)*2 +1

    n_r=np.int( np.sqrt(n_t) ) # Number of rows
    if n_t%n_r==0: # shows the ramainder of the division
        n_c=np.int(n_t/n_r) # Number of columns
    else:
        n_c=np.int(n_t/n_r) +1 # Number of columns
    
    Var_plot_ave=np.nanmean(P_Var[ P_peaktime-P_lag:P_peaktime+P_lag,:,:], axis=0)
    
    fig=plt.figure()    
    for ii in range (0, n_t):
        
        yr=(P_lag*-1) + ii*P_lag_period
        
        if P_anomalies=='yes':
            Var_plot_ii=P_Var[P_peaktime-yr,:,:] - Var_plot_ave
        else:
            Var_plot_ii=P_Var[P_peaktime-yr,:,:]
        
        ax = fig.add_subplot(n_r,n_c,ii+1)
    
        if P_proj=='npstere' or P_proj=='spstere':
            if P_proj=='npstere':
                m = Basemap( projection='npstere',lon_0=0, boundinglat=30)
            elif P_proj=='spstere':
                m = Basemap( projection='spstere',lon_0=180, boundinglat=-30)

            if ii==0 or ii==n_c or ii==n_c*2 or ii==n_c*3 or ii==n_c*4 or ii==n_c*5 or ii==n_c*6 or ii==n_c*7 or ii==n_c*8:
                m.drawmeridians(np.arange(0,360,30), labels=[1,0,0,0], linewidth=1, color='k', fontsize=12) # labels = [left,right,top,bottom] # Longitudes
            elif ii == (n_c*(n_r-1)): # Adds longitude ranges only to the last subplots that appear at the bottom of plot
                m.drawmeridians(np.arange(0,360,30), labels=[1,0,0,1], linewidth=1, color='k', fontsize=12) # labels = [left,right,top,bottom] # Longitudes      
            elif ii >= n_t-n_c: # Adds longitude ranges only to the last subplots that appear at the bottom of plot
                m.drawmeridians(np.arange(0,360,30), labels=[0,0,0,1], linewidth=1, color='k', fontsize=12) # labels = [left,right,top,bottom] # Longitudes      
            else:
                m.drawmeridians(np.arange(0,360,30), labels=[0,0,0,0], linewidth=1, color='k', fontsize=12) # labels = [left,right,top,bottom] # Longitudes
            m.drawparallels(np.arange(-90,90,20), labels=[1,0,0,0], linewidth=1, color='k', fontsize=12) # labels = [left,right,top,bottom] # Latitutes       

        else:
            m = Basemap( projection=P_proj, lon_0=P_lon0, llcrnrlon=P_lon0-180, llcrnrlat=P_latS, urcrnrlon=P_lon0+180, urcrnrlat=P_latN)
             
            if ii == (n_c*(n_r-1)): # Adds longitude ranges only to the last subplots that appear at the bottom of plot
                m.drawparallels(np.arange(P_latS, P_latN+0.001, 30.),labels=[True,False,False,False], linewidth=0.01, color='k', fontsize=12) # labels = [left,right,top,bottom] # Latitutes
                m.drawmeridians(np.arange(0,360,90.),labels=[False,False,False,True], linewidth=0.01, color='k', fontsize=12) # labels = [left,right,top,bottom] # Longitudes    
            elif ii==0 or ii==n_c or ii==n_c*2 or ii==n_c*3 or ii==n_c*4 or ii==n_c*5 or ii==n_c*6 or ii==n_c*7 or ii==n_c*8:
                m.drawparallels(np.arange(P_latS, P_latN+0.001, 30.),labels=[True,False,False,False], linewidth=0.01, color='k', fontsize=12) # labels = [left,right,top,bottom] # Latitutes
            elif ii >= n_t-n_c and ii != (n_c*(n_r-1)): # Adds longitude ranges only to the last subplots that appear at the bottom of plot
                #m.drawparallels(np.arange(P_latS, P_latN+0.001, 30.),labels=[False,False,False,False], linewidth=0.01, color='k', fontsize=12) # labels = [left,right,top,bottom] # Latitutes
                m.drawmeridians(np.arange(0,360,90.),labels=[False,False,False,True], linewidth=0.01, color='k', fontsize=12) # labels = [left,right,top,bottom] # Longitudes
            else:
                m.drawparallels(np.arange(P_latS, P_latN+0.001, 30.),labels=[False,False,False,False], linewidth=0.01, color='k', fontsize=12) # labels = [left,right,top,bottom] # Latitutes
                m.drawmeridians(np.arange(0,360,90.),labels=[False,False,False,False], linewidth=0.01, color='k', fontsize=12) # labels = [left,right,top,bottom] # Longitudes

        if P_c_fill=='fill':
            m.fillcontinents(color='0.8')
        m.drawcoastlines(linewidth=1.0, linestyle='solid', color='k', antialiased=1, ax=None, zorder=None)
        im=m.contourf(P_Lon, P_Lat, Var_plot_ii,P_range,latlon=True, cmap=P_cmap, extend='both')
        plt.title(' Year =  '+str(yr), fontsize=12)
        
    if P_proj=='npstere' or P_proj=='spstere':
        plt.subplots_adjust(left=0.15, bottom=0.05, right=0.85, top=0.9, hspace=0.3, wspace=0.1) # the amount of height/width reserved for space between subplots
    else:
        plt.subplots_adjust(left=0.15, bottom=0.05, right=0.85, top=0.9, hspace=0.2, wspace=0.1) # the amount of height/width reserved for space between subplots
    cbar_ax = fig.add_axes([0.87, 0.1, 0.015, 0.8]) # [right,bottom,width,height] 
    fig.colorbar(im, cax=cbar_ax)
    #fig.colorbar(im, cax=cbar_ax, ticks=[-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]) # To add colorbar tick locations and values
    #cbar_ax.set_yticklabels(['- 1', '- 0.8', '- 0.6', '- 0.4', '- 0.2', '0', '0.2', '0.4', '0.6', '0.8', '1']) # To add colorbar tick locations and values
    if P_anomalies=='yes':
        plt.suptitle(P_title+'\n[Anomalies with respect to ('+str(P_peaktime)+'+/-'+str(P_lag)+'yrs) period average ]' , fontsize=18)
    else:
        plt.suptitle(P_title, fontsize=18)    
    plt.show()
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized() # Maximizes the plot window to save figures in full    

    return fig


def func_plot2Dcolor_contourf(P_Var_x1, P_Var_y1, P_Var_z1, P_range, P_xlable, P_ylable, P_title, P_unit, P_cmap, P_invert_yaxis):
### P_Var_x1= Plotting variable X-axis values, 1D(X) || P_Var_y1= Plotting variable Y-axis values, 1D(Y) || P_Var_z1= Plotting variable at contour values 2D(X,Y)
### P_xlable=lable to be shown on x-axis || P_ylable=lable to be shown on y-axis || P_range=range of plotted values, can be vector or number || P_title=Plot title || P_unit=Plot colorbar unit
### P_cmap= plt.cm.seismic , plt.cm.jet || P_invert_yaxis='invert_yaxis' inverts the y axis (lower values plotted at top)
    
#%% Example :
    
#P_Var_x1=Lon_regrid_1D
#P_Var_y1=Depths
#P_Var_z1=Density
#P_range=np.linspace(0,0.03) 
#P_xlable='Longitude'  ;P_ylable='Depth [m]';  P_unit = '(C)'
#P_title='dRho/dZ - All Years - Labrador Sea [20W-65W,50N-65N] - '+str(year_start)+'-'+str(year_end)+' - '+str(GCM)
#fig=func_plot2Dcolor_contourf(P_Var_x1, P_Var_y1, P_Var_z1, P_range, P_xlable, P_ylable, P_title, P_unit, plt.cm.jet, 'invert_yaxis')
#fig.savefig(dir_figs+'name.png', format='png', dpi=300, transparent=True, bbox_inches='tight')
#%% 
    fig=plt.figure()
    im = plt.contourf(P_Var_x1, P_Var_y1, P_Var_z1, P_range, cmap=P_cmap, extend='both')

    if P_invert_yaxis == 'invert_yaxis':
        plt.gca().invert_yaxis()
    plt.xticks(fontsize = 26); plt.yticks(fontsize = 26)
    plt.xlabel(P_xlable, fontsize=18)
    plt.ylabel(P_ylable, fontsize=18)
    plt.title(P_title, fontsize=18)
    #m = Basemap( projection='cyl', lon_0=0) # This is only added to be used for the cbar in the next line
    #cbar = m.colorbar(im,"right", size="3%", pad="2%", extend='max') # extend='both' will extend the colorbar in both sides (upper side and down side)
    cbar = plt.colorbar(im)
    cbar.set_label(P_unit)
    cbar.ax.tick_params(labelsize=18) 
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized() # Maximizes the plot window to save figures in full
        
    return fig


def func_plot2Dcontour_1var(P_Var_x1, P_Var_y1, P_Var_z1, P_xlable, P_ylable, P_title, P_color1, P_countour_no, P_invert_yaxis):
### P_Var_x1= Plotting variable X-axis values, 1D(X) || P_Var_y1= Plotting variable Y-axis values, 1D(Y) || P_Var_z1= Plotting variable at contour values 2D(X,Y)
### P_xlable=lable to be shown on x-axis || P_ylable=lable to be shown on y-axis || P_range=range of plotted values, can be vector or number || P_title=Plot title || P_color1=contour color
### P_countour_no=number of contour lines || P_invert_yaxis='yes' inverts the y axis (lower values plotted at top)
   
#%% Example :
    
#P_Var_x1=Lat_regrid_1D
#P_Var_y1=Depths
#P_Var_z1=Conv1
#P_xlable='Latitude'
#P_ylable='Depth [m]'
#P_title='Potential Density Contours (red=conv, blue=nonconv) - All Southern Ocean - '+str(year_start)+'-'+str(year_end)+' - '+str(GCM)
#fig = func_plot2Dcontour_2var(P_Var_x1, P_Var_y1, P_Var_z1, P_xlable, P_ylable, P_title, 'r', 10, 'invert_yaxis')
#fig.savefig(dir_figs+'name.png', format='png', dpi=300, transparent=True, bbox_inches='tight')
#%%
  
    fig=plt.figure()
    im1=plt.contour(P_Var_x1, P_Var_y1, P_Var_z1, P_countour_no, colors=P_color1)
    plt.clabel(im1, fontsize=8, inline=1)
    if P_invert_yaxis == 'invert_yaxis':
        plt.gca().invert_yaxis()
    plt.xticks(fontsize = 26); plt.yticks(fontsize = 26)
    plt.xlabel(P_xlable, fontsize=18)
    plt.ylabel(P_ylable, fontsize=18)
    plt.title(P_title, fontsize=18)
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized() # Maximizes the plot window to save figures in full
        
    return fig


def func_plot2Dcontour_2var(P_Var_x1, P_Var_y1, P_Var_z1, P_Var_x2, P_Var_y2, P_Var_z2, P_xlable, P_ylable, P_title, P_color1, P_color2, P_countour_no, P_invert_yaxis):
### P_Var_x1= Plotting variable X-axis values, 1D(X) || P_Var_y1= Plotting variable Y-axis values, 1D(Y) || P_Var_z1= Plotting variable at contour values 2D(X,Y)
### P_xlable=lable to be shown on x-axis || P_ylable=lable to be shown on y-axis || P_range=range of plotted values, can be vector or number || P_title=Plot title || P_color1=contour color
### P_countour_no=number of contour lines || P_invert_yaxis='yes' inverts the y axis (lower values plotted at top)

#%% Example :
    
#P_Var_x1=P_Var_x2=Lat_regrid_1D
#P_Var_y1=P_Var_y2=Depths
#P_Var_z1=Conv1
#P_Var_z2=Nonconv
#P_xlable='Latitude'
#P_ylable='Depth [m]'
#P_title='Potential Density Contours (red=conv, blue=nonconv) - All Southern Ocean - '+str(year_start)+'-'+str(year_end)+' - '+str(GCM)
#fig = func_plot2Dcontour_2var(P_Var_x1, P_Var_y1, P_Var_z1, P_Var_x2, P_Var_y2, P_Var_z2, P_xlable, P_ylable, P_title, 'r', 'b', 10, 'invert_yaxis')
#fig.savefig(dir_figs+'name.png', format='png', dpi=300, transparent=True, bbox_inches='tight')
#%%
  
    fig=plt.figure()
    im1=plt.contour(P_Var_x1, P_Var_y1, P_Var_z1, P_countour_no, colors=P_color1)
    plt.clabel(im1, fontsize=8, inline=1)
    im1=plt.contour(P_Var_x2, P_Var_y2, P_Var_z2, P_countour_no, colors=P_color2)
    plt.clabel(im1, fontsize=8, inline=1)
    if P_invert_yaxis == 'invert_yaxis':
        plt.gca().invert_yaxis()
    plt.xticks(fontsize = 26); plt.yticks(fontsize = 26)
    plt.xlabel(P_xlable, fontsize=18)
    plt.ylabel(P_ylable, fontsize=18)
    plt.title(P_title, fontsize=18)
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized() # Maximizes the plot window to save figures in full
        
    return fig


def func_plotline_1var(P_Var_x, P_Var_y, P_xlable, P_ylable, P_title, P_color, P_invert_yaxis):
### P_Var_x= Plotting variable at X-axis, 1D(X) || P_Var_y= Plotting variable at Y-axis , 1D(Y)
### P_xlable=lable to be shown on x-axis || P_ylable=lable to be shown on y-axis || P_title=Plot title || P_color=contour color || P_invert_yaxis='yes' inverts the y axis (lower values plotted at top)

#%% Example :
    
#P_Var_x=Density #or P_Var_x=years=np.linspace(year_start,year_end,year_end-year_start+1)
#P_Var_y=Depths
#P_xlable='Potential Density [kg/m3]'
#P_ylable='Depth [m]'
#P_title='Potential Density Ave - '+str(year_start)+'-'+str(year_end)+' - '+str(GCM)
#fig = func_plotline_1var(P_Var_x, P_Var_y, P_xlable, P_ylable, P_title, 'darkcyan', 'invert_yaxis')
#fig.savefig(dir_figs+'name.png', format='png'time,, dpi=300, transparent=True, bbox_inches='tight')
#%%
    fig=plt.figure()
    im1=plt.plot(P_Var_x, P_Var_y, c=P_color, linewidth=3.0)
    if P_invert_yaxis == 'invert_yaxis':
        plt.gca().invert_yaxis()
    plt.xticks(fontsize = 26); plt.yticks(fontsize = 26)
    plt.xlabel(P_xlable, fontsize=18)
    plt.ylabel(P_ylable, fontsize=18)
    plt.title(P_title, fontsize=18)
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized() # Maximizes the plot window to save figures in full
        
    return fig


def func_plotline_2var(P_Var_x1, P_Var_y1, P_Var_x2, P_Var_y2, P_xlable, P_ylable, P_title, P_color1, P_color2, P_legend1, P_legend2, P_legend_loc, P_invert_yaxis):
### P_Var_x= Plotting variable at X-axis, 1D(X) || P_Var_y= Plotting variable at Y-axis , 1D(Y) || P_xlable=lable to be shown on x-axis || P_ylable=lable to be shown on y-axis || P_title=Plot title 
### P_color1=contour color || P_legend1=legend of firts line, to be shown on the plot || P_legend_loc=location of legend, 'best' or 'lower left' or 'right' or 'center' or ...
### P_invert_yaxis='yes' inverts the y axis (lower values plotted at top)

#%% Example :
    
#P_Var_x1=Density_1
#P_Var_x2=Density_2
#P_Var_y1=P_Var_y2=Depths
#P_xlable='Potential Density [kg/m3]'
#P_ylable='Depth [m]'
#P_title='Potential Density Ave - '+str(year_start)+'-'+str(year_end)+' - '+str(GCM)
#fig = func_plotline_2var(P_Var_x1, P_Var_y1, P_Var_x2, P_Var_y2, P_xlable, P_ylable, P_title, 'r', 'b', 'Lab. Sea conv', 'Lab. Sea nonconv', 'lower left', 'invert_yaxis')
#fig.savefig(dir_figs+'name.png', format='png', dpi=300, transparent=True, bbox_inches='tight')
### P_legend_loc = 'best' or 'lower left' or 'right' or 'center' or ...
#%%
    fig=plt.figure()
    im1=plt.plot(P_Var_x1, P_Var_y1, c=P_color1, label=P_legend1, linewidth=3.0)
    im2=plt.plot(P_Var_x2, P_Var_y2, c=P_color2, label=P_legend2, linewidth=3.0)
    if P_invert_yaxis == 'invert_yaxis':
        plt.gca().invert_yaxis()
    plt.legend(prop={'size': 24}, loc=P_legend_loc, fancybox=True, framealpha=0.8)
    plt.xticks(fontsize = 26); plt.yticks(fontsize = 26)
    plt.xlabel(P_xlable, fontsize=18)
    plt.ylabel(P_ylable, fontsize=18)
    plt.title(P_title, fontsize=18)
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized() # Maximizes the plot window to save figures in full
        
    return fig

#### Compute and plot lagged regression plot####
def func_plot_lagcor_sig(P_Var_x, P_Var_y, lag, P_title, P_color, P_legend, P_legend_loc, P_v_i):
#%% Example :
    
#fig=plt.figure()
#func_plot_lagcor_sig(WS_index_surface_rm[:-9], LAB_index_rm[:-9], 40, P_title, 'y', 'LAB index (Temp at 500m depth)', 'best', '-')
#func_plot_lagcor_sig(WS_index_surface_rm[:-9], Winds_40S60S_0W30W_m[:-9], 40, P_title, 'g', 'Zonal Winds [40S60S, 0W30W]', 'best', '-')
#func_plot_lagcor_sig(WS_index_surface_rm[:-9], AMOC_max_50S_m[:-9], 40, P_title, 'b', 'AMOC max 50S', 'best', '-')
#func_plot_lagcor_sig(WS_index_surface_rm[:-9], AMOC_max_30S_m[:-9], 40, P_title, 'r', 'AMOC max 30S', 'best', '-')
#plt.ylim(-0.7,0.7)
#fig.savefig(dir_figs+'name.png', format='png', dpi=300, transparent=True, bbox_inches='tight')
###  P_v_i='yes' shows the P-value plot of the corrolation
### P_legend_loc = 'best' or 'lower left' or 'right' or 'center' or ...
#%%    
    R_val=[]
    P_val=[]
    from scipy import stats
    for i in range(2*lag):
        slope, intercept, r_value, p_value, std_err = stats.linregress(P_Var_x[lag:len(P_Var_x)-lag], P_Var_y[i:len(P_Var_y)-2*lag+i])
        R_val.append(r_value)
        P_val.append(p_value)
    xx=np.linspace(-lag,lag+1, 2*lag)
    plt.grid(True,which="both",ls="-", color='0.65')
    plt.plot(xx, R_val, P_color, label=P_legend, linewidth=3.0)
    if P_v_i=='yes':
        plt.plot(xx, P_val, P_color, ls='--')#, label='Significance (P-value)')
    plt.xlabel('Years lag', fontsize=18)
    plt.ylabel('Correlation coefficient (R)', fontsize=18)
    plt.title(P_title, fontsize=18)
    plt.xticks(fontsize = 20); plt.yticks(fontsize = 24)
    plt.legend()
    #plt.show()
    plt.legend(prop={'size': 15}, loc=P_legend_loc, fancybox=True, framealpha=0.8)
    l = plt.axhline(y=0, color='k')
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized() # Maximizes the plot window to save figures in full

    #return fig


#### Compute and plot PSD using welch method####
def plot_PSD_welch_conf(P_Var, P_fq, P_c_probability, P_rho, P_smooth, P_title, P_color, P_legend, P_legend_loc):

### P_c_probability= 0.95 or 0.05 or '-' = confidence interval levels, if the input is not a number then teh confidence intervals won't be plptted
### P_rho = 'yes' or 0.7 or '-' = Red noise significance line, if 'yes' then the Rho and line will be calculated, if P_rho is a number then it will be given as the Rho value, else NO Red noise significance line will be plotted
### P_smooth = 9 (should be even number) or '-' = Number of years for smoothing the PSD line and confidence intervals, if no number is given then NO smoothing will be applied
### P_Var= Plotting variable at X-axis, 1D(X) || P_title=Plot title || P_color=Plot line color || P_legend=Plot variable name to be shown in Plot Legend 
### P_fq=PSD Welch method's sampling frequency of the x time series in units of Hz (Defaults to 1.0) || P_legend_loc=location of legend, 'best' or 'lower left' or 'right' or 'center' or ...
#%% Example :

#P_Var=WS_index ; P_legend='WS Convection Index' ; P_color='r'
#P_title='Power Spectral Density - WS Convection Index with 95% confidence intervals - '+str(year_start)+'-'+str(year_end)+' - '+str(GCM)
#fig=plt.figure()
#ff,PSD=plot_PSD_welch_conf(P_Var, 1, 0.95, 'yes', 9, P_title, P_color, P_legend, 'lower right')
#plt.ylim(1e-5,1e3) #;plt.xlim(1,500)
#fig.savefig(dir_figs+'name.png', format='png', dpi=300, transparent=True, bbox_inches='tight') 
### 

    from scipy.stats import chi2    
    from scipy import signal

    ff,PSD = signal.welch(P_Var,P_fq) # ff= Array of sample frequencies  ,  PSD = Pxx, Power spectral density or power spectrum of x (which is P_Var)
    X_var=np.linspace(1,len(P_Var)/2+1,len(P_Var)/2+1)
    X_var=ff**(-1); X_var=X_var
    
    if P_rho=='yes':
        Rho = np.nansum( (P_Var[:-1] - np.nanmean(P_Var,axis=0)) * (P_Var[1:] - np.nanmean(P_Var,axis=0)) ,axis=0) / np.nansum( (P_Var[:-1] - np.nanmean(P_Var,axis=0)) * (P_Var[:-1] - np.nanmean(P_Var,axis=0)) ,axis=0)
    elif type(P_rho)==float:
        Rho=P_rho # Rho is the memory parameter
    
    if type(P_smooth)==float or type(P_smooth)==int:
        P_smooth=np.int(P_smooth)
        v = 2*P_smooth # P is the number of estimates in welch function and also the degree of freedom.
        
        sm= np.int( (P_smooth-1)/2)
        P_smooth = np.int( (sm*2)+1 ) # P_smoothhs to be even number for centered smoothing; in case odd number was given, it's changed to an even number by subtracting 1
        PSD_m=copy.deepcopy(PSD) ## Smoothing the variable
        for ii in range(sm,PSD.shape[0]-sm+1):
            PSD_m[ii]=np.nanmean(PSD[ii-sm:ii+sm])
        
        PSD_m=PSD_m[sm:-sm]
        X_var_m=X_var[sm:-sm]
        P_legend=P_legend+' ('+str(np.int(P_smooth))+'yr smoothed)'
        
    else:
        v=2

        PSD_m=copy.deepcopy(PSD)
        X_var_m=copy.deepcopy(X_var)
        
    if type(P_c_probability)==float:
        if P_c_probability < 0.5:
            P_c_probability=1-P_c_probability # In case the P_c_probability is input 0.05 instead of 0.95 for example
        alfa = 1 - P_c_probability

    if P_rho=='yes' or type(P_rho)==float or type(P_rho)==int:
        if type(P_c_probability)!=float: # In case the P_c_probability is not given since confidence interval calculation is not necessary, but red noise significance line is needed
            alfa=0.05
            
        F_x_v = (1-Rho**2) / (1 + Rho**2 - 2*Rho*np.cos(2*np.pi*ff ) )  #  F_x_v is the power spectraum   
        F_x_v_star=np.float( np.real( np.nanmean(PSD,axis=0) / np.nanmean(F_x_v,axis=0) ) ) * F_x_v 
        Pr_alpha = (1/v) * F_x_v_star * np.float( chi2.ppf([1 - alfa], v) )
    
    plt.grid(True,which="both",ls="-", color='0.65')
    plt.loglog(X_var_m,PSD_m, color=P_color, label=P_legend)
    plt.legend(loc='best')
    plt.xlabel('Period (years)', fontsize=18)
    plt.ylabel('Spectral Density', fontsize=18) 
    plt.xticks(fontsize = 20); plt.yticks(fontsize = 20)
    #plt.gca().invert_xaxis()
    if type(P_c_probability)==float:
        Chi = chi2.ppf([1 - alfa / 2, alfa / 2], v)
        
        PSDc_lower = PSD_m * ( v / Chi[0] )
        PSDc_upper = PSD_m * ( v / Chi[1] ) 
        plt.loglog(X_var_m,PSDc_lower, color='g', ls='--', label=str(np.int( (1 - alfa) *100))+'% confidence intervals')
        plt.loglog(X_var_m,PSDc_upper, color='g', ls='--')
    if P_rho=='yes' or type(P_rho)==float or type(P_rho)==int:
        plt.loglog(X_var,Pr_alpha , color='b', ls='--', label=str(np.int( (1 - alfa) *100))+'% Red Noise Significance Level')
    plt.legend(prop={'size': 20}, loc=P_legend_loc, fancybox=True, framealpha=0.8)
    plt.title(P_title, fontsize=18) 
    plt.show()
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized() # Maximizes the plot window to save figures in full
    return ff,PSD























































    
    
    
    