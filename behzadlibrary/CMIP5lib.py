from netCDF4 import MFDataset, Dataset
import numpy as np
from netCDF4 import num2date, date2num
import matplotlib.mlab as ml
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, addcyclic
import numpy.ma as ma
from scipy.interpolate import griddata
import seawater as sw
#readinf netcdf variable, lon,lat, depth and time
class netcdf_read:
    def __init__(self,path,GCM,variable):
        try:
            self.fin = MFDataset(path, GCM)
            print ('Opened ', GCM)
            self.fin = MFDataset(path, GCM)
            self.frequency=self.fin.frequency
        except:
            print ('There is no such', GCM)
        if self.frequency=='yr':
            print (self.frequency)
            self.x=np.asarray(self.fin.variables['lon'][:])
            if np.min(self.x)<0:
                # converts lons to 0 to 360 values
                self.x[self.x<0]+=360
            self.y=np.asarray(self.fin.variables['lat'][:])
            self.times=self.fin.variables['time']
            self.lvls = self.fin.variables['lev_bnds']
            self.lvl = self.fin.variables['lev']
            self.variable =self.fin.variables[str(variable)]
            #self.missing_value=self.fin.variables[str(variable)].missing_value
            print ('DONE LOADING', GCM)
        if self.frequency=='mon':
            print (self.frequency)
            self.x=np.asarray(self.fin.variables['lon'][:])
            if np.min(self.x)<0:
                # converts lons to 0 to 360 values
                self.x[self.x<0]+=360
            self.y=np.asarray(self.fin.variables['lat'][:])
            self.times=self.fin.variables['time']
            self.variable =self.fin.variables[str(variable)]
            #self.missing_value=self.fin.variables[str(variable)].missing_value
            self.frequency=self.fin.frequency
            try:
                self.lvls = self.fin.variables['lev_bnds']
                self.lvl = self.fin.variables['lev']
            except:
                print ("monthly data with no layers")
            print ('DONE LOADING', GCM)
    def find_time(self, times, year_start, year_end):
        start_date=-999
        end_date=-999
        if self.frequency=='yr':
            for i in range(len(times[:])):
                
                date=str(num2date(times[i],units=times.units,calendar=times.calendar))
                words = date.split('-')
                if str(int(words[0]))==str(year_start):
                    start_date=i
                    print ('start date',\
                    str(num2date(times[i],units=times.units,calendar=times.calendar)))
                if str(int(words[0]))==str(year_end):
                    end_date=i
                    print ('end date',\
                    str(num2date(times[i],units=times.units,calendar=times.calendar)))
        if self.frequency=='mon':
             for i in range(len(times[:])):
                date=str(num2date(times[i],units=times.units,calendar=times.calendar))
                words = date.split('-')
                if str(int(words[0]))+'-'+words[1]==str(year_start)+'-01':
                    start_date=i
                    print ('start date',\
                    str(num2date(times[i],units=times.units,calendar=times.calendar)))
                if str(int(words[0]))+'-'+words[1]==str(year_end)+'-12':
                    end_date=i
                    print ('end date',\
                    str(num2date(times[i],units=times.units,calendar=times.calendar)))
        if start_date==-999:
            print ('start date not found')
        if end_date==-999:
            print ('start date not found, calculating end date')
            if self.frequency=='yr':
                end_date=start_date+year_end-year_start+1
            if self.frequency=='yr':
                end_date=start_date+(year_end-year_start+1)*12
            
        return start_date,end_date

    def find_depth(self, lev, depth_start, depth_end):

        depth_index_start=0
        depth_index_end=0
        if depth_start==0:
            depth_index_start=0
        if depth_end==0:
            depth_index_end=1
        else:
            for i in range(len(lev[:])):
                if lev[i]<=depth_start:
                    depth_index_start=i
            for i in range(len(lev[:])):
                if lev[i]<=depth_end:
                    depth_index_end=i
        if depth_index_end==0:
            depth_index_end+=1
        return depth_index_start, depth_index_end
    
    #order of args in function -  data, start, end, depth_start, depth_end
    def extract_data(self, *args):
        if self.frequency=='yr':
            data_total=[]
            for i in range(args[2]-args[1]+1):
                if len(args)==3:
                    data_total.append(np.asarray(args[0][args[1]+i:args[1]+i+1]))
                else:
                    data_total.append(np.asarray(args[0][args[1]+i:args[1]+i+1,\
                                      args[3]:args[4]]))
            data_total=np.concatenate((data_total[:]))
        if self.frequency=='mon':
            data_total=[]
            if args[0].ndim==4:
                for i in range(args[2]-args[1]+1):
                    if len(args)==3:
                        data_total.append(np.asarray(args[0][args[1]+i:args[1]+i+1]))
                    else:
                        data_total.append(np.asarray(args[0][args[1]+i:args[1]+i+1,\
                                          args[3]:args[4]]))
            else:
                for i in range(args[2]-args[1]+1):
                    data_total.append(np.asarray(args[0][args[1]+i:args[1]+i+1]))
            #data_total=np.asarray(data_total)
            #print (data_total.shape)
            data_total=np.concatenate((data_total[:]))
        return data_total
    def extract_depth(self, *args):
        depths=self.fin.variables['lev'][:]
        if len(depths)==0:
            depths=0
        return depths
    def extract_depth_bounds(self, *args):
        depths_b=self.fin.variables['lev_bnds'][:]
        if len(depths_b)==0:
            depths_b=0
        return depths_b
            
    def close_ncfile(self, fin):
        fin.close()

def interpolate_2_reg_grid(lon, lat, data):
    ny, nx = 181, 360
    xmin, xmax = 0, 359
    ymin, ymax = -90, 90
    xi = np.linspace(xmin, xmax, nx)
    yi = np.linspace(ymin, ymax, ny)
    x_i, y_i = np.meshgrid(xi, yi)
    if np.ndim(lon)==1:
        lon,lat=np.meshgrid(lon, lat)
        
    lon_i = np.asarray(lon)
    lat_i = np.asarray(lat)
    lon_i = lon_i.flatten()
    lat_i = lat_i.flatten()
    coords=np.squeeze(np.dstack((lon_i,lat_i)))

    data=np.squeeze(data)
    if data.ndim==2:#this is for 2d regridding
        z = np.asarray(data)
        z = z.flatten()
        zi = griddata(coords, z, (x_i, y_i), method='nearest')
        return x_i, y_i, zi
    if data.ndim==3:#this is for 3d regridding
        data_3d=[]
        for d in range(len(data)):
            z = np.asarray(data[d,:,:])
            z = z.flatten()
            zi = griddata(coords, z, (x_i, y_i), method='nearest')
            data_3d.append(zi)
        return x_i, y_i, np.asarray(data_3d)


def interpolate_2_custom_grid(lon, lat, data, new_lon, new_lat):
    nx, ny = new_lon, new_lat
    xmin, xmax = 0, 359
    ymin, ymax = -90, 90
    xi = np.linspace(xmin, xmax, nx)
    yi = np.linspace(ymin, ymax, ny)
    x_i, y_i = np.meshgrid(xi, yi)
    if np.ndim(lon)==1:
        lon,lat=np.meshgrid(lon, lat)
        
    lon_i = np.asarray(lon)
    lat_i = np.asarray(lat)
    lon_i = lon_i.flatten()
    lat_i = lat_i.flatten()
    coords=np.squeeze(np.dstack((lon_i,lat_i)))

    data=np.squeeze(data)
    if data.ndim==2:#this is for 2d regridding
        z = np.asarray(data)
        if np.ndim(z)>1:
            z = z.flatten()
        zi = griddata(coords, z, (x_i, y_i), method='nearest')
        return x_i, y_i, zi
    if data.ndim==3:#this is for 3d regridding
        data_3d=[]
        for d in range(len(data)):
            z = np.asarray(data[d,:,:])
            if np.ndim(z)>1:
                z = z.flatten()
            zi = griddata(coords, z, (x_i, y_i), method='nearest')
            data_3d.append(zi)
        return x_i, y_i, np.asarray(data_3d)

#####################################################################
################### PLOTTING AND ANALYZING DATA #####################
#####################################################################
    
def plot_global_map(lon, lat, data, GCM, variable, year_start, year_end):            
    fig = plt.figure()
    ax = fig.add_axes([0.1,0.1,0.85,0.85])
    m = Basemap(projection='mill',lon_0=180)
    m.drawcoastlines(linewidth=1.25)
    m.fillcontinents(color='0.8')
    m.drawmapboundary(fill_color='0.9')
    zi_mask=ma.masked_where(data >= 999, data)
    im1 = m.pcolormesh(lon,lat,zi_mask,shading='flat',cmap=plt.cm.jet,latlon=True)
    m.drawparallels(np.arange(-90.,99.,30.))
    m.drawmeridians(np.arange(0.,360.,60.))
    cb = m.colorbar(im1,"bottom", size="5%", pad="2%")
    plt.clim(np.min(zi_mask),np.max(zi_mask))
    ax.set_title(str(variable)+' '+str(GCM)+' '+str(year_start)+'-'+str(year_end))
    #plt.show()

#### running mean calculation ####
def runningMeanFast(x, N):
    return np.convolve(x, np.ones((N,))/N)[(N-1):]

#### Compute and plot PSD using welch method####
def plot_PSD_welch(y, fq, c,l):
    from scipy import signal
    plt.grid(True,which="both",ls="-", color='0.65')
    f,PSD = signal.welch(y,fq)
    x=np.linspace(1,len(y)/2+1,len(y)/2+1)
    x=f**(-1)
    x=x
    plt.loglog(x,PSD, color=c, label=l)
    plt.legend(loc='best')
    plt.xlabel('Period (years)')
    plt.ylabel('Spectral Density')
    #plt.gca().invert_xaxis()
    #plt.show()
    return x,PSD

#### Compute and plot lagged regression plot####
def lag_cor(x,y,lag, c, l):
    stat=[]
    from scipy import stats
    for i in range(2*lag):
        slope, intercept, r_value, p_value, std_err = stats.linregress(x[lag:len(x)-lag], y[i:len(y)-2*lag+i])
        stat.append(r_value)
    x=np.linspace(-lag,lag+1, 2*lag)
    plt.grid(True,which="both",ls="-", color='0.65')
    plt.plot(x, stat, c, label=l, linewidth=3.0)
    plt.xlabel('Year lag')
    plt.ylabel('r')

def lag_cor_data(x,y,lag):
    stat=[]
    from scipy import stats
    for i in range(2*lag):
        slope, intercept, r_value, p_value, std_err = stats.linregress(x[lag:len(x)-lag], y[i:len(y)-2*lag+i])
        stat.append(r_value)
    return stat


def calc_Atl_Mask():
    ny, nx = 181, 360
    xmin, xmax = 0, 359
    ymin, ymax = -90, 90
    xi = np.linspace(xmin, xmax, nx)
    yi = np.linspace(ymin, ymax, ny)
    lon, lat = np.meshgrid(xi, yi)
####     This is Atlantic Mask
####     I know this is an ugly monster,
####     the thing the np.logical_not can not take more than 3 arguments,
####     and 4 needed, so I created this....
    [i,j] = np.where(np.logical_or(
                       np.logical_or((lat<=12) & (lon>=0) & (20>=lon),(lat<=12) & (290<=lon)& (lon<=360)),#### SH indeces
                       np.logical_or((lat>12) & (lon>=0) & (20>=lon),(lat>12) & (270<=lon) & (lon<=360))))#### NH indeces####
    return [i,j]

def calc_Atl_SO_Mask():
    ny, nx = 181, 360
    xmin, xmax = 0, 359
    ymin, ymax = -90, 90
    xi = np.linspace(xmin, xmax, nx)
    yi = np.linspace(ymin, ymax, ny)
    lon, lat = np.meshgrid(xi, yi)
####     This is Atlantic_SO Mask
####     I know this is an ugly monster,
####     the thing is that the np.logical_or can not take more than 3 arguments,
####     and 4 needed, so I created this....
    [i,j] = np.where(np.logical_or(
                       (lat<-32),
                       np.logical_or((-32<=lat<=12) & (lon>=0) & (20>=lon),(lat<=12) & (290<=lon)& (lon<=360)),#### SH indeces
                       np.logical_or((lat>12) & (lon>=0) & (20>=lon),(lat>12) & (270<=lon) & (lon<=360))))#### NH indeces####
    return [i,j]

def calc_stream(*args):
    ##### order of args :
    ##### 1) data (netcdf_read object from CMIP5lib.py),
    ##### 2) start year,
    ##### 3) end year,
    ##### 4) Mask,
    ##### 5) plot option to check the depth and mask  = 1/0
    depths=args[0].extract_depth()
    ### these are upper and lower depths of each cell in an ocean grid
    depths_b=args[0].extract_depth_bounds()
    ### calculate the depth of each cell in an ocean grid
    depths_r=depths_b[:,1]-depths_b[:,0]
    ### find timeindeces
    start,end = args[0].find_time(args[0].times, args[1], args[2])
    stream_final=[]
    transport_final=[]
    transport_0_1000=[]
    transport_2000_3000=[]

    ### we averaging velocities over the year for monthly data
    for i in range(int((end+1-start)/12)):
        print (i, start+12*i, start+12*i+12)
        data_extracted=args[0].extract_data(args[0].variable,start+12*i,start+12*i+11)
        data=np.squeeze(data_extracted)
        data=np.mean(data, axis=0)
        print (data.shape)
        lon,lat,data_i=interpolate_2_reg_grid(args[0].x,args[0].y,data)
        ##### converting 1e+20 to nan ######
        data_i[data_i>1000]=np.nan
        if i==0: 
            data_depth=np.full([len(lon),len(lon[0])], np.nan)
            data_depth_ranges=np.full([len(data_i),len(data_i[0]),len(data_i[0][0])], np.nan)
        [ii,jj] = args[3]
        for k in range(len(ii)):
            #### i calculate he depth by looking how many nans is in the depth column
            if sum(~np.isnan(data_i[:,ii[k],jj[k]]))>0:
                if i==0:
                    data_depth[ii[k],jj[k]]=depths[sum(~np.isnan(data_i[:,ii[k],jj[k]]))-1]
                    for l in range(sum(~np.isnan(data_i[:,ii[k],jj[k]]))):
                        data_depth_ranges[l,ii[k],jj[k]]=depths_r[l]
                        
        #### calculating volume transport
        #### first multiplying by 111km*cos(lat)
        mul_by_lat=data_i*(np.cos(np.deg2rad(lat))*111000)
        #### second multiplying by depth
        transport=mul_by_lat*data_depth_ranges/1000000
        #### calculating integral over dz
        transport_lon=np.nansum(transport,axis=2)
        #### calculating cum integral over dz
        #stream=np.nancumsum(transport_lon,axis=0)
        stream_final.append(transport_lon)
        print(transport.shape)
        transport_0_1000.append(np.nanmean(transport[0:34,:,:],axis=0))
        transport_2000_3000.append(np.nanmean(transport[39:43,:,:],axis=0))

        

        
        
        #transport_final.append(transport)
    stream_final=np.asarray(stream_final)
    transport_0_1000=np.asarray(transport_0_1000)
    transport_2000_3000=np.asarray(transport_2000_3000)


    transport_4d_mean_0_1000=np.nanmean(transport_0_1000,axis=0)
    transport_4d_mean_2000_3000=np.nanmean(transport_2000_3000,axis=0)

    maxvals=np.nanmax(transport_4d_mean_0_1000,axis=1)
    minvals=np.nanmin(transport_4d_mean_2000_3000,axis=1)

    ind_max = np.array([np.argwhere(transport_4d_mean_0_1000 == [x]) for x in maxvals])
    ind_max=np.concatenate(ind_max).astype(None)
    ind_max=np.concatenate(ind_max).astype(None)
    ii_max=ind_max[0::2].astype(int)
    jj_max=ind_max[1::2].astype(int)
    ind_min = np.array([np.argwhere(transport_4d_mean_2000_3000 == [x]) for x in minvals])
    ind_min=np.concatenate(ind_min).astype(None)
    ind_min=np.concatenate(ind_min).astype(None)
    ii_min=ind_min[0::2].astype(int)
    jj_min=ind_min[1::2].astype(int)

    if args[4]==1:
        m = Basemap( projection='mill',lon_0=180)
        m.fillcontinents(color='0.8')
        m.drawmapboundary(fill_color='0.9')
        m.drawmapboundary(fill_color='#000099')
        m.drawparallels(np.arange(-90,90,20), labels=[1,1,0,1])
        m.drawmeridians(np.arange(0,360,30), labels=[1,1,0,1])
        im=m.contourf(lon,lat,data_depth,200,latlon=True, cmap=plt.cm.jet)
        cb = m.colorbar(im)
        plot_global_map(lon, lat, transport_4d_mean_0_1000, 'GCM ', 'mean transport upper 1000m', args[1], args[2])
        plot_global_map(lon, lat, transport_4d_mean_2000_3000, 'GCM ', 'mean transport 2000m-3000m', args[1], args[2])
        plt.show()
        return stream_final, transport_0_1000,transport_2000_3000, ii_max,jj_max, ii_min, jj_min
    else:
        return stream_final, transport_0_1000,transport_2000_3000, ii_max,jj_max, ii_min, jj_min

def calc_ENSO(*args):
    ny, nx = 181, 360
    xmin, xmax = 0, 359
    ymin, ymax = -90, 90
    xi = np.linspace(xmin, xmax, nx)
    yi = np.linspace(ymin, ymax, ny)
    lon, lat = np.meshgrid(xi, yi)
    # ENSO index based on 190 220 -5 5 rectangele
    ##### order of args :
    ##### 1) data (netcdf_read object from CMIP5lib.py),
    ##### 2) start year,
    ##### 3) end year,
    ##### 4) plot option to check the depth and mask  = 1/0
    start,end = args[0].find_time(args[0].times, args[1], args[2])
    [ii,jj] = np.where(np.logical_and((lat<=5) &(lon>=190), (220>=lon) &(lat>=-5)))### NH indeces####
    ENSO=[]
    for i in range(int((end+1-start)/12)):
        
        print (i, start+12*i, start+12*i+12)
        data_extracted=args[0].extract_data(args[0].variable,start+12*i,start+12*i+11, 0, 1)
        print (data_extracted.shape)
        data=np.squeeze(data_extracted)
        data=np.mean(data, axis=0)
        print (data.shape)
        lon,lat,data_i=interpolate_2_reg_grid(args[0].x,args[0].y,data)
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
    if args[3]==1:
        m = Basemap( projection='mill',lon_0=180)
        m.fillcontinents(color='0.8')
        m.drawmapboundary(fill_color='0.9')
        m.drawmapboundary(fill_color='#000099')
        m.drawparallels(np.arange(-90,90,20), labels=[1,1,0,1])
        m.drawmeridians(np.arange(0,360,30), labels=[1,1,0,1])
        plt.title('Test of the region')
        mask=ma.masked_where(data_plot == np.nan, data_plot)
        im=m.contourf(lon,lat,mask,200,latlon=True, cmap=plt.cm.jet)
        cb = m.colorbar(im)
        
        fig=plt.figure()
        years=np.linspace(args[1], args[2], args[2]-args[1]+1)
        plt.plot(years,ENSO, 'k') 
        y2=np.zeros(len(ENSO))
        plt.fill_between(years, ENSO, y2, where=ENSO >= y2, color = 'r', interpolate=True)
        plt.fill_between(years, ENSO, y2, where=ENSO <= y2, color = 'b', interpolate=True)
        plt.axhline(linewidth=1, color='k')
        plt.show()
        return ENSO
    else:
        return ENSO

def calc_NAO(*args):
    ##### NAO index based on 80W 30E 10N 85N rectangele
    ##### order of args :
    ##### 1) data (netcdf_read object from CMIP5lib.py),
    ##### 2) start year,
    ##### 3) end year,
    ##### 4) plot option to check the depth and mask  = 1/0
    NAO=[]
    start,end = args[0].find_time(args[0].times, args[1], args[2])
    for i in range(int((end+1-start)/12)):
        print (i, start+12*i, start+12*i+12)
        data_extracted=args[0].extract_data(args[0].variable,start+12*i,start+12*i+11, 0, 1)
        print (data_extracted.shape)
        data=np.squeeze(data_extracted)
        data=np.mean(data, axis=0)
        print (data.shape, args[0].x.shape, args[0].y.shape)
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
    if args[3]==1:
        m = Basemap( projection='mill',lon_0=0)
        m.fillcontinents(color='0.8')
        m.drawmapboundary(fill_color='0.9')
        m.drawmapboundary(fill_color='#000099')
        m.drawparallels(np.arange(-90,90,20), labels=[1,1,0,1])
        m.drawmeridians(np.arange(0,360,30), labels=[1,1,0,1])
        cmap_limit=np.nanmax(np.abs(spatial_pattern_NAO))
        levels=np.linspace(-cmap_limit,cmap_limit,200)
        plt.title('NAO 1st EOF of SLP')
        mask=ma.masked_where(spatial_pattern_NAO == np.nan, spatial_pattern_NAO)
        lon[lon > 180] -= 360
        im=m.contourf(lon,lat,mask,levels,latlon=True, cmap=plt.cm.seismic)
        cb = m.colorbar(im)
        
        fig=plt.figure()
        years=np.linspace(args[1], args[2], args[2]-args[1]+1)
        
        plt.plot(years,time_series_NAO, 'k') 
        y2=np.zeros(len(time_series_NAO))
        plt.fill_between(years, time_series_NAO, y2, where=time_series_NAO >= y2, color = 'r', interpolate=True)
        plt.fill_between(years, time_series_NAO, y2, where=time_series_NAO <= y2, color = 'b', interpolate=True)
        plt.axhline(linewidth=1, color='k')
        plt.show()
        return time_series_NAO
    else:
        return time_series_NAO

def calc_EOF(*args):
    ##### NAO index based on 80W 30E 10N 85N rectangele
    ##### order of args :
    ##### 1) data (netcdf_read object from CMIP5lib.py),
    ##### 2) start year,
    ##### 3) end year,
    ##### 4) [lon_min, lon_max, lat_min, lat_max] 
    ##### 5) plot option to check the depth and mask  = 1/0
    ##### 6) projection
    ##### 7) depth_min
    ##### 8) depth_max

    EOF=[]
    start,end = args[0].find_time(args[0].times, args[1], args[2])
    for i in range(int((end+1-start)/12)):
        print (i, start+12*i, start+12*i+12)
        
        if len(args)==6:
            depth_start,depth_end=args[0].find_depth(args[0].lvl,args[6],args[6])
            data_extracted=args[0].extract_data(args[0].variable,start+12*i,start+12*i+11, depth_start, depth_end)
        if len(args)==7:
            if np.ndim(args[0].x)==1:
                data_depth=np.full([len(args[0].x),len(args[0].y)], np.nan)
            depth_start,depth_end=args[0].find_depth(args[0].lvl,args[6],args[7])
            data_extracted=args[0].extract_data(args[0].variable,start+12*i,start+12*i+11, depth_start, depth_end)
            depths_b=args[0].extract_depth_bounds()
            ### calculate the depth of each cell in an ocean grid
            depths_r=depths_b[:,1]-depths_b[:,0]
        else:    
            data_extracted=args[0].extract_data(args[0].variable,start+12*i,start+12*i+11, 0, 1)
        print (data_extracted.shape)
        data=np.squeeze(data_extracted)
        data=np.mean(data, axis=0)
        print (data.shape, args[0].x.shape, args[0].y.shape)
        lon,lat,data_i=interpolate_2_custom_grid(args[0].x,args[0].y,data, 180, 91)
        data_i[data_i>100000]=np.nan
        data_EOF=[]
        if i==0:
            

            
            if args[3][0]>args[3][1]:
                [ii,jj] = np.where(np.logical_or((lat<=args[3][3]) &(lon>=args[3][0])&(lat>=args[3][2]), (args[3][1]>=lon)&(lat<=args[3][3])&(lat>=args[3][2]) ))### NH indeces####
            else:
                [ii,jj] = np.where(np.logical_and((lat<=args[3][3]) &(lon>=args[3][0])&(lat>=args[3][2]), (args[3][1]>=lon)&(lat<=args[3][3])&(lat>=args[3][2]) ))### NH indeces####

            if len(args)==7: 
                data_depth=np.full([len(lon),len(lon[0])], np.nan)
                data_depth_ranges=np.full([len(data_i),len(data_i[0]),len(data_i[0][0])], np.nan)
                for k in range(len(ii)):
                    #### i calculate he depth by looking how many nans is in the depth column
                    if sum(~np.isnan(data_i[:,ii[k],jj[k]]))>0:
                        data_depth[ii[k],jj[k]]=depths[sum(~np.isnan(data_i[:,ii[k],jj[k]]))-1]
                        for l in range(sum(~np.isnan(data_i[:,ii[k],jj[k]]))):
                            data_depth_ranges[l,ii[k],jj[k]]=depths_r[l]
                data_i=np.sum(data_i*data_depth_ranges[depth_start:depth_end], axis=0)/np.nansum(depths_r[depth_start:depth_end])
            
            lat_f=[];
            lon_f=[];
            nan_indeces=[];
            data_plot=np.full([len(data_i),len(data_i[0])], np.nan)
            #data_EOF=np.full([int((end+1-start)/12),len(data_i),len(data_i[0])], np.nan)
        for k in range(len(ii)):
            #data_EOF[i,ii[k],jj[k]]=data_i[ii[k],jj[k]]*np.sqrt(np.cos(np.deg2rad(lat[ii[k],jj[k]])))
            EOF_i=data_i[ii[k],jj[k]]*np.sqrt(np.cos(np.deg2rad(lat[ii[k],jj[k]])))
            data_EOF.append(EOF_i)
            if i==0:
                data_plot[ii[k],jj[k]]=data_i[ii[k],jj[k]]
                lat_f.append(lat[ii[k],jj[k]])
                lon_f.append(lon[ii[k],jj[k]])
                nan_indeces.append(data_i[ii[k],jj[k]])

        EOF.append(data_EOF)
    nan_indeces=np.asarray(nan_indeces)
    print(int((end+1-start)/12),len(EOF))
    EOF=np.asarray(EOF)
    lat_f=np.asarray(lat_f)
    lon_f=np.asarray(lon_f)
    lon_f = lon_f[~np.isnan(nan_indeces)]
    lat_f = lat_f[~np.isnan(nan_indeces)]
    EOF = EOF[~np.isnan(EOF)]
    print(int((end+1-start)/12),len(EOF))
    EOF=np.reshape(EOF,(int((end+1-start)/12),int(len(EOF)/int((end+1-start)/12)))) 
    print (EOF.shape,lon_f.shape)
    print (EOF[0])
    C=np.cov(np.transpose(EOF))
    C=np.array(C, dtype=np.float32)
    eigval,eigvec=np.linalg.eig(C)
    
##    lat=np.unique(lat_f)
##    lon=np.unique(lon_f)
    print(len(lat_f),len(lon_f))
##    lon, lat = np.meshgrid(lon, lat)
##    lon, lat = np.meshgrid(lon_f, lat_f)
##    lon, lat =lon_f, lat_f
    if args[4]==1:
        fig, ax = plt.subplots(nrows=2, ncols=2)
        for i in range(4):
            #spatial_pattern_EOF=np.reshape(eigvec[:,i],(len(lat),len(lat[0])))
            spatial_pattern_EOF=np.asarray(eigvec[:,i])
            print(np.isnan(lon_f.any()),np.isnan(lat_f.any()),np.isnan(spatial_pattern_EOF.any()))
            print(lon_f,lat_f,spatial_pattern_EOF,args[0].x.shape)
            
            xmin, xmax = 0, 359
            ymin, ymax = -90, 90
            xi = np.linspace(xmin, xmax, 180)
            yi = np.linspace(ymin, ymax, 91)
            x_i, y_i = np.meshgrid(xi, yi)
            coords=np.squeeze(np.dstack((lon_f,lat_f)))
            zi = griddata(coords, spatial_pattern_EOF, (x_i, y_i), method='nearest')
            
            #lons,lats,spatial_pattern_EOFs=interpolate_2_custom_grid(lon_f[:],lat_f[:],spatial_pattern_EOF[:], 180, 91)
            
            plt.subplot(2, 2, i+1)
            cmap_limit=np.nanmax(np.abs(spatial_pattern_EOF))
            levels=np.linspace(-cmap_limit,cmap_limit,200)
            if args[5]=='spstere' or 'npstere':
                m = Basemap( projection=args[5],lon_0=0,boundinglat=0)
            else:
                m = Basemap( projection=args[5],lon_0=0,\
                llcrnrlat=args[3][2],urcrnrlat=args[3][3],\
                llcrnrlon=args[3][0]-360,urcrnrlon=args[3][1])
            m.fillcontinents(color='0.8')
            m.drawmapboundary(fill_color='0.9')
            m.drawmapboundary(fill_color='#000099')
            m.drawparallels(np.arange(-90,90,20), labels=[1,1,0,1])
            m.drawmeridians(np.arange(0,360,30), labels=[1,1,0,1])
            plt.title('EOF # '+str(i+1))
            mask=ma.masked_where(zi == np.nan, zi)
            lon[lon > 180] -= 360
            im=m.contourf(lon,lat,mask,levels,latlon=True, cmap=plt.cm.seismic)
            cb = m.colorbar(im, pad=0.7)
        fig, ax = plt.subplots(nrows=2, ncols=2)
        for i in range(4):
            time_series_EOF=np.dot(np.transpose(eigvec[:,i]),np.transpose(EOF))
            time_series_EOF=(time_series_EOF-np.nanmean(time_series_EOF))/np.std(time_series_EOF)    
            time_series_EOF=runningMeanFast(time_series_EOF, 10)
            plt.subplot(2, 2, i+1)
            years=np.linspace(args[1], args[2], args[2]-args[1]+1)
            plt.plot(years,time_series_EOF, 'k') 
            y2=np.zeros(len(time_series_EOF))
            plt.fill_between(years, time_series_EOF, y2, where=time_series_EOF >= y2, color = 'r', interpolate=True)
            plt.fill_between(years, time_series_EOF, y2, where=time_series_EOF <= y2, color = 'b', interpolate=True)
            plt.axhline(linewidth=1, color='k')
            plt.title('EOF # '+str(i+1))
        return EOF
    else:
        return EOF


def calc_MLD(*args):
    import seawater as sw
    # args :
    # 1) netcdf_read object of thetao;
    # 2) netcdf_read object of so;
    # 3) month;
    # 4) 0/1 for SH/NH (SH below 50S; NH above 50N)
    # 5) year start
    # 6) year_end
    # 7) netcdf_read object of area
    # 8) lon
    # 9) lat
    # 10) plot option 0/1 np/yes
    if args[3]==0:
        depth_MLD_tr=2000
    else:
        depth_MLD_tr=1000
        
    start,end = args[0].find_time(args[0].times, args[4], args[5])
    deep_conv_area=[]
    areacello=args[6]
    lon,lat,areacello=interpolate_2_custom_grid(args[0].x,args[0].y,areacello, args[7], args[8])
    data_plot=np.full([args[5]-args[4]+1,len(lon),len(lon[0])], np.nan)
    for i in range(int((end+1-start)/12)):
        #print (int((end+1-start)/12))
        #print(start+12*i,start+12*i+11)
        print(args[4]+i)
        data_thetao_extracted=args[0].extract_data(args[0].variable,start+12*i+args[2]-1,start+12*i+args[2]-1)
        data_so_extracted=args[1].extract_data(args[1].variable,start+12*i+args[2]-1,start+12*i+args[2]-1)
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
        lon,lat,data_i=interpolate_2_custom_grid(args[0].x,args[0].y,data_dens, args[7], args[8])
        #lon,lat,data_i=interpolate_2_reg_grid(args[0].x,args[0].y,data_dens)
        data_i[data_i>100000]=np.nan
        
        if (int(args[3])==int(0)):
            [ii,jj] = np.where(lat<=-50)###indeces####
        elif (int(args[3])==int(1)):
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
                elif MLD==49:
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
                    data_plot[i,ii[k],jj[k]]=float(interpol_z)     
        deep_conv_area.append(area)
    deep_conv_area=np.asarray(deep_conv_area)
##    st=np.nanstd(deep_conv_area)
##    st_area=area/st
##    st2 = np.where(st_area>=1.9)
##    average_MLD=np.nanmean(data_plot[st2],axis=0)
    average_MLD=np.nanmean(data_plot,axis=0)
    if args[3]==0:
        indeces = np.where(np.logical_or((lon<=30) & (average_MLD>depth_MLD_tr), (lon>=300) &(average_MLD>depth_MLD_tr)))
    else:
        indeces = np.where(np.logical_and((lon>=30) & (average_MLD>depth_MLD_tr), (lon<=330) &(average_MLD>depth_MLD_tr)))        
    if args[9]==1:
        fig=plt.figure()
        years=np.linspace(args[4],args[5],args[5]-args[4]+1)
        #print (years,area)
        plt.plot(years,deep_conv_area,'b')
        #plt.plot(years[st2],deep_conv_area[st2],'r')
        fig=plt.figure()
        if args[3]==0:
            m = Basemap( projection='spstere',lon_0=0,boundinglat=-30)
        else:
            m = Basemap( projection='npstere',lon_0=0,boundinglat=30)
        #m = Basemap(projection='mill',lon_0=180)
        m.drawcoastlines(linewidth=1.25)
        m.fillcontinents(color='0.8')
        #m.drawmapboundary(fill_color='#000099')
        m.drawparallels(np.arange(-90,90,20), labels=[1,1,0,1])
        m.drawmeridians(np.arange(0,360,30), labels=[1,1,0,1])
        #lon[lon > 180] -= 360
        im=m.contourf(lon,lat,average_MLD,200,latlon=True, cmap=plt.cm.jet)
        plt.colorbar(im)
        m.scatter(lon[indeces],lat[indeces],2,latlon=True)
    return deep_conv_area, data_plot, lon, lat, indeces


def time_depth_plot(*args):
    # args :
    # 1) netcdf_read object;
    # 2) year start
    # 3) year_end
    # 4) indices
    # 5) lon
    # 6) lat
    # 7) plot option 0/1 np/yes
    # 8) depth for Convection Index
    start,end = args[0].find_time(args[0].times, args[1], args[2])
    [ii,jj]=args[3]
    region=[]
    for i in range(int((end+1-start)/12)):
        data=args[0].extract_data(args[0].variable,start+12*i,start+12*i+11)
        data=np.asarray(data)
##        print(start)
##        print (data)
        data[data>100000]=np.nan
        data=np.nanmean(data,axis=0)
        data=np.squeeze(data)
        lon,lat,data_i=interpolate_2_custom_grid(args[0].x,args[0].y,data, args[4], args[5])
        data_i=data_i[:,ii,jj]
##        print(np.nanmax(data_i))
##        print(np.nanmin(data_i))
        print(args[1]+i)
        region.append(np.nanmean(data_i,axis=1))
    print(args[6])
    if int(args[6])==1:
        fig=plt.figure()
        years=np.linspace(args[1],args[2],args[2]-args[1]+1)
        d=args[0].lvl[:]
        im = plt.contourf(years, d, np.transpose(region), cmap=plt.cm.jet)
        plt.gca().invert_yaxis()
        l = plt.axhline(y=args[7])
        plt.colorbar(im)
    a=args[0].find_depth(args[0].lvl, args[7], args[7])
    print(a[0])
    region=np.asarray(region)
    print(region.shape)
    convection_index=region[:,a[0]]
    return region,convection_index






        
