#!/usr/bin/env python
# coding: utf-8

#Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
from pydap.client import open_url
from datetime import date
from mpl_toolkits.basemap import Basemap
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import calendar
import seaborn as sns
from datetime import date
from os import listdir
from os.path import isfile, join
from scipy.ndimage import zoom
import os
from matplotlib.patches import Polygon

def geom_mean(x, shift=1):
    x_shifted=x+shift
    gsm = np.exp(np.mean(np.log(x_shifted))) - shift
    return gsm

def geom_shift(x, shift=1):
    x_shifted = x + shift
    gsd = np.std((np.log(x_shifted)))
    return gsd

def geometric(historic, ci=0.95):
    #Take geometric Mean with shift parameter (see epiCo R)
    central = np.apply_along_axis(geom_mean, axis=0, arr=historic)
    
    #Calculate standard deviation along axis with shift parameter
    interval = np.apply_along_axis(geom_shift, axis=0, arr=historic)
    
    #Calc intervals
    up_lim = np.exp(np.log(central) + np.abs(interval))
    low_lim = np.exp(np.log(central) - np.abs(interval))
    
    return central, up_lim, low_lim

def endemic_channel(predictands,city,outliers=0.99,plots=True):  
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    c = predictands[predictands.city==city]
    c=c.sort_values(by=['year','month'])
    
    #Specify a threshold for outliers
    remove = np.quantile(c.cases,outliers)

    if plots==True:
        # Show extreme values removed from endemic channel
        y=c.cases
        x = np.arange(1, len(y)+1)
        plt.plot(x, y, label='Data')
        # Highlight points exceeding the 99th percentile
        plt.scatter(x[y > remove], y[y > remove], color='red', label='Exceeding 99th percentile')
        plt.title('Values removed from endemic channel calculation')
        plt.show()
    

    c=c[c.cases<remove]
    
    
    pivot_df = c.pivot_table(index='year', columns='month', values='cases', aggfunc='sum')
    pivot_df=pivot_df.dropna()
    
    
    central, upper, lower = geometric(pivot_df + 1)

    ec = pd.DataFrame({'central':central,'upper':upper,'lower':lower})
    
    if plots==True:
        ec.plot()
        plt.title('Endemic Channel')
        plt.show()
        
        
    return ec

#Read in datasets
def read_denv(path, metric, incidence_per=100000, locations=True):
    cali=pd.read_csv(path+"cali_consolidado_2006_2021.csv")
    cali['location'] = "cali"
    cucuta=pd.read_csv(path+"cucuta_consolidado_2006_2021.csv")
    cucuta['location'] = "cucuta"
    leticia=pd.read_csv(path+"leticia_consolidado_2008_2021.csv")
    leticia['location']='leticia'
    medellin=pd.read_csv(path+"medellin_consolidado_2009_2021.csv")
    medellin['location']='medellin'
    #Concatenate data
    ldf=pd.concat([cali,cucuta,leticia,medellin])
    ldf['datetime']=pd.to_datetime(ldf['FECHA'])
    ldf.set_index('datetime',inplace=True)
    
    ldf['incidence'] = (ldf['DENGUE']+ldf['DENGUE GRAVE'])/ldf.POBLACION
    
    if metric == "count":
        predictands = pd.DataFrame(columns = ['city','month','year','cases'])
        for city in ['medellin','cucuta','cali','leticia']:
            print(city)
            for selected_month in range(1,13):
                hold = ldf[ldf['location'] == city]
                hold['TotDENGUE'] = hold['DENGUE']+hold['DENGUE GRAVE']
                hold1 = hold[hold.index.month == selected_month]['TotDENGUE'].resample("1M").mean().dropna()
                d={"city":city,"month":selected_month,"year":hold1.index.year,"cases":hold1.values}
                predictands=pd.concat([predictands,pd.DataFrame(d)])
                
    if metric == "incidence":
        predictands = pd.DataFrame(columns = ['city','month','year','cases'])
        for city in ['medellin','cucuta','cali','leticia']:
            print(city)
            for selected_month in range(1,13):
                hold = ldf[ldf['location'] == city]
                hold1 = hold[hold.index.month == selected_month]['incidence'].resample("1M").mean().dropna()
                d={"city":city,"month":selected_month,"year":hold1.index.year,"cases":hold1.values * incidence_per}
                predictands=pd.concat([predictands,pd.DataFrame(d)])
    
    cities = "none"
    
    
    
    if locations:
        cities = pd.DataFrame({'x': [-76.529916,-72.509743 , -69.946317,-75.587740], 'y': [3.444979,7.903487 ,-4.212734 ,6.242832 ], 'name': ['cali','cucuta','leticia','medellin']})
        cities = gpd.GeoDataFrame(cities, geometry=gpd.points_from_xy(cities['x'], cities['y']))
        
    
    return predictands, cities





#Get data from IRI climate library
# Download IRI Data
#urls, variable = dataURLs(sstA=True,gph=True,gphMb=[500,1000],znl=True,znlMb=[500,1000])
def dataURLs(sstA,gph,gphMb,znl,znlMb):
    urls = []
    variable = []
    names=[]
    
    if sstA:
        
        urls.append('http://iridl.ldeo.columbia.edu/SOURCES/.NOAA/.NCDC/.ERSST/.version5/.sst/T/%28Jan%201950%29%28Dec%202023%29RANGEEDGES/dods')
        variable.append('sst')
        names.append('ssta')
        
    if gph:
        for mb in gphMb:
            GPHurl = 'http://iridl.ldeo.columbia.edu/SOURCES/.NOAA/.NCEP-NCAR/.CDAS-1/.MONTHLY/.Intrinsic/.PressureLevel/.phi/P/%28'+str(mb)+'%29VALUES/dods'
            urls.append(GPHurl)
            variable.append('phi')
            names.append('gph'+str(mb))
        
    if znl:
        for mb in znlMb:
            ZNLurl = 'http://iridl.ldeo.columbia.edu/SOURCES/.NOAA/.NCEP-NCAR/.CDAS-1/.MONTHLY/.Intrinsic/.PressureLevel/.u/P/%28'+str(mb)+'%29VALUES/dods'
            urls.append(ZNLurl)
            variable.append('u')
            names.append('znl'+str(mb))
    
    return urls, variable, names
        

    
def openDAP(urls, variable, names, startyear, endyear, startmon, endmon):
    dataframes=[]
    latframe=[]
    lonframe=[]

    
    for v, url in enumerate(urls):
        dataset = open_url(url)

        data = dataset[variable[v]]

        #Months out from forecast
        months=data['T'][:]-0.5

        S = []
        for month in months: 
            S.append(date(1960,1,1) + pd.DateOffset(months = month))

        S=pd.to_datetime(S)

        #Lat/lon
        sstlat = dataset['Y'][:]
        sstlon = dataset['X'][:]

        #Create mask
        lon, lat = np.meshgrid(sstlon,sstlat)

        print('starting download')
        dataList = []
        for i, T in enumerate(S):
            if ((T.year>=startyear) & (T.year<=endyear)):
                if((T.month>=startmon) & (T.month<=endmon)):
                    dataList.append(data[variable[v]][i].squeeze())
                    #print(T)

        
        print('Finished downloading now saving '+names[v])

        savename = '/Volumes/Data Drive/Colombia/openDAPdownload/'+names[v]
        np.save(savename,dataList)

        savename = '/Volumes/Data Drive/Colombia/openDAPdownload/'+names[v]+'lat'
        np.save(savename,sstlat)

        savename = '/Volumes/Data Drive/Colombia/openDAPdownload/'+names[v]+'lon'
        np.save(savename,sstlon)

        savename = '/Volumes/Data Drive/Colombia/openDAPdownload/'+names[v]+'datetime'
        np.save(savename,S)
        
def loadClimData(names):
    dataframes=[]
    latframe=[]
    lonframe=[]
    dtframe=[]
    for name in names:
        mypath='/Volumes/Data Drive/Colombia/openDAPdownload/'
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        datafiles = [mypath+f for f in onlyfiles if name in f]
        for data in datafiles:
            if 'lon' in data:
                lonframe.append(np.load(data))  
            if 'lat' in data:
                latframe.append(np.load(data))
            if 'datetime' in data:
                dtframe.append(np.load(data))
            if ('lat' not in data) & ('lon' not in data) & ('datetime' not in data):
                dataframes.append(np.load(data))
    return dataframes, lonframe, latframe, dtframe



#NIPA-ish Monte Carlo
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def vcorr(X, y):
    # Function to correlate a single time series with a gridded data field
    # X - Gridded data, 3 dimensions (ntim, nlat, nlon)
    # Y - Time series, 1 dimension (ntim)

    ntim, nlat, nlon = X.shape
    ngrid = nlat * nlon

    y = y.reshape(1, ntim)
    X = X.reshape(ntim, ngrid).T
    Xm = np.nanmean(X,axis = 1).reshape(ngrid,1)
    ym = np.nanmean(y)
    r_num = np.nansum((X-Xm) * (y-ym), axis = 1)
    r_den = np.sqrt(np.nansum((X-Xm)**2, axis = 1) * np.nansum((y-ym)**2))
    r = (r_num/r_den).reshape(nlat, nlon)

    return r

def rank_simple(vector):
    return sorted(range(len(vector)), key=vector.__getitem__)

def rankdata(a):
    n = len(a)
    ivec=rank_simple(a)
    svec=[a[rank] for rank in ivec]
    sumranks = 0
    dupcount = 0
    newarray = [0]*n
    for i in range(n):
        sumranks += i
        dupcount += 1
        if i==n-1 or svec[i] != svec[i+1]:
            averank = sumranks / float(dupcount) + 1
            for j in range(i-dupcount+1,i+1):
                newarray[ivec[j]] = averank
            sumranks = 0
            dupcount = 0
    return newarray


def spcorr(X, y):
    
    # Function to correlate a single time series with a gridded data field
    # X - Gridded data, 3 dimensions (ntim, nlat, nlon)
    # Y - Time series, 1 dimension (ntim)
    ntim, nlat, nlon = X.shape
    ngrid = nlat * nlon

    
    X = X.reshape(ntim, ngrid).T
    
    y=rankdata(y)
    y=np.array(y)
    y = y.reshape(1, ntim)

    X=np.apply_along_axis(rankdata, 1, X)
    
    d = X-y

    r_num = 6*(np.nansum(d**2,axis=1))
    r_den = ntim*((ntim**2)-1)
    
    rs = (1-(r_num/r_den)).reshape(nlat, nlon)
    
    return rs

def sig_test(r, n, twotailed = True):
    import numpy as np
    from scipy.stats import t as tdist
    df = n - 2

    # Create t-statistic
    # Use absolute value to be able to deal with negative scores
    t = np.abs(r * np.sqrt(df/(1-r**2)))
    p = (1 - tdist.cdf(t,df))
    if twotailed:
        p = p * 2
    return p


def bootcorr(fieldData,clim_data,latField,lonField, ntim = 1000, corrconf = 0.95, bootconf = 0.80,
            debug = False, variable='none',spearman = False):
        from numpy import meshgrid, zeros, ma, isnan, linspace

        corrlevel = 1 - corrconf

        corr_grid = spcorr(X = fieldData, y = clim_data)

        n_yrs = len(clim_data)

        p_value = sig_test(corr_grid, n_yrs)
        

        #Mask insignificant gridpoints
        corr_grid = ma.masked_array(corr_grid, ~(p_value < corrlevel))
        #Mask land
        corr_grid = ma.masked_array(corr_grid, isnan(corr_grid))
        
        
        #Mask northern/southern ocean if SST
        if variable[:3]=='sst':
            corr_grid.mask[latField > 30] = True
            corr_grid.mask[latField < -60] = True
        #Mask southern hemisphere if Gph
        if variable[:3]=='gph':
            corr_grid.mask[latField < -15] = True
        if variable[:3]=='znl':
            corr_grid.mask[latField < -15] = True
            corr_grid.mask[:,(lonField <180)] = True
        
        nlat = fieldData.shape[1]
        nlon = fieldData.shape[2]
        
        
         ###INITIALIZE A NEW CORR GRID####

        count = np.zeros((nlat,nlon))

        dat = clim_data.copy()

        store=[]
        for boot in range(0,ntim):

            ###SHUFFLE THE YEARS AND CREATE THE BOOT DATA###
            idx = np.random.randint(0, len(dat) - 1, len(dat))
            boot_fieldData = np.zeros((len(idx), nlat, nlon))
            boot_fieldData[:] = fieldData[idx]
            boot_climData = np.zeros((len(idx)))
            boot_climData = dat[idx]
           
            if spearman:
                boot_corr_grid = spcorr(X = boot_fieldData, y = boot_climData)
            
            else:
                boot_corr_grid = vcorr(X = boot_fieldData, y = boot_climData)

            p_value = sig_test(boot_corr_grid, n_yrs)

            count[p_value <= corrlevel] += 1
            
            store.append(np.sum(p_value <= corrlevel))
        

        #print(np.sum(store>np.sum(p_value < corrlevel))/len(store))
        #plt.rcParams.update({'font.size': 15})
        #hfig, hax = plt.subplots()
        #hax.hist(store,bins=20,color='gray',edgecolor='black')
        #hax.set_xlabel('Num. Significant Grids')
        #hax.axvline(x=np.sum(p_value < corrlevel), color='red', linestyle='--')
        #plt.show()
        
        
        

        ###CREATE MASKED ARRAY USING THE COUNT AND BOOTCONF ATTRIBUTES
        corr_grid = np.ma.masked_array(corr_grid, count < bootconf * ntim)

        
        return corr_grid

def regcorr(fieldData, clim_data, ntim = 100, corrconf = 0.95, bootconf = 0.80,
            debug = False):
        from numpy import meshgrid, zeros, ma, isnan, linspace

        corrlevel = 1 - corrconf

        corr_grid = vcorr(X = fieldData, y = clim_data)

        n_yrs = len(clim_data)

        p_value = sig_test(corr_grid, n_yrs)
        

        #Mask insignificant gridpoints
        corr_grid = ma.masked_array(corr_grid, ~(p_value < corrlevel))
        #Mask land
        corr_grid = ma.masked_array(corr_grid, isnan(corr_grid))
        #Mask northern/southern ocean
        corr_grid.mask[sstlat > 30] = True
        corr_grid.mask[sstlat < -60] = True
        nlat = len(sstlat)
        nlon = len(sstlon)

        
        return corr_grid


def GridPCA(corr_grid,fieldData):
    X=fieldData[:,~corr_grid.mask].T
    
    X=X[~np.isnan(X).any(axis=1),:]
    sc = StandardScaler()
    Xs=sc.fit_transform(X)

    pca = PCA()   
    
    Xp = pca.fit_transform(Xs.T)
    pcs = Xp[:,pca.explained_variance_ratio_>0.1]
    
    return pcs

def GridMeanObs(corr_grid,fieldData):
    X=fieldData[:,~corr_grid.mask].T
    
    X=X[~np.isnan(X).any(axis=1),:]

    means = np.nanmean(X, axis=0)
    
    return means


def RegPCA(fieldData):
    data = fieldData
    lat, lon, time = data.shape
    X = data.reshape((lat * lon, time))

    X[X==-999] = np.nan
    valid_rows = ~np.any(np.isnan(X), axis=1)
    X=X[valid_rows]

    sc = StandardScaler()
    Xs=sc.fit_transform(X)
    
    pca = PCA()
    pca_result = pca.fit_transform(Xs)
    ex = pca.explained_variance_ratio_
    pcs = pca.components_[ex>0.1,:]
    
    return pcs
    

def correlate_grid(predictands, dataframes, latList, lonList, dtframe,cities, map_global, varname,variable='none', lag = 1, fontsize=12):
    
    plt.rcParams.update({'font.size': fontsize})

    alldata = []
    masksave=[]
    citysave = []

    S=pd.to_datetime(dtframe)
    Tson=S[(S.year>=1950) & (S.year<=2021) & (S.month>=9) & (S.month<=11)]

    lat = latList
    lon = lonList
    
    S=S[(S.year>=2006) & (S.year<=2021)]

    data=np.dstack(dataframes)

    for city in set(predictands['city']):

        fig, ax = plt.subplots(3,4,figsize=(15, 10))
        ax=ax.ravel()

        for i, month in enumerate(set(predictands['month'])):
            ptnd = predictands[(predictands['month'] == month) & (predictands['city']==city)]

            data[data==-999] = np.nan

            #Change lag here
            Tdata=(S.year>=np.min(ptnd['year'])) & (S.year<=np.max(ptnd['year'])) & (S.month==range(1,13)[month-lag])

            print('Correlating '+calendar.month_name[month]+' cases with '+calendar.month_name[range(1,13)[month-lag]]+' climate')
            
            dataMean = data[:,:,Tdata]

            ny = data.shape[0]; nx = data.shape[1]

            x=dataMean
            y=ptnd['cases']

            y = np.array(y)
            x=np.transpose(x, (2, 0, 1))

            if map_global:

                corMap=bootcorr(x,y,latField=lat,lonField=lon,bootconf=0.8,ntim=1000,spearman=False,variable=variable)
                allCors=vcorr(x,y)

                #Mask northern/southern ocean if SST
                if variable[:3]=='sst':
                    allCors[lat > 30] = np.nan
                    allCors[lat < -60] = np.nan
                #Mask southern hemisphere if Gph
                if variable[:3]=='gph':
                    allCors[lat < -15] = np.nan
                if variable[:3]=='znl':
                    allCors[lat < -15] = np.nan
                    allCors[:,(lon <180)] = np.nan
                

                if np.nansum(corMap>0):
                    pc = GridPCA(corMap,x)
                    means = GridMeanObs(corMap,x)
                    #Make dataframe
                    df = pd.DataFrame(pc)
                    df['means']=means
                    df['city'] = city
                    df['month'] = month
                    df['year']=ptnd['year']
                    alldata.append(df)
                else:
                    #Make dataframe
                    df = pd.DataFrame([np.nan]*len(y))
                    df['means']=[np.nan]*len(y)
                    df['city'] = city
                    df['month'] = month
                    df['year']=ptnd['year']
                    alldata.append(df)


                m = Basemap(projection='cyl',llcrnrlon=0,llcrnrlat=-90,urcrnrlon=357.5,urcrnrlat=90,resolution='c')
                plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.00)
                lons, lats = m.makegrid(nx, ny) # get lat/lons of ny by nx evenly space grid.
                mx, my = m(lons, lats)  # transform coordinates
                
                                                   

                levels = np.linspace(-1,1,6)
                cs = m.contour(mx,my,allCors,cmap='coolwarm',levels=levels,ax=ax[i])
                levels = np.linspace(-1,1,21)
                cs = m.contourf(mx,my,corMap,cmap='coolwarm',levels=levels,ax=ax[i])
                cbar = m.colorbar(cs,location='bottom',pad="5%",ax=ax[i])
                cbar.set_label('Pearson R')
                m.drawcoastlines()
                ax[i].set_title(calendar.month_name[i+1])
                
                #plot ENSO
                x1,y1 = m(-90,-10)
                x2,y2 = m(-90,0)
                x3,y3 = m(-80,0)
                x4,y4 = m(-80,-10)
                poly = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],edgecolor='green',linewidth=1)
                ax[i].add_patch(poly)


                x1,y1 = m(-150,-5)
                x2,y2 = m(-150,5)
                x3,y3 = m(-90,5)
                x4,y4 = m(-90,-5)
                poly = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],edgecolor='red',linewidth=1)
                ax[i].add_patch(poly)

                x1,y1 = m(-170,-5)
                x2,y2 = m(-170,5)
                x3,y3 = m(-120,5)
                x4,y4 = m(-120,-5)
                poly = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],edgecolor='orange',linewidth=1)
                ax[i].add_patch(poly)


                x1,y1 = m(-200,-5)
                x2,y2 = m(-200,5)
                x3,y3 = m(-150,5)
                x4,y4 = m(-150,-5)
                poly = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],edgecolor='blue',linewidth=1)
                ax[i].add_patch(poly)
                



            else:
                #Get city and buffer
                buffer=cities[cities.name==city.capitalize()].buffer(1, cap_style=3)

                corMap=regcorr(x,y)

                latmask=(sstlat > buffer.total_bounds[1]) & (sstlat < buffer.total_bounds[3])
                lonmask=(sstlon > buffer.total_bounds[0]+360) & (sstlon < buffer.total_bounds[2]+360)

                corMap.mask[latmask] = False
                corMap.mask[:,lonmask] = False

                corMap.mask[~latmask] = True
                corMap.mask[:,~lonmask] = True

                #clon = np.where(lon > 180, lon - 360, lon)
                #ax[i].scatter(clon,lat,c=corMap,zorder=2,s=20,cmap='bwr')
                #col.plot(ax=ax[i],facecolor='none',edgecolor='black',zorder=2)
                                    #cities[cities.name==city.capitalize()].plot(ax=ax[i],color='purple',edgecolor='black',markersize=12,zorder=1)
                #buffer.boundary.plot(ax=ax[i], color = 'black')
                #ax[i].set_ylim(-5,12.5)
                #ax[i].set_xlim(-80,-65)

                if np.nansum(corMap>0)>0:
                    if np.nansum(corMap>0)>1:
                        pc = GridPCA(corMap,x)
                    else:
                        pc = x[:,~corMap.mask]

                    #Make dataframe
                    df = pd.DataFrame(pc)
                    df['city'] = city
                    df['month'] = month
                    df['year']=ptnd['year']
                    alldata.append(df)

                    sns.regplot(pc,y,ax=ax[i],lowess=True)


                else:
                    #Make dataframe
                    df = pd.DataFrame([np.nan]*len(y))
                    df['city'] = city
                    df['month'] = month
                    df['year']=ptnd['year']
                    alldata.append(df)

            citysave.append(city)
            masksave.append(corMap.mask)
        
        
        #Save mask
        p = '/Volumes/Data Drive/Colombia/dataMasks/'
        f=varname
        np.save(p+f,masksave)
        
        p = '/Volumes/Data Drive/Colombia/dataMasks/'
        f='cityorder'+varname
        np.save(p+f,citysave)
        
        #Save PC
        savePC=pd.concat(alldata)
        savePC.to_csv('/Users/maxbeal/Desktop/PhD/Amazon/Data/DENV_preds/'+varname+'.csv')

        #Plots
        cbar_ax = fig.add_axes([1.01, 0.15, 0.01, 0.7])
        cbar_ax.set_label('Pearson R')
        fig.colorbar(cs,cbar_ax)
        fig.suptitle(city)
        fig.tight_layout()
        plt.show()



#Download NMME
from pydap.client import open_url
import pandas as pd
import numpy as np
def downloadNMME(urls, lead,savename):

    dataframes=[]
    nmmeList = []
    frame = []
    timesave=[]

    for url in urls:
        dataset = open_url(url)
        variable = dataset.keys()[-1]
        print(variable)
        data = dataset[variable]
        

        #L (months ahead) ,M (ensemble members), S (months from 1960-01-01) ,lon,lat

        #Months out from forecast
        months_out=data['L'][:]

        fcst_ref = data['S'][:]
        from datetime import date

        S = []
        for month in fcst_ref: 
            S.append(date(1960,1,1) + pd.DateOffset(months= month))

        S=pd.to_datetime(S)
        timesave.append(S)

        #Lat/lon
        sstlat = dataset['Y'][:]
        sstlon = dataset['X'][:]

        #Create mask
        lon, lat = np.meshgrid(sstlon,sstlat)
        


        from numpy import squeeze
        for i,ref in enumerate(S): #Months prediction originates
            if ref.year >=2006 and ref.year<=2021:
                for j, out in enumerate(months_out): #Months forecasted out (choose)
                    if out == lead:
                        grid = data.array[i,j,:,:].squeeze()
                        nmmeList.append(grid) #Save NMME field
                        d={"fcst_month":ref,"forecast_out":out}
                        frame.append(d)
                        print(ref,out)
                        
                lastdate=ref

            else:
                continue

        dataframes.append(pd.DataFrame(frame))
        
        
    #Save NMME
    saveframe = np.dstack(nmmeList)
    p = '/Volumes/Data Drive/Colombia/NMME/'
    np.save(p+savename,saveframe)
    
    np.save(p+'lat'+savename,sstlat)
    np.save(p+'lon'+savename,sstlon)
    
    #NMME=pd.concat(dataframes[1])
    NMME=dataframes[1]

    fcst_target=[]
    NMME['forecast_out']=(NMME['forecast_out']+0.5)

    for i, mon in enumerate(NMME['fcst_month']):
        fcst_target.append(NMME['fcst_month'].iloc[i] + pd.DateOffset(months=NMME['forecast_out'].values[i]))
    NMME['target']=fcst_target
    
    #Save CSV

    NMME.to_csv(p+savename)


def downscale_array(input_array, target_shape):
    """
    Downscale the input array to the target shape using scipy.ndimage.zoom.

    Parameters:
    - input_array (numpy.ndarray): The input array to be downscaled.
    - target_shape (tuple): The target shape of the downscaled array.

    """
    # Calculate the zoom factors for each dimension
    zoom_factors = tuple(target_shape[i] / input_array.shape[i] for i in range(len(target_shape)))

    # Downscale the array using scipy.ndimage.zoom
    downscaled_array = zoom(input_array, zoom_factors, order=1,cval=np.nanmean(input_array))

    return downscaled_array




#Finds the largest continuous correlation group on the mask
def largest_continuous_false_group(matrix):
    def dfs(i, j, current_group):
        if 0 <= i < rows and 0 <= j < cols and not visited[i, j] and matrix[i, j] == False:
            visited[i, j] = True
            current_group.append((i, j))
            
            # Check neighbors
            dfs(i - 1, j, current_group)  # Up
            dfs(i + 1, j, current_group)  # Down
            dfs(i, j - 1, current_group)  # Left
            dfs(i, j + 1, current_group)  # Right

    rows, cols = matrix.shape
    visited = np.zeros_like(matrix, dtype=bool)
    largest_group = []

    for i in range(rows):
        for j in range(cols):
            if not visited[i, j] and matrix[i, j] == False:
                current_group = []
                dfs(i, j, current_group)

                if len(current_group) > len(largest_group):
                    largest_group = current_group

    return largest_group



import math

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth specified in decimal degrees
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles.
    return c * r

def nearest_point(lat, lon, points):
    """
    Find the nearest point in a matrix of latitude and longitude
    to a given latitude and longitude coordinate
    """
    min_distance = float('inf')
    nearest_point = None

    for point in points:
        distance = haversine(lat, lon, point[0], point[1])
        if distance < min_distance:
            min_distance = distance
            nearest_point = point

    return nearest_point




#Apply correlation masks to NMME forecasts, get PCs
#0-12 masksave is cali Jan-Dec


def GridPCA_nmme(mask,fieldData):

    X=fieldData[~mask,:]
    X=X[~np.isnan(X).any(axis=1),:]
    sc = StandardScaler()
    Xs=sc.fit_transform(X)

    pca = PCA()
    #print("Variance explained: ",pca.explained_variance_ratio_)
    
    
    Xp = pca.fit_transform(Xs.T)
    pcs = Xp[:,pca.explained_variance_ratio_>0.1]
    
    return pcs

def GridMean(mask,fieldData):
    
    X=fieldData[~mask,:]
    X=X[~np.isnan(X).any(axis=1),:]
    #sc = StandardScaler()
    #Xs=sc.fit_transform(X)

    means = np.nanmean(X, axis=0)
    
    return means

def maskForecast(nmme_data,masksave,dateDf,cityNames,savename, plots=False):
    alldata = []
    nmme_data = nmme_data[:,:,~dateDf.duplicated(keep='first')] #get rid of duplicate dates(keep first)
    dateDf = dateDf[~dateDf.duplicated(keep='first')]
    months = np.concatenate([np.linspace(1,12,12)]*4)
    
    
    for i in range(0,len(masksave)):
        mask = masksave[i]

        Tdata=(dateDf.dt.year>=2006) & (dateDf.dt.year<=2021) & (dateDf.dt.month==months[i])
        doi = nmme_data[:,:,Tdata]
        
        
        downscaleList =[]
        target_shape = (mask.shape[0],mask.shape[1])
        for j in range(0,doi.shape[2]):
            downscaleList.append(downscale_array(nmme_data[:,:,j], target_shape))
        downscale_data=np.dstack(downscaleList)
        

        
        doi = downscale_data
        if (len(dateDf[Tdata])>0):
            maskHold=[]
            
            if np.nansum(~mask)>5:
                pc = GridPCA_nmme(mask,doi)
               
                means = GridMean(mask,doi)

                
                
                for year in range(1,len(dateDf[Tdata].dt.year)):
                    masked_data = np.where(~mask, doi[:,:,year], np.nan)
                    maskHold.append(masked_data)


                    if plots:
                        ny = masked_data.shape[0]; nx = masked_data.shape[1]
                        m = Basemap(projection='cyl',llcrnrlon=0,llcrnrlat=-90,urcrnrlon=357.5,urcrnrlat=90,resolution='c')
                        plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.00)
                        lons, lats = m.makegrid(nx, ny) # get lat/lons of ny by nx evenly space grid.
                        mx, my = m(lons, lats)  # transform coordinates


                        cs = m.contour(mx,my,masked_data>0,cmap='coolwarm')
                        cs = m.contourf(mx,my,doi[:,:,year],cmap='coolwarm')

                        plt.show()

                
                masked_data=np.dstack(maskHold)
            

            else:
                means = pd.DataFrame([np.nan]*doi.shape[2])
                pc = pd.DataFrame([np.nan]*doi.shape[2])


            
            #Make dataframe
            df = pd.DataFrame(pc)
            df['means'] = means
            df['city'] = cityNames[i]
            df['month'] = months[i]
            df['year']=dateDf[Tdata].dt.year.reset_index()['target']
            alldata.append(df)

    save=pd.concat(alldata)
    
    p = '/Users/maxbeal/Desktop/PhD/Amazon/Data/DENV_preds/'
    f = savename
    save.to_csv(p+f+'.csv')
    
    return save



def filter_files(directory, modelname, varname,lag):
    file_names = []
    # Iterate over all files in the directory
    for file_name in os.listdir(directory):
        # Check if the substring is contained in the file name
        if (modelname in file_name) & (varname in file_name) & (str(lag) in file_name):
            file_names.append(file_name)
            
            if 'lat' in file_name:
                flat = file_name
            if 'lon' in file_name:
                flon = file_name
            if 'time' in file_name:
                ftime = file_name
            if 'lon' not in file_name and 'lat' not in file_name and 'time' not in file_name:
                print(file_name)
                f = file_name
                
    return f, flat, flon, ftime


def process_nmme(nmme_p, mask_p, f, flat, flon, ftime, map_global, varname, denv_metric, MCregions = True, consolidate_masks=False):
    #Data
    nmme_data=np.load(nmme_p+f)
    #Time
    nmme_meta = pd.read_csv(nmme_p+ftime)
    S = pd.to_datetime(nmme_meta['target'])

    if map_global:
        #Read mask data
        fmask=varname+'.npy'
        data_mask=np.load(mask_p+fmask)
        
        fcity=[file for file in os.listdir(mask_p) if ('cityorder' in file) & (varname in file)]
        cityNames=np.load(mask_p+fcity[0])
        
        print('loading: '+fmask+' masks')

        #Option to run a funciton that will choose the largest continuous mask region
        if consolidate_masks:
            targeted_mask = []
            for mask in range(1,len(data_mask)):
                result = largest_continuous_false_group(data_mask[mask])
                mask = np.zeros_like(data_mask[mask], dtype=bool)
                for i, j in result:
                    mask[i, j] = True
                data_mask.append(~mask)

        if MCregions:
            #Masking Function, saves PCs
            savename = 'nmme_'+varname+'_3mo' #CHANGE IF LAG INCREASES
            nmme = maskForecast(nmme_data,masksave=data_mask,dateDf=S,cityNames = cityNames, savename = savename,plots=False)

    else:
        latitudes=np.load(nmme_p+flat)
        longitudes=np.load(nmme_p+flon)
        lat_mesh, lon_mesh = np.meshgrid(latitudes, longitudes, indexing='ij')
        points = np.column_stack((lat_mesh.ravel(), lon_mesh.ravel()))

        nmme_data = nmme_data[:,:,~S.duplicated(keep='first')] #get rid of duplicate dates(keep first)
        S = S[~S.duplicated(keep='first')]

        pdt, cities = read_denv('~/Desktop/PhD/Amazon/Data/',metric=denv_metric) #Predictands, cities gdf
        nmmesave=[]
        for num in range(0,len(cities)):
            nearest = nearest_point(cities.y[num], cities.x[num], points)
            print("Nearest point:", nearest)
            localVar = nmme_data[:,:,:][(nearest[0] == lat_mesh) & (nearest[1]==lon_mesh)].squeeze()
            nmmesave.append(pd.DataFrame({'month':S.dt.month,'year':S.dt.year,'city':cities.name[num],'means':localVar}))
        nmme=pd.concat(nmmesave)
    
    return nmme



def subset_data(df, city, month):
    subset = df[(df['city'] == city) & (df['month'] == month)]
    return subset

def r2(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    mean_actual = np.mean(actual)
    tss = np.sum((actual - mean_actual) ** 2)
    rss = np.sum((actual - predicted) ** 2)
    r_squared = 1 - (rss / tss)
    return r_squared
