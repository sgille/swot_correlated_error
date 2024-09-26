def skill_matrix(MSLA, Psi, k_n, l_n, MModes, Rm, lon, lat, T_time):
    
    '''
    Evaluate the skillfulness of each wave in fitting the daily average AVISO SSH anomaly. 
    
    Input: 
    SSHA_vector: AVISO SSH anomaly, 
    Psi (horizontal velocity and pressure structure functions), 
    k_n (zonal wavenumber), 
    l_n (latitudional wavenumber), 
    frequency, 
    longitude, latitude and time. 
    
    Output: skill matrix， SSH anomalies as vector, longitude, latitude, time and Rossby deformation radius.
    '''
    
    import numpy as np
    from tqdm import tqdm
    from numpy import linalg as LA
    from scipy import linalg
    
    Phi0 = lat.mean() # central latitude (φ0)
    Omega = 7.27e-5 # Ω is the angular speed of the earth
    Earth_radius = 6.371e6 / 1e5 #Earth_radius # meters
    Beta = 2 * Omega * np.cos(Phi0*np.pi/180.)  / Earth_radius
    f0 = 2 * Omega * np.sin(Phi0*np.pi/180.) #1.0313e-4 # 
    
    dlon = lon - lon.mean()
    dlat = lat - lat.mean()
    
    SSHA_masked = np.ma.masked_invalid(MSLA)
    SSHA_vector = np.zeros(MSLA.size)
    time_vector = np.zeros(MSLA.size)
    Iindex, Jindex, Tindex = np.zeros(MSLA.size), np.zeros(MSLA.size), np.zeros(MSLA.size)
    
    count = 0
    for tt in range(MSLA.shape[2]):
        for jj in range(MSLA.shape[0]):  # loop over latitude
            for ii in range(MSLA.shape[1]):  # loop over longitude
                if(SSHA_masked[jj, ii, tt] != np.nan): 
                    SSHA_vector[count] = MSLA[jj, ii, tt]   # MSLA subscripts:  lat, lon, time
                    #lon_vector[count] = lon[jj]
                    #lat_vector[count] = lat[ii]
                    time_vector[count] = T_time[tt]
                    Iindex[count], Jindex[count], Tindex[count] = int(ii), int(jj), int(tt)
                    count = count + 1

    H0 = np.zeros([len(SSHA_vector), 2]) # Number of data * Number of models
    skill = np.zeros([len(k_n), len(l_n), MModes])
    omega = np.zeros([len(k_n), len(l_n), MModes])
    
    for kk in range(len(k_n)):
        for ll in range(len(l_n)):
            for mm in range(MModes):
                omega[kk, ll, mm] =  Beta * k_n[kk, mm] / (k_n[kk, mm] ** 2 + l_n[ll, mm] ** 2 + Rm[mm] ** -2) # non-dispersive wave

#with tqdm(total= len(k_n) * len(l_n)* MModes) as pbar:
    for kk in range(len(k_n)):
        for ll in range(len(l_n)):
            for mm in range(MModes):
                for count in range(len(Iindex)):
                    # change lon, lat to (dlon, dlat = (lon, lat) - mean
                    # conversion to distance 
                    H0[count, 0] = Psi[0, mm] * np.cos(k_n[kk, mm] * dlon[int(Iindex[count])] + l_n[ll, mm] * dlat[int(Jindex[count])] + omega[kk, ll, mm] * T_time[int(Tindex[count])]) 
                    H0[count, 1] = Psi[0, mm] * np.sin(k_n[kk, mm] * dlon[int(Iindex[count])] + l_n[ll, mm] * dlat[int(Jindex[count])] + omega[kk, ll, mm] * T_time[int(Tindex[count])])       

                M = 2

                RR, PP = .1, 1

                HTH = np.matmul(H0.T, H0)

                for pp in range(M):
                    HTH[pp, pp] = HTH[pp, pp] +  RR/PP

                D = np.matmul(LA.inv(HTH), H0.T)  

                X_ = np.matmul(D, SSHA_vector)

                # calculate residual
                residual = SSHA_vector - np.matmul(H0, X_)

                # variance of residual
                # evaluate skill (1- rms_residual/rms_ssha_vector) and store the skill
                # skill value nn, ll, mm, = skill value
                skill[kk, ll, mm] = 1 - (np.mean(residual**2)) / (np.mean(SSHA_vector**2))

                #pbar.update(1)
                    
    return skill, SSHA_vector, Iindex, Jindex, Tindex


def inversion(Y, H_v, P_over_R):
    
    '''
    Solve for X given observations (Y), basis function (H_v) and signal to noise ratio (P_over_R).
    Return: X (amplitudes of Rossby waves)
    This is all in model space.
    '''
    
    import numpy as np
    from numpy import linalg as LA

    HTH = np.matmul(H_v.T, H_v)
    
    HTH = HTH +  P_over_R #, P: uncertainty in model, R: uncertainty in data, actually R_over_P
    
    D = np.matmul(LA.inv(HTH), H_v.T)
    
    amp = np.matmul(D, Y)
    
    Y_estimated = np.matmul(H_v, amp)
    
    return amp, Y_estimated
    
def build_h_matrix2(MSLA, MModes, k_n, l_n, longitude, latitude, T_time, Psi, Rm, day):
    
    '''
    Build H matrix or basis function for Rossby wave model.
    
    Input:
    SSHA_vector: SSH anomalies as a vector,
    Psi (horizontal velocity and pressure structure functions), 
    k_n (zonal wavenumber), 
    l_n (latitudional wavenumber), 
    frequency, 
    longitude, latitude and time. 
    
    Output: H matrix for Rossby wave model
    
    '''
    
    import numpy as np
    
    Phi0 = latitude.mean() # central latitude (φ0)
    Omega = 7.27e-5 # Ω is the angular speed of the earth
    Earth_radius = 6.371e6 / 1e5 # meters
    Beta = 2 * Omega * np.cos(Phi0*np.pi/180.) / Earth_radius 
    f0 = 2 * Omega * np.sin(Phi0*np.pi/180.) 

    dlon = longitude - longitude.mean()
    dlat = latitude - latitude.mean()
    #print('lon',lon.mean(),'lat',lat.mean())
    M = len(k_n) * len(l_n)

    omega = np.zeros([len(k_n), len(l_n), MModes])
    Iindex, Jindex, Tindex = np.zeros(MSLA.size), np.zeros(MSLA.size), np.zeros(MSLA.size)
    day_use = np.zeros(MSLA.size)
    SSHA_vector = np.zeros(MSLA.size)
    
    count = 0
    for tt in range(MSLA.shape[2]):
        for jj in range(MSLA.shape[0]):
            for ii in range(MSLA.shape[1]):
                if (MSLA[jj:jj+1,ii,tt].mask==False):
                    SSHA_vector[count] = MSLA[jj, ii, tt]
                    day_use[count]=day+tt
                    Iindex[count], Jindex[count], Tindex[count] = int(ii), int(jj), int(tt)
                    count = count + 1

    count_max=count
    H_cos, H_sin = np.zeros([count_max,M]), np.zeros([count_max, M])
    H_all = np.zeros([count_max, M * 2])
    
    # print(count_max, MModes, k_n, l_n)
    nn = 0 
    for kk in range(len(k_n)):
        for ll in range(len(l_n)):
            for mm in range(MModes):
                omega[kk, ll, mm] = Beta * k_n[kk, mm] / (k_n[kk, mm] ** 2 + l_n[ll, mm] ** 2 + Rm[mm] ** -2)          
                for count in range(count_max):
                    H_cos[count, nn] = Psi[0, mm] * np.cos(k_n[kk, mm] * dlon[int(Iindex[count])] + l_n[ll, mm] * dlat[int(Jindex[count])] + omega[kk, ll, mm] * T_time[int(day_use[count])])
                    H_sin[count, nn] = Psi[0, mm] * np.sin(k_n[kk, mm] * dlon[int(Iindex[count])] + l_n[ll, mm] * dlat[int(Jindex[count])] + omega[kk, ll, mm] * T_time[int(day_use[count])])
                nn += 1 
    
    
    H_all[:, 0::2] = H_cos 
    H_all[:, 1::2] = H_sin
    
    return H_all, SSHA_vector
    
def make_error_over_time(days, alpha, latitude_use, longitude_use, deltax_use, asc_des_use):
# days, alpha, yswath_index_left, yswath_index_right, y_mask_left, y_mask_right):
    
    '''
    This function models the time-varying error parameters in satellite swath data, including timing error, roll error, baseline dilation error, and phase error. The roll errors and baseline dilation errors are assumed to be correlated and are generated on the satellite swath. 
    
    Inputs:
    days:  counter for daily records
    alpha:  scaled noise parameter to use as amplitudes for correlated error
    latitude_use:  latitude of array (used only to determine size of array)
    longitude_use:  not used
    deltax_use:  cross-track distance from SWOT L2 product
    asc_des_use:  flag to distinguish ascending and descending passes, so that they are assigned different error terms

The output of the function includes an array of correlated error.
    '''
    import numpy as np
    
#     print(latitude_use.shape, deltax_use.shape, latitude_use[:5],longitude_use[:5])
    
    Tdim, ALdim, ACdim = len(days), len(latitude_use), 2 # time dimension, points in swath
    timing_err = np.ma.masked_all([Tdim, ALdim])
    # Roll error
    roll_err = np.ma.masked_all([Tdim, ALdim])
    # Baseline dilation error
    baseline_dilation_err = np.ma.masked_all([Tdim, ALdim])
    # phase error
    phase_err = np.ma.masked_all([Tdim, ALdim])
    phase_err_3 = np.ma.masked_all([Tdim, ALdim])
    phase_err_4 = np.ma.masked_all([Tdim, ALdim])
    phase_err_5 = np.ma.masked_all([Tdim, ALdim])
    phase_err_6 = np.ma.masked_all([Tdim, ALdim])
    cor_err = np.ma.masked_all([Tdim, ALdim])

    for tt in range(len(days)):
        for xx in range(len(latitude_use)):
            idir=int(asc_des_use[xx])
            # timing error = alpha[0] * X^0 
            timing_err[tt,xx] = alpha[tt, 0,idir] # * xc1_left[:, xx] # alpha[0] == alpha_timing, alpha[0] * X^0  
            # roll error = alpha[1] * X^1
            roll_err[tt, xx] = alpha[tt, 1,idir] * deltax_use[xx]  # alpha[1] == alpha_roll, alpha[1] * X^1
            # baseline dialation error = alpha[2] * X^2
            baseline_dilation_err[tt, xx] = alpha[tt, 2,idir] * deltax_use[xx]**2 #  alpha[2] == alpha_baseline, alpha[2] * X^2
            # phase error
            H_neg = np.heaviside(-1 * deltax_use[xx], 1) 
            H_pos = np.heaviside(deltax_use[xx], 1)
            phase_err_3[tt, xx] = alpha[tt, 3,idir] * H_neg  
            phase_err_4[tt, xx] = alpha[tt, 4,idir] * deltax_use[xx] * H_neg
            phase_err_5[tt, xx] = alpha[tt, 5,idir] * H_pos 
            phase_err_6[tt, xx] = alpha[tt, 6,idir] * deltax_use[xx] * H_pos
            phase_err[tt, xx] = phase_err_3[tt, xx] + phase_err_4[tt, xx] + phase_err_5[tt, xx] + phase_err_6[tt,xx]
            cor_err[tt,xx] = timing_err[tt,xx] + roll_err[tt,xx] + baseline_dilation_err[tt,xx] + phase_err[tt,xx]
    
    return cor_err 

def build_hswath_matrix2(MSLA, MModes, k_n, l_n, lon,lat,lon_swath, lat_swath, index, T_time, Psi, Rm, day):
    
    '''
    Build H matrix or basis function for Rossby wave model.
    
    Input:
    SSHA_vector: SSH anomalies as a vector,
    Psi (horizontal velocity and pressure structure functions), 
    k_n (zonal wavenumber), 
    l_n (latitudional wavenumber), 
    frequency, 
    longitude, latitude and time. 
    
    Output: H matrix for Rossby wave model
    
    '''
    
    import numpy as np
    
    Phi0 = lat.mean() # central latitude (φ0)
    Omega = 7.27e-5 # Ω is the angular speed of the earth
    Earth_radius = 6.371e6 / 1e5 # meters
    Beta = 2 * Omega * np.cos(Phi0*np.pi/180.) / Earth_radius 
    f0 = 2 * Omega * np.sin(Phi0*np.pi/180.) 

    dlon = lon - lon.mean()
    dlat = lat - lat.mean()
    dlon_swath = lon_swath - lon.mean()
    dlat_swath = lat_swath -lat.mean()
    #print('lon',lon.mean(),'lat',lat.mean())
    M = len(k_n) * len(l_n)

    omega = np.zeros([len(k_n), len(l_n), MModes])
#     day_use = np.zeros(MSLA.shape[2])
    
    count_max = len(index[0])
    # count = 0
    # for tt in range(MSLA.shape[2]):
    #     for jj in range(MSLA.shape[0]):
    #         for ii in range(MSLA.shape[1]):
    #             if (MSLA[jj:jj+1,ii,tt].mask==False):
    #                 SSHA_vector[count] = MSLA[jj, ii, tt]
    #                 day_use[count]=day+tt
    #                 Iindex[count], Jindex[count], Tindex[count] = int(ii), int(jj), int(tt)
    #                 count = count + 1

    # count_max=count
    ndays=MSLA.shape[2]
    H_cos, H_sin = np.zeros([ndays*count_max,M]), np.zeros([ndays*count_max, M])
    H_all = np.zeros([ndays*count_max, M * 2])
    
    # print(count_max, MModes, k_n, l_n)
    nn = 0     
    for kk in range(len(k_n)):
        for ll in range(len(l_n)):
            for mm in range(MModes):
                omega[kk, ll, mm] = Beta * k_n[kk, mm] / (k_n[kk, mm] ** 2 + l_n[ll, mm] ** 2 + Rm[mm] ** -2)          
                idata=0
                for tt in range(MSLA.shape[2]):
                    for count in range(count_max):
                        H_cos[idata, nn] = Psi[0, mm] * np.cos(k_n[kk, mm] * dlon_swath[index[0][count],index[1][count]]+ l_n[ll, mm] * dlat_swath[index[0][count],index[1][count]] + omega[kk, ll, mm] * T_time[int(day+tt)])
                        H_sin[idata, nn] = Psi[0, mm] * np.sin(k_n[kk, mm] * dlon_swath[index[0][count],index[1][count]]+ l_n[ll, mm] * dlat_swath[index[0][count],index[1][count]] + omega[kk, ll, mm] * T_time[int(day+tt)])
                        idata += 1
                nn += 1 
    
    
    H_all[:, 0::2] = H_cos 
    H_all[:, 1::2] = H_sin
    
    return H_all

#-------------------------------------
def read_simulator_error(start_day,ndays,index,index2):
    import numpy as np
    import scipy
    import datetime as dt
    from glob2 import glob
    import xarray as xr

    stats_output=np.zeros([ndays*2,6])
    total_error=np.array([])
    # start_date=dt.date(start_year,start_month,start_day)
    # delta=dt.timedelta(days=1)
    icount=0

# read first ascending and descending files only for latitude, longitude, and cross track distance
    # date_string=start_date.strftime("/home/sgille/swot_simulator/karin/%Y/SWOT_L2_LR_SSH_Expert_001_002_%Y%m%d*.nc")
    # date_string=print('/home/sgille/swot_simulator/karin/%(Year)i/SWOT_L2_LR_SSH_Expert_%(counter)03d_002_*.nc' % {'Year': start_year, 'counter':icount+1}) 
    simulator_files = sorted(glob('/home/sgille/swot_simulator/karin/20??/SWOT_L2_LR_SSH_Expert_' + str(start_day).zfill(3) + '_002*.nc'))
    # simulator_files = sorted(glob(date_string))
    print(simulator_files)
    simulator_ds = xr.open_mfdataset(simulator_files, combine='nested', concat_dim = 'num_lines') # , engine='store', chunks={'time': 10})
        
    # date_string2=start_date.strftime("/home/sgille/swot_simulator/karin/%Y/SWOT_L2_LR_SSH_Expert_001_017_%Y%m%d*.nc")
    # date_string2=print('/home/sgille/swot_simulator/karin/%(Year)i/SWOT_L2_LR_SSH_Expert_%(counter)03d_017_*.nc' % {'Year': start_year, 'counter':icount+1})
    simulator_files2 = sorted(glob('/home/sgille/swot_simulator/karin/20??/SWOT_L2_LR_SSH_Expert_' + str(start_day).zfill(3) + '_017*.nc'))
    #sorted(glob(date_string2))
    simulator_ds2 = xr.open_mfdataset(simulator_files2, combine='nested', concat_dim = 'num_lines') 

    latitude = simulator_ds['latitude'].values   
    longitude = simulator_ds['longitude'].values-360
    cross_track = simulator_ds['cross_track_distance'].values

    latitude2 = simulator_ds2['latitude'].values   
    longitude2 = simulator_ds2['longitude'].values-360
    cross_track2 = simulator_ds2['cross_track_distance'].values

    # sim_err_baseline_dilation = simulator_ds['simulated_error_baseline_dilation'].values
    # sim_err_karin = simulator_ds['simulated_error_karin'].values
    # sim_err_roll = simulator_ds['simulated_error_roll'].values
    # sim_err_phase = simulator_ds['simulated_error_phase'].values
    # sim_err_orbital = simulator_ds['simulated_error_orbital'].values
    # sim_err_timing = simulator_ds['simulated_error_timing'].values
    # # total_error1=sim_err_baseline_dilation + sim_err_karin + sim_err_roll + sim_err_phase + sim_err_orbital + sim_err_timing
    # total_error1=sim_err_baseline_dilation + sim_err_karin + sim_err_roll + sim_err_phase + sim_err_timing
    # stats_output[icount*2,0]=np.std(sim_err_baseline_dilation[index])
    # stats_output[icount*2,1]=np.std(sim_err_karin[index])
    # stats_output[icount*2,2]=np.std(sim_err_roll[index])
    # stats_output[icount*2,3]=np.std(sim_err_phase[index])
    # stats_output[icount*2,4]=np.std(sim_err_orbital[index])
    # stats_output[icount*2,5]=np.std(sim_err_timing[index])

    # sim_err_baseline_dilation2 = simulator_ds2['simulated_error_baseline_dilation'].values
    # sim_err_karin2 = simulator_ds2['simulated_error_karin'].values
    # sim_err_roll2 = simulator_ds2['simulated_error_roll'].values
    # sim_err_phase2 = simulator_ds2['simulated_error_phase'].values
    # sim_err_orbital2 = simulator_ds2['simulated_error_orbital'].values
    # sim_err_timing2 = simulator_ds2['simulated_error_timing'].values
    # # total_error2=sim_err_baseline_dilation2 + sim_err_karin2 + sim_err_roll2 + sim_err_phase2 + sim_err_orbital2 + sim_err_timing2
    # total_error2=sim_err_baseline_dilation2 + sim_err_karin2 + sim_err_roll2 + sim_err_phase2 + sim_err_timing2
    # stats_output[icount*2+1,0]=np.std(sim_err_baseline_dilation2[index])
    # stats_output[icount*2+1,1]=np.std(sim_err_karin2[index])
    # stats_output[icount*2+1,2]=np.std(sim_err_roll2[index])
    # stats_output[icount*2+1,3]=np.std(sim_err_phase2[index])
    # stats_output[icount*2+1,4]=np.std(sim_err_orbital2[index])
    # stats_output[icount*2+1,5]=np.std(sim_err_timing2[index])
    # total_error = np.concatenate((total_error1[index],total_error2[index2]))

 
# now loop through all of the files to read error
    for i in range(0,ndays):
        simulator_files = sorted(glob('/home/sgille/swot_simulator/karin/20??/SWOT_L2_LR_SSH_Expert_' + str(start_day+i).zfill(3) + '_002*.nc'))
        simulator_ds = xr.open_mfdataset(simulator_files, combine='nested', concat_dim = 'num_lines') # , engine='store', chunks={'time': 10})
        
        simulator_files2 = sorted(glob('/home/sgille/swot_simulator/karin/20??/SWOT_L2_LR_SSH_Expert_' + str(start_day+i).zfill(3) + '_017*.nc'))
        simulator_ds2 = xr.open_mfdataset(simulator_files2, combine='nested', concat_dim = 'num_lines') 

        sim_err_baseline_dilation = simulator_ds['simulated_error_baseline_dilation'].values
        sim_err_karin = simulator_ds['simulated_error_karin'].values
        sim_err_roll = simulator_ds['simulated_error_roll'].values
        sim_err_phase = simulator_ds['simulated_error_phase'].values
        sim_err_orbital = simulator_ds['simulated_error_orbital'].values
        sim_err_timing = simulator_ds['simulated_error_timing'].values
        # total_error1=sim_err_baseline_dilation + sim_err_karin + sim_err_roll + sim_err_phase + sim_err_orbital + sim_err_timing
        total_error1=sim_err_baseline_dilation + sim_err_karin + sim_err_roll + sim_err_phase + sim_err_timing
        stats_output[i*2,0]=np.std(sim_err_baseline_dilation[index])
        stats_output[i*2,1]=np.std(sim_err_karin[index])
        stats_output[i*2,2]=np.std(sim_err_roll[index])
        stats_output[i*2,3]=np.std(sim_err_phase[index])
        stats_output[i*2,4]=np.std(sim_err_orbital[index])
        stats_output[i*2,5]=np.std(sim_err_timing[index])           
        
        sim_err_baseline_dilation2 = simulator_ds2['simulated_error_baseline_dilation'].values
        sim_err_karin2 = simulator_ds2['simulated_error_karin'].values
        sim_err_roll2 = simulator_ds2['simulated_error_roll'].values
        sim_err_phase2 = simulator_ds2['simulated_error_phase'].values
        sim_err_orbital2 = simulator_ds2['simulated_error_orbital'].values
        sim_err_timing2 = simulator_ds2['simulated_error_timing'].values
        # total_error2=sim_err_baseline_dilation2 + sim_err_karin2 + sim_err_roll2 + sim_err_phase2 + sim_err_orbital2 + sim_err_timing2
        total_error2=sim_err_baseline_dilation2 + sim_err_karin2 + sim_err_roll2 + sim_err_phase2 + sim_err_timing2
        stats_output[i*2+1,0]=np.std(sim_err_baseline_dilation2[index2])
        stats_output[i*2+1,1]=np.std(sim_err_karin2[index2])
        stats_output[i*2+1,2]=np.std(sim_err_roll2[index2])
        stats_output[i*2+1,3]=np.std(sim_err_phase2[index2])
        stats_output[i*2+1,4]=np.std(sim_err_orbital2[index2])
        stats_output[i*2+1,5]=np.std(sim_err_timing2[index2])
        total_error = np.concatenate((total_error,total_error1[index],total_error2[index2]))

    return total_error, stats_output


#     day_use = np.zeros(MSLA.shape[2])
    
    count_max = len(lon_swath)
    # count = 0
    # for tt in range(MSLA.shape[2]):
    #     for jj in range(MSLA.shape[0]):
    #         for ii in range(MSLA.shape[1]):
    #             if (MSLA[jj:jj+1,ii,tt].mask==False):
    #                 SSHA_vector[count] = MSLA[jj, ii, tt]
    #                 day_use[count]=day+tt
    #                 Iindex[count], Jindex[count], Tindex[count] = int(ii), int(jj), int(tt)
    #                 count = count + 1

    # count_max=count
    ndays=MSLA.shape[2]
    H_cos, H_sin = np.zeros([ndays*count_max,M]), np.zeros([ndays*count_max, M])
    H_all = np.zeros([ndays*count_max, M * 2])
    
    # print(count_max, MModes, k_n, l_n)
    nn = 0     
    for kk in range(len(k_n)):
        for ll in range(len(l_n)):
            for mm in range(MModes):
                omega[kk, ll, mm] = Beta * k_n[kk, mm] / (k_n[kk, mm] ** 2 + l_n[ll, mm] ** 2 + Rm[mm] ** -2)          
                idata=0
                for tt in range(MSLA.shape[2]):
                    for count in range(count_max):
                        H_cos[idata, nn] = Psi[0, mm] * np.cos(k_n[kk, mm] * dlon_swath[count]+ l_n[ll, mm] * dlat_swath[count] + omega[kk, ll, mm] * T_time[int(day+tt)])
                        H_sin[idata, nn] = Psi[0, mm] * np.sin(k_n[kk, mm] * dlon_swath[count]+ l_n[ll, mm] * dlat_swath[count] + omega[kk, ll, mm] * T_time[int(day+tt)])
                        idata += 1
                nn += 1 

#--------------------------------------------------------------
def build_hswath_matrix2_for_simulator(MSLA, MModes, k_n, l_n, lon,lat,lon_swath, lat_swath, index, T_time, Psi, Rm, day):
    
    '''
    Build H matrix or basis function for Rossby wave model.
    
    Input:
    SSHA_vector: SSH anomalies as a vector,
    Psi (horizontal velocity and pressure structure functions), 
    k_n (zonal wavenumber), 
    l_n (latitudional wavenumber), 
    frequency, 
    longitude, latitude and time. 
    
    Output: H matrix for Rossby wave model
    
    '''
    
    import numpy as np
    
    Phi0 = lat.mean() # central latitude (φ0)
    Omega = 7.27e-5 # Ω is the angular speed of the earth
    Earth_radius = 6.371e6 / 1e5 # meters
    Beta = 2 * Omega * np.cos(Phi0*np.pi/180.) / Earth_radius 
    f0 = 2 * Omega * np.sin(Phi0*np.pi/180.) 

    dlon = lon - lon.mean()
    dlat = lat - lat.mean()
    dlon_swath = lon_swath - lon.mean()
    dlat_swath = lat_swath -lat.mean()
    #print('lon',lon.mean(),'lat',lat.mean())
    M = len(k_n) * len(l_n)

    omega = np.zeros([len(k_n), len(l_n), MModes])
#     day_use = np.zeros(MSLA.shape[2])
    
    count_max = len(lon_swath)

    ndays=MSLA.shape[2]
    H_cos, H_sin = np.zeros([ndays*count_max,M]), np.zeros([ndays*count_max, M])
    H_all = np.zeros([ndays*count_max, M * 2])
    
    # print(count_max, MModes, k_n, l_n)
    nn = 0     
    for kk in range(len(k_n)):
        for ll in range(len(l_n)):
            for mm in range(MModes):
                omega[kk, ll, mm] = Beta * k_n[kk, mm] / (k_n[kk, mm] ** 2 + l_n[ll, mm] ** 2 + Rm[mm] ** -2)          
                idata=0
                for tt in range(MSLA.shape[2]):
                    for count in range(count_max):
                        H_cos[idata, nn] = Psi[0, mm] * np.cos(k_n[kk, mm] * dlon_swath[count]+ l_n[ll, mm] * dlat_swath[count] + omega[kk, ll, mm] * T_time[int(day+tt)])
                        H_sin[idata, nn] = Psi[0, mm] * np.sin(k_n[kk, mm] * dlon_swath[count]+ l_n[ll, mm] * dlat_swath[count] + omega[kk, ll, mm] * T_time[int(day+tt)])
                        idata += 1
                nn += 1 
    
    omega = np.zeros([len(k_n), len(l_n), MModes])  
    
    H_all[:, 0::2] = H_cos 
    H_all[:, 1::2] = H_sin
    
    return H_all