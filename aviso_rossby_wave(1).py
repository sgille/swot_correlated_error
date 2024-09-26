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


def inversion(Y, H_v, R_over_P):
    
    '''
    Solve for X given observations (Y), basis function (H_v) and signal to noise ratio (P_over_R).
    Return: X (amplitudes of Rossby waves)
    This is all in model space.
    '''
    
    import numpy as np
    from numpy import linalg as LA

    HTH = np.matmul(H_v.T, H_v)
    
    HTH = HTH +  R_over_P #, P: uncertainty in model, R: uncertainty in data, actually R_over_P
    
    D = np.matmul(LA.inv(HTH), H_v.T)
    
    amp = np.matmul(D, Y)
    
    Y_estimated = np.matmul(H_v, amp)
    
    return amp, Y_estimated

def inversion2(Y, H_v, P_over_R):
    
    '''
    Solve for X given observations (Y), basis function (H_v) and signal to noise ratio (P_over_R).
    Return: X (amplitudes of Rossby waves)
    This is all in model space.
    '''
    
    import numpy as np
    from numpy import linalg as LA

    #R_factor = 0.01 ** 2 # 1 cm noise
    
    R_factor = 0.001 ** 2 # .1 cm noise
    
    R_factor_inv = 1 / R_factor
    
    HTH = np.matmul(H_v.T, H_v) * R_factor_inv 
    
    P_diag = P_over_R.diagonal()
    
    P_over_R_inv = np.zeros(P_over_R.shape)
    
    np.fill_diagonal(P_over_R_inv, P_diag ** -1)
    
    HTH = HTH +  P_over_R_inv #, P: uncertainty in model, R: uncertainty in data, actually R_over_P
    
    D = np.matmul(LA.inv(HTH), H_v.T * R_factor_inv) 
    
    amp = np.matmul(D, Y)
    
    Y_estimated = np.matmul(H_v, amp)
    
    return amp, Y_estimated

def forecast_ssh(MSLA, amp, H_all):
    
    '''
    Make SSH predictions with the estimated Rossby wave amplutudes.
    Input: timestamp, estimated amplitudes, True AVISO SSH anomalies and H matrix (basis functions).
    '''
    
    import numpy as np
    from tqdm import tqdm
    from numpy import linalg as LA
    from scipy import linalg
    
    # forecast SSH
    SSHA_predicted = np.matmul(H_all, amp)
    
    time_vector = np.zeros(MSLA.size)
    lon_vector, lat_vector = np.zeros(MSLA.size),np.zeros(MSLA.size)
    Iindex, Jindex, Tindex = np.zeros(MSLA.size), np.zeros(MSLA.size), np.zeros(MSLA.size)
    
    SSHA_vector = np.zeros(MSLA.size)
    
    # flatten SSH
    count = 0
    for ii in range(MSLA.shape[0]):
        for jj in range(MSLA.shape[1]):
            for tt in range(MSLA.shape[2]):
                if(MSLA[ii, jj, tt] != np.nan): 
                    SSHA_vector[count] = MSLA[ii, jj, tt]
                    count = count + 1
                    
                    
    #print(count)
    # calculate residual variance
    residual = SSHA_vector - SSHA_predicted

    # evaluate skill (1- rms_residual/rms_ssha_vector) and store the skill
    # skill value nn, ll, mm, = skill value
    #
    residual_iter = (np.mean(residual**2)) / (np.mean(SSHA_vector**2))
    
    return SSHA_predicted, SSHA_vector, residual_iter


def reverse_vector(True_MSLA, SSHA_predicted):
    
    '''
    Reverse the vectorization.
    '''
    
    import numpy as np
    
    MSLA_est = np.zeros(True_MSLA.shape)
    
    count = 0
    for ii in range(True_MSLA.shape[0]):
        for jj in range(True_MSLA.shape[1]):
            for tt in range(True_MSLA.shape[2]):
                #if(True_MSLA[ii, jj, tt] != np.nan):
                MSLA_est[ii, jj, tt] = SSHA_predicted[count]
                count += 1
                    
    return MSLA_est


def build_h_matrix(MSLA, MModes, k_n, l_n, lon, lat, T_time, Psi, Rm, day):
    
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
    #print('lon',lon.mean(),'lat',lat.mean())
    M = len(k_n) * len(l_n)
    H_cos, H_sin = np.zeros([MSLA.size, M]), np.zeros([MSLA.size, M])
    H_all = np.zeros([MSLA.size, M * 2])
    omega = np.zeros([len(k_n), len(l_n), MModes])
    Iindex, Jindex, Tindex = np.zeros(MSLA.size), np.zeros(MSLA.size), np.zeros(MSLA.size)
    day_use = np.zeros(MSLA.size)
    SSHA_vector = np.zeros(MSLA.size)
    
    count = 0
    for tt in range(MSLA.shape[2]):
        for jj in range(MSLA.shape[0]):
            for ii in range(MSLA.shape[1]):
                SSHA_vector[count] = MSLA[jj, ii, tt]
                day_use[count]=day+tt
                Iindex[count], Jindex[count], Tindex[count] = int(ii), int(jj), int(tt)
                count = count + 1

    nn = 0 
    for kk in range(len(k_n)):
        for ll in range(len(l_n)):
            for mm in range(MModes):
                omega[kk, ll, mm] = Beta * k_n[kk, mm] / (k_n[kk, mm] ** 2 + l_n[ll, mm] ** 2 + Rm[mm] ** -2)
                for count in range(len(Iindex)):
                    H_cos[count, nn] = Psi[0, mm] * np.cos(k_n[kk, mm] * dlon[int(Iindex[count])] + l_n[ll, mm] * dlat[int(Jindex[count])] + omega[kk, ll, mm] * T_time[int(day_use[count])])
                    H_sin[count, nn] = Psi[0, mm] * np.sin(k_n[kk, mm] * dlon[int(Iindex[count])] + l_n[ll, mm] * dlat[int(Jindex[count])] + omega[kk, ll, mm] * T_time[int(day_use[count])])
                nn += 1
                
    H_all[:, 0::2] = H_cos 
    H_all[:, 1::2] = H_sin
    
    return H_all
    
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


def build_swath(swath_width, x_width, day, lon, lat):
    
    '''
     Generate the x, y, t indices for multiple satellite passings over a given swath width and time period. 
    
    '''
    
    import numpy as np
    
    x_width = len(lon)
    
    # swath 1

    xswath_index0 = np.arange(0, x_width , 1) 
    yswath_index0 = np.arange(0, swath_width, 1)
    yswath_index_left = np.ma.masked_all([x_width, swath_width])
    xswath_index_left = np.ma.masked_all([x_width, swath_width])
    for yy in range(swath_width):
        xswath_index_left[:, yy] = xswath_index0
        
    for xx in range(x_width):
        yswath_index_left[xx] = yswath_index0 + xx
        
    yswath_index_left = np.ma.masked_outside(yswath_index_left, 0, len(lat) - 1)
    xswath_index_left = np.ma.masked_outside(xswath_index_left, 0, len(lon) - 1)
    y_mask_left = np.ma.getmask(yswath_index_left)
    x_mask_left  = np.ma.getmask(xswath_index_left)
    mask_left = np.ma.mask_or(y_mask_left,x_mask_left)
    xswath_index_left = np.ma.MaskedArray(xswath_index_left, mask_left)
    yswath_index_left = np.ma.MaskedArray(yswath_index_left, mask_left)
    
    # swath 2

    xswath_index1 = np.arange(len(lon) - x_width, len(lon))
    yswath_index1 = np.arange(len(lat) - swath_width, len(lat))
    yswath_index_right = np.ma.masked_all([x_width, swath_width])
    xswath_index_right = np.ma.masked_all([x_width, swath_width])

    for yy in range(swath_width):
        xswath_index_right[:, yy] = xswath_index1
        
    for xx in range(x_width):    
        yswath_index_right[xx] = yswath_index1 - xx  
        
    yswath_index_right = np.ma.masked_outside(yswath_index_right, 0, len(lat) - 1)
    xswath_index_right = np.ma.masked_outside(xswath_index_right, 0, len(lon) - 1)
    y_mask_right = np.ma.getmask(yswath_index_right)
    x_mask_right = np.ma.getmask(xswath_index_right)
    mask_right = np.ma.mask_or(y_mask_right,x_mask_right)
    xswath_index_right = np.ma.MaskedArray(xswath_index_right, mask_right)
    yswath_index_right = np.ma.MaskedArray(yswath_index_right, mask_right)

    yvalid_index = np.append(yswath_index_left.compressed().astype(int), yswath_index_right.compressed().astype(int)) 
    xvalid_index = np.append(xswath_index_left.compressed().astype(int), xswath_index_right.compressed().astype(int))
    
    tindex, xindex, yindex = [], [], []
    xindex =  np.tile(xvalid_index, len(day))
    yindex =  np.tile(yvalid_index, len(day))
    for dd in day:
        tmp = np.tile(dd, len(yvalid_index))
        tindex = np.append(tindex, tmp)
    
    return xindex, yindex, tindex, yswath_index_left, yswath_index_right, mask_left, mask_right




def make_error(days, alpha, yswath_index_left, yswath_index_right, y_mask_left, y_mask_right):
    
    '''
    This function models the time-varying error parameters in satellite swath data, including timing error, roll error, baseline dilation error, and phase error. The roll errors and baseline dilation errors are assumed to be correlated and are generated on the satellite swath. The function takes as inputs the number of days the data is repeated, the model parameters of the roll errors and baseline dilation errors, and the swath index for swath 1 and 2, as well as the swath masks.

The output of the function includes the valid data points of the timing error, roll errors, baseline dilation error, and phase error, as well as the valid coordinates as the distance from the center of the swath ("xc1_valid") and the quadratic of that distance ("xc2_valid").
    '''
    import numpy as np
    
    # timing error
    timing_err_left, timing_err_right = np.ma.masked_all(yswath_index_left.shape), np.ma.masked_all(yswath_index_right.shape)
    # Roll error
    roll_err_left, roll_err_right = np.ma.masked_all(yswath_index_left.shape), np.ma.masked_all(yswath_index_right.shape)
    # Baseline dilation error
    baseline_dilation_err_left, baseline_dilation_err_right = np.ma.masked_all(yswath_index_left.shape), np.ma.masked_all(yswath_index_right.shape)
    # phase error
    phase_err_left, phase_err_right = np.ma.masked_all(yswath_index_left.shape), np.ma.masked_all(yswath_index_right.shape)
    phase_err_left3, phase_err_right3 = np.ma.masked_all(yswath_index_left.shape), np.ma.masked_all(yswath_index_right.shape)
    phase_err_left4, phase_err_right4 = np.ma.masked_all(yswath_index_left.shape), np.ma.masked_all(yswath_index_right.shape)
    phase_err_left5, phase_err_right5 = np.ma.masked_all(yswath_index_left.shape), np.ma.masked_all(yswath_index_right.shape)
    phase_err_left6, phase_err_right6 = np.ma.masked_all(yswath_index_left.shape), np.ma.masked_all(yswath_index_right.shape)
    al, ac = roll_err_left.shape
    xc = (ac-1) / 2
    
    # swath 1
    xc1_left, xc2_left = np.ma.masked_all(yswath_index_left.shape), np.ma.masked_all(yswath_index_left.shape)
    H_neg_left, H_pos_left = np.ma.masked_all(yswath_index_left.shape), np.ma.masked_all(yswath_index_left.shape)

    for xx in np.arange(ac):
        xc1_left[:, xx] = (xx - xc)  #* .25     #.25 degree resolution
        xc2_left[:, xx] = (xx - xc)  ** 2  #* .25
        # timing error = alpha[0] * X^0 
        timing_err_left[:, xx] = alpha[0] # * xc1_left[:, xx] # alpha[0] == alpha_timing, alpha[0] * X^0  
        # roll error = alpha[1] * X^1
        roll_err_left[:, xx] = alpha[1] * xc1_left[:, xx]  # alpha[1] == alpha_roll, alpha[1] * X^1
        # baseline dialation error = alpha[2] * X^2
        baseline_dilation_err_left[:, xx] = alpha[2] * xc2_left[:, xx] #  alpha[2] == alpha_baseline, alpha[2] * X^2
        # phase error
        H_neg_left = np.heaviside(-1 * xc1_left[:, xx], 1) 
        H_pos_left = np.heaviside(xc1_left[:, xx], 1)
        phase_err_left3[:, xx] = alpha[3] * H_neg_left  
        phase_err_left4[:, xx] = alpha[4] * xc1_left[:, xx] * H_neg_left
        phase_err_left5[:, xx] = alpha[5] * H_pos_left 
        phase_err_left6[:, xx] = alpha[6] * xc1_left[:, xx] * H_pos_left
        phase_err_left[:, xx] = phase_err_left3[:, xx] + phase_err_left4[:, xx] + phase_err_left5[:, xx] + phase_err_left6[:, xx]

    # swath 2
    xc1_right, xc2_right = np.ma.masked_all(yswath_index_right.shape), np.ma.masked_all(yswath_index_right.shape)
    H_neg_right, H_pos_right = np.ma.masked_all(yswath_index_right.shape), np.ma.masked_all(yswath_index_right.shape)

    for xx in np.arange(ac):
        xc1_right[:, xx] = (xx - xc) #* .25 #.25 degree resolution, 1deg longitude ~ 85km * .85e5
        xc2_right[:, xx] = (xx - xc)  ** 2  # * .25  #.25 degree resolution
        # timing error = alpha[0] * X^0 #IND = -7
        timing_err_right[:, xx] = alpha[0] # * xc1_right[:, xx] # alpha[0] == alpha_timing
        # roll error = alpha[1] * X^1 #IND = -6
        roll_err_right[:, xx] = alpha[1] * xc1_right[:, xx]
        # baseline dialation error # -5
        baseline_dilation_err_right[:, xx] = alpha[2] * xc2_right[:, xx]
        # phase error = alpha[2] * X^2
        H_neg_right[:, xx] = np.heaviside(-1 * xc1_right[:, xx], 1)
        H_pos_right[:, xx] = np.heaviside(xc1_right[:, xx], 1) 
        # phase error
        phase_err_right3[:, xx] = alpha[3] * H_neg_right[:, xx] # IND =-4   
        phase_err_right4[:, xx] = alpha[4] * xc1_right[:, xx] * H_neg_right[:, xx] # IND =-3
        phase_err_right5[:, xx] = alpha[5] * H_pos_right[:, xx] # -2
        phase_err_right6[:, xx] = alpha[6] * xc1_right[:, xx] * H_pos_right[:, xx] # IND = -1
        phase_err_right[:, xx] = phase_err_right3[:, xx] + phase_err_right4[:, xx] + phase_err_right5[:, xx] + phase_err_right6[:, xx]


    roll_err_left_masked = np.ma.MaskedArray(roll_err_left, y_mask_left)
    roll_err_right_masked = np.ma.MaskedArray(roll_err_right, y_mask_right)
    timing_err_left_masked = np.ma.MaskedArray(timing_err_left, y_mask_left)
    timing_err_right_masked = np.ma.MaskedArray(timing_err_right, y_mask_right)
    baseline_dilation_err_left_masked = np.ma.MaskedArray(baseline_dilation_err_left, y_mask_left)
    baseline_dilation_err_right_masked = np.ma.MaskedArray(baseline_dilation_err_right, y_mask_right)
    phase_err_left_masked = np.ma.MaskedArray(phase_err_left, y_mask_left)
    phase_err_right_masked = np.ma.MaskedArray(phase_err_right, y_mask_right)
    xc1_left_masked = np.ma.MaskedArray(xc1_left, y_mask_left)
    xc2_left_masked = np.ma.MaskedArray(xc2_left, y_mask_left)
    xc1_right_masked = np.ma.MaskedArray(xc1_right, y_mask_right)
    xc2_right_masked = np.ma.MaskedArray(xc2_right, y_mask_right)
    
    timing_err_left_valid = timing_err_left_masked.compressed() # retrieve the valid data 
    timing_err_right_valid = timing_err_right_masked.compressed() # retrieve the valid data 
    roll_err_left_valid = roll_err_left_masked.compressed() # retrieve the valid data 
    roll_err_right_valid = roll_err_right_masked.compressed() # retrieve the valid data 
    baseline_dilation_err_left_valid = baseline_dilation_err_left_masked.compressed() # retrieve the valid data 
    baseline_dilation_err_right_valid = baseline_dilation_err_right_masked.compressed() # retrieve the valid data 
    phase_err_left_valid = phase_err_left_masked.compressed() # retrieve the valid data 
    phase_err_right_valid = phase_err_right_masked.compressed() # retrieve the valid data     
    xc1_left_valid = xc1_left_masked.compressed() # retrieve the valid data 
    xc2_left_valid = xc2_left_masked.compressed() # retrieve the valid data 
    xc1_right_valid = xc1_right_masked.compressed() # retrieve the valid data 
    xc2_right_valid = xc2_right_masked.compressed() # retrieve the valid data 
    
    # concat left and right swath
    
    timing_err_valid_index = np.append(timing_err_left_valid, timing_err_right_valid) 
    roll_err_valid_index = np.append(roll_err_left_valid, roll_err_right_valid) 
    baseline_dilation_err_index = np.append(baseline_dilation_err_left_valid, baseline_dilation_err_right_valid)
    phase_err_valid_index = np.append(phase_err_left_valid, phase_err_right_valid)
    xc1_index = np.append(xc1_left_valid, xc1_right_valid)
    xc2_index = np.append(xc2_left_valid, xc2_right_valid)
    
    # repeat errors for "days"

    roll_err_valid = np.repeat(roll_err_valid_index, len(days))
    timing_err_valid = np.repeat(timing_err_valid_index, len(days)) 
    baseline_dilation_err_valid = np.repeat(baseline_dilation_err_index, len(days)) 
    phase_err_valid = np.repeat(phase_err_valid_index, len(days)) 
    xc1_valid = np.repeat(xc1_index, len(days))
    xc2_valid = np.repeat(xc2_index, len(days))
    
    
    return timing_err_valid, roll_err_valid, baseline_dilation_err_valid, phase_err_valid, phase_err_left3, phase_err_left4, phase_err_left5, phase_err_left6, xc1_valid, xc2_valid 



def calculate_errors(Tdim, Valid_points, ssh_estimated_swath, ssh, cor_err, err_estimated_swath, MSLA_swath, xvalid_index, yvalid_index, lon, lat, date_time):
    import matplotlib.pyplot as plt
    import cmocean as cmo
    import numpy as np
    
    ssh_diff = ssh_estimated_swath - ssh
    err_diff = np.sqrt(np.mean((cor_err - err_estimated_swath)**2, axis=1)) / np.sqrt(np.mean(cor_err**2, axis=1))
    ssh_diff_percent = np.sqrt(np.mean(ssh_diff**2, axis=1)) / np.sqrt(np.mean(ssh**2, axis=1))
    err_map = np.zeros([Tdim, len(lon), len(lat)])
    ssh_map = np.zeros([Tdim, len(lon), len(lat)])
    ssh_true = np.zeros([Tdim, len(lon), len(lat)])
    err_true = np.zeros([Tdim, len(lon), len(lat)])
    msla_obs = np.zeros([Tdim, len(lon), len(lat)])

    for tt in range(Tdim):
        for ii in range(Valid_points):
            err_map[tt, xvalid_index[ii], yvalid_index[ii]] = err_estimated_swath[tt, ii]
            ssh_map[tt, xvalid_index[ii], yvalid_index[ii]] = ssh_estimated_swath[tt, ii]
            err_true[tt, xvalid_index[ii], yvalid_index[ii]] = cor_err[tt, ii]
            ssh_true[tt, xvalid_index[ii], yvalid_index[ii]] = ssh[tt, ii]
            msla_obs[tt, xvalid_index[ii], yvalid_index[ii]] = MSLA_swath[tt, ii]

        err_diff1 = np.sqrt(np.mean((err_true[tt] - err_map[tt])**2)) / np.sqrt(np.mean(err_true[tt]**2))
        ssh_diff1 = np.sqrt(np.mean((ssh_true[tt] - ssh_map[tt])**2)) / np.sqrt(np.mean(ssh_true[tt]**2))

    return err_diff, ssh_diff_percent, err_map, ssh_map, err_true, ssh_true


def plot_time_series(tt, ssh, MSLA_swath, ssh_estimated_swath, ssh_diff, ssh_diff_percent, cor_err, err_estimated_swath, err_diff):
    
    import matplotlib.pyplot as plt
    import cmocean as cmo

    plt.figure(figsize=(10, 8))

    plt.subplot(311)
    plt.plot(ssh[tt], 'b-', label='True SSH')
    plt.plot(MSLA_swath[tt], 'g', label = 'True SSH + Error')
    plt.plot(ssh_estimated_swath[tt], 'r--', label='Estimated SSH')
    plt.xlabel('Data point', fontsize=14)
    plt.ylabel('SSH (m)', fontsize=14)
    plt.title('SSH estimation', fontsize=16)
    plt.legend(fontsize=12)

    plt.subplot(312)
    plt.plot(ssh_diff[tt], 'b', label='True SSH - SSH estimate')
    plt.xlabel('Data point', fontsize=14)
    plt.ylabel('SSH (m)', fontsize=14)
    plt.title('True SSH - SSH estimate, ' + str(ssh_diff_percent[tt] * 100)[:5] + '%', fontsize=16)
    plt.legend(fontsize=12)
    plt.tight_layout()

    plt.subplot(313)
    plt.plot(cor_err[tt], 'b-', label='True error')
    plt.plot(err_estimated_swath[tt], 'r--', label='Estimated error')
    plt.legend(fontsize=12)
    plt.title('True error - error estimate, ' + str(err_diff[tt] * 100)[:5] + '%', fontsize=16)

def plot_ssh_err_estimate(tt, date_time, ssh, MSLA_swath, ssh_estimated_swath, ssh_diff, ssh_diff_percent, cor_err, err_estimated_swath, err_diff):
    
    import matplotlib.pyplot as plt
    import cmocean as cmo

    plot_time_series(tt, ssh, MSLA_swath, ssh_estimated_swath, ssh_diff, ssh_diff_percent, cor_err, err_estimated_swath, err_diff)
    plt.savefig('./ssh_err_estimate/ssh_err_parameter_'+  str(date_time[tt])[:10] +'.png')
    #plt.close()

def plot_ssh_err_est_maps(tt, day0, day1, lon, lat, err_true, ssh_true, ssh_map, err_map, err_diff, ssh_diff):
    
    import matplotlib.pyplot as plt
    import cmocean as cmo
    
    fig = plt.figure(figsize = (10, 12))

    plt.subplot(321)
    plt.pcolormesh(lon, lat, err_true+ ssh_true, cmap = cmo.cm.balance, vmin = -.2, vmax = .2)
    plt.colorbar()
    plt.xlabel('Longitude (\N{DEGREE SIGN}W)', fontsize = 14)
    plt.ylabel('Latitude (\N{DEGREE SIGN}N)', fontsize = 14)
    plt.title('a) True error + true SSH', fontsize = 14)

    plt.subplot(323)
    plt.pcolormesh(lon, lat, ssh_map, vmin = -.2, vmax = .2, cmap = cmo.cm.balance)
    plt.colorbar()
    plt.xlabel('Longitude (\N{DEGREE SIGN}W)', fontsize = 14)
    plt.ylabel('Latitude (\N{DEGREE SIGN}N)', fontsize = 14)
    plt.title('c) SSH estimate, 1-stage ' + str((1- ssh_diff[tt]) * 100)[:5] + '%', fontsize = 14)

    plt.subplot(325)
    plt.pcolormesh(lon, lat, ssh_true - ssh_map, cmap = cmo.cm.balance, vmin = -.2, vmax = .2)
    plt.colorbar()
    plt.xlabel('Longitude (\N{DEGREE SIGN}W)', fontsize = 14)
    plt.ylabel('Latitude (\N{DEGREE SIGN}N)', fontsize = 14)
    plt.title('e) True SSH - SSH estimate, ' + str(ssh_diff[tt] * 100)[:5] + '%', fontsize = 14)

    plt.subplot(322)
    plt.pcolormesh(lon, lat, err_true, cmap = cmo.cm.balance, vmin = -.2, vmax = .2)
    plt.colorbar()
    plt.xlabel('Longitude (\N{DEGREE SIGN}W)', fontsize = 14)
    plt.ylabel('Latitude (\N{DEGREE SIGN}N)', fontsize = 14)
    plt.title('b) True error', fontsize = 14)

    plt.subplot(324)
    plt.pcolormesh(lon, lat, err_map, cmap = cmo.cm.balance, vmin = -.2, vmax = .2)
    plt.colorbar()
    plt.xlabel('Longitude (\N{DEGREE SIGN}W)', fontsize = 14)
    plt.ylabel('Latitude (\N{DEGREE SIGN}N)', fontsize = 14)
    plt.title('d) Error estimate ' +  str((1- err_diff[tt]) * 100)[:4] + '%', fontsize = 14)

    plt.subplot(326)
    plt.pcolormesh(lon, lat, err_true - err_map, cmap = cmo.cm.balance, vmin = -.2, vmax = .2)
    plt.colorbar()
    plt.xlabel('Longitude (\N{DEGREE SIGN}W)', fontsize = 14)
    plt.ylabel('Latitude (\N{DEGREE SIGN}N)', fontsize = 14)
    plt.title('f) True Error - Error estimate, ' + str(err_diff[tt] * 100)[:4] + '%' , fontsize = 14)

    plt.tight_layout()
    plt.savefig('ssh_err_est_maps/aviso_ssh_estimate_'+ str(date_time[tt])[:10] +
                '_1step_' + str(int(day1 - day0)*5) +'day.png', dpi = 300)
    #plt.close()

def calculate_ssh_err_diff(Tdim, Valid_points, ssh_estimated_swath, ssh, err_est_1step, cor_err):

    import numpy as np

    ssh_diff = np.zeros([Tdim, Valid_points])
    err_diff = np.zeros([Tdim, Valid_points])
    ssh_diff_percent, err_diff_percent = np.zeros([Tdim]), np.zeros([Tdim])

    for tt in range(Tdim):
        ssh_diff[tt] = ssh_estimated_swath[tt * Valid_points : (tt+1) * Valid_points] - ssh[tt * Valid_points : (tt+1) * Valid_points]
        err_diff[tt] = err_est_1step[tt * Valid_points : (tt+1) * Valid_points] - cor_err[tt * Valid_points : (tt+1) * Valid_points]
        ssh_diff_percent[tt] = np.sqrt(ssh_diff[tt]**2).mean() / np.sqrt(ssh[tt * Valid_points : (tt+1) * Valid_points]**2).mean()
        err_diff_percent[tt]  = np.sqrt(np.mean(err_diff[tt]**2).mean()) / np.sqrt(cor_err[tt * Valid_points : (tt+1) * Valid_points]**2).mean()

    return ssh_diff, err_diff, ssh_diff_percent, err_diff_percent

def remap_to_2d_map(tt, Valid_points, lon, lat, ssh_estimated_swath, ssh,  err_est_1step, cor_err, xvalid_index, yvalid_index):

    import numpy as np
    
    err_map = np.zeros([len(lon), len(lat)])
    ssh_map = np.zeros([len(lon), len(lat)])
    ssh_true = np.zeros([len(lon), len(lat)])
    err_true = np.zeros([len(lon), len(lat)])
    #msla_obs = np.zeros([len(lon), len(lat)])

    for ii in range(Valid_points):
        err_map[xvalid_index[ii], yvalid_index[ii]] = err_est_1step[tt * Valid_points : (tt+1) * Valid_points][ii]
        ssh_map[xvalid_index[ii], yvalid_index[ii]]  = ssh_estimated_swath[tt * Valid_points : (tt+1) * Valid_points][ii]
        err_true[xvalid_index[ii], yvalid_index[ii]] = cor_err[tt * Valid_points : (tt+1) * Valid_points][ii]
        ssh_true[xvalid_index[ii], yvalid_index[ii]] = ssh[tt * Valid_points : (tt+1) * Valid_points][ii]
        #msla_obs[xvalid_index[ii], yvalid_index[ii]] = MSLA_swath[tt * Valid_points : (tt+1) * Valid_points][ii]

    return ssh_map, ssh_true, err_map, err_true#, msla_obs



def build_SWOT_swath(msla, lon, lat, swot_longitude, swot_latitude, day):
    
    import numpy as np
    
   
    # Define the bounds for the California Current System (CCS)
    cc_bounds = {
        "lon": [360-135, 360-115],
        "lat": [30, 40]
    }

    swot_latitude=latitude
    swot_longitude=longitude
    # Filter SWOT data points within the CCS bounds
    valid_swot_mask = (swot_latitude >= cc_bounds["lat"][0]) & (swot_latitude <= cc_bounds["lat"][1]) & \
                      (swot_longitude >= cc_bounds["lon"][0]) & (swot_longitude <= cc_bounds["lon"][1])
    #swot_longitude = swot_longitude[valid_swot_mask]
    #swot_latitude = swot_latitude[valid_swot_mask]
    
    lon_bound1a=np.zeros(len(lat))
    lon_bound1b=np.zeros(len(lat))
    lon_nadir=np.zeros(len(lat))
    lon_bound2a=np.zeros(len(lat))
    lon_bound2b=np.zeros(len(lat))

    #  bounds are set based on fixed grid, assuming 2 50 km swaths, with 20 km between swaths.  
    #  This discards points at edges of swath
    # These limits could be adjusted to incorporate more AVISO data
    index=np.argsort(swot_latitude[:,5])    
    lon_bound1a=np.interp(lat,swot_latitude[index,5],swot_longitude[index,5])
    index=np.argsort(swot_latitude[:,30]) 
    lon_bound1b=np.interp(lat,swot_latitude[index,30],swot_longitude[index,30])
    index=np.argsort(swot_latitude[:,35]) 
    lon_nadir=np.interp(lat,swot_latitude[index,35],swot_longitude[index,35])
    index=np.argsort(swot_latitude[:,40]) 
    lon_bound2a=np.interp(lat,swot_latitude[index,40],swot_longitude[index,40])
    index=np.argsort(swot_latitude[:,65]) 
    lon_bound2b=np.interp(lat,swot_latitude[index,65],swot_longitude[index,65])
    
    # Generate a meshgrid from the lon and lat arrays
    LON, LAT = np.meshgrid(lon, lat)
    
    yswath=np.ma.masked_all(msla.shape)
    xswath=np.ma.masked_all(msla.shape)
    msla_use=np.ma.masked_all(msla.shape)
    mask_test=np.ma.MaskedArray(LON,False)
    yswath=np.ma.MaskedArray(LAT,False)
    xswath=np.ma.MaskedArray(LON,False)
    lon_nadir_array=np.transpose(np.tile(lon_nadir,(len(lon),1)))
    xcross=np.ma.MaskedArray(LON-lon_nadir_array,False)

    for j in range(len(lat)):
        lontest1=np.logical_and((LON[j,:]>=min(lon_bound1a[j],lon_bound1b[j])),(LON[j,:]<=max(lon_bound1b[j],lon_bound1a[j])))
        lontest2=np.logical_and(LON[j,:]>=min(lon_bound2a[j],lon_bound2b[j]),LON[j,:]<=max(lon_bound2a[j],lon_bound2b[j]))
        lontest=np.logical_or(lontest1,lontest2)
        yswath[j,:] = np.ma.masked_where(lontest, LAT[j,:],copy=True)
        xswath[j,:] = np.ma.masked_where(lontest, LON[j,:],copy=True)
        msla_use[j,:] = np.ma.masked_where(~lontest, msla[j,:])
        xcross[j,:] = np.ma.masked_where(~lontest, xcross[j,:])
        
        
    return xswath,yswath,xcross,msla_use  


def make_error_over_time(days, alpha, latitude_use, longitude_use, deltax_use, asc_des_use):
# days, alpha, yswath_index_left, yswath_index_right, y_mask_left, y_mask_right):
    
    '''
    This function models the time-varying error parameters in satellite swath data, including timing error, roll error, baseline dilation error, and phase error. The roll errors and baseline dilation errors are assumed to be correlated and are generated on the satellite swath. The function takes as inputs the number of days the data is repeated, the model parameters of the roll errors and baseline dilation errors, and the swath index for swath 1 and 2, as well as the swath masks.

The output of the function includes the valid data points of the timing error, roll errors, baseline dilation error, and phase error, as well as the valid coordinates as the distance from the center of the swath ("xc1_valid") and the quadratic of that distance ("xc2_valid").
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
def read_simulator_error(start_day,ndays):
    import numpy as np
    import datetime as dt

    stats_output=np.zeros([ndays*2,6])
    # start_date=dt.date(start_year,start_month,start_day)
    # delta=dt.timedelta(days=1)
    icount=0

# read first ascending and descending files only for latitude, longitude, and cross track distance
    # date_string=start_date.strftime("/home/sgille/swot_simulator/karin/%Y/SWOT_L2_LR_SSH_Expert_001_002_%Y%m%d*.nc")
    # date_string=print('/home/sgille/swot_simulator/karin/%(Year)i/SWOT_L2_LR_SSH_Expert_%(counter)03d_002_*.nc' % {'Year': start_year, 'counter':icount+1}) 
    simulator_files = sorted(glob('/home/sgille/swot_simulator/karin/20??/SWOT_L2_LR_SSH_Expert_' + str(start_day).zfill(3) + '_002*.nc'))
    # simulator_files = sorted(glob(date_string))
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

    sim_err_baseline_dilation = simulator_ds['simulated_error_baseline_dilation'].values
    sim_err_karin = simulator_ds['simulated_error_karin'].values
    sim_err_roll = simulator_ds['simulated_error_roll'].values
    sim_err_phase = simulator_ds['simulated_error_phase'].values
    sim_err_orbital = simulator_ds['simulated_error_orbital'].values
    sim_err_timing = simulator_ds['simulated_error_timing'].values
    # total_error1=sim_err_baseline_dilation + sim_err_karin + sim_err_roll + sim_err_phase + sim_err_orbital + sim_err_timing
    total_error1=sim_err_baseline_dilation + sim_err_karin + sim_err_roll + sim_err_phase + sim_err_timing
    stats_output[icount*2,0]=np.std(sim_err_baseline_dilation[index])
    stats_output[icount*2,1]=np.std(sim_err_karin[index])
    stats_output[icount*2,2]=np.std(sim_err_roll[index])
    stats_output[icount*2,3]=np.std(sim_err_phase[index])
    stats_output[icount*2,4]=np.std(sim_err_orbital[index])
    stats_output[icount*2,5]=np.std(sim_err_timing[index])

    sim_err_baseline_dilation2 = simulator_ds2['simulated_error_baseline_dilation'].values
    sim_err_karin2 = simulator_ds2['simulated_error_karin'].values
    sim_err_roll2 = simulator_ds2['simulated_error_roll'].values
    sim_err_phase2 = simulator_ds2['simulated_error_phase'].values
    sim_err_orbital2 = simulator_ds2['simulated_error_orbital'].values
    sim_err_timing2 = simulator_ds2['simulated_error_timing'].values
    # total_error2=sim_err_baseline_dilation2 + sim_err_karin2 + sim_err_roll2 + sim_err_phase2 + sim_err_orbital2 + sim_err_timing2
    total_error2=sim_err_baseline_dilation2 + sim_err_karin2 + sim_err_roll2 + sim_err_phase2 + sim_err_timing2
    stats_output[icount*2+1,0]=np.std(sim_err_baseline_dilation2[index])
    stats_output[icount*2+1,1]=np.std(sim_err_karin2[index])
    stats_output[icount*2+1,2]=np.std(sim_err_roll2[index])
    stats_output[icount*2+1,3]=np.std(sim_err_phase2[index])
    stats_output[icount*2+1,4]=np.std(sim_err_orbital2[index])
    stats_output[icount*2+1,5]=np.std(sim_err_timing2[index])
    total_error = np.concatenate((total_error1[index],total_error2[index2]))

 
# now loop through all of the files to read error
    for i in range(1,ndays):
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
        stats_output[i*2+1,0]=np.std(sim_err_baseline_dilation2[index])
        stats_output[i*2+1,1]=np.std(sim_err_karin2[index])
        stats_output[i*2+1,2]=np.std(sim_err_roll2[index])
        stats_output[i*2+1,3]=np.std(sim_err_phase2[index])
        stats_output[i*2+1,4]=np.std(sim_err_orbital2[index])
        stats_output[i*2+1,5]=np.std(sim_err_timing2[index])
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
    H_all[:, 0::2] = H_cos 
    H_all[:, 1::2] = H_sin
    
    return H_all