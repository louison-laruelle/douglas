# DOuGLAS v1.0 - GPLv3
# Louison Laruelle, laruelle@gfz.de
# Manual: https://doi.org/10.48440/wsm.2025.002
# Download: https://github.com/louison-laruelle/douglas

##################################################################################################

# Python Detection of Outliers in Geomechanics using Linear-elastic Assumption and Statistics is 
# a Python 3.x tool allowing the fast identification of outliers in stress magnitudes dataset. 
# It supports the Moose and Abaqus solvers and the Tecplot GeoStressCmd add-on 
# (https://doi.org/10.5880/wsm.2020.001), and runs on both Windows and Linux systems. It can be 
# used as a stand-alone script or called from another script. This script interpolates parts of 
# the PyFAST Calibration script (https://doi.org/10.5880/wsm.2021.003).
#
# The basic principle of the tool is explained in the DOuGLAS Manual referenced above. Additional 
# required information is provided in this file. For simple usage, refer only to the user input 
# section. Additional comments are provided throughout the script.
#
# Important Information: When using Tecplot Macros, three steps with different boundary conditions
# are expected to be loaded in Tecplot. Make sure the folder variable is set correctly and that 
# the subdirectory 'data' exists in the working directory.

filename = "/home/username/projects/YourProject/magnitudes.xlsx"
bcs = [[4, 2, 4],[-4, -5, -3]]
name = 'magnitudes'
threshold = 95#% 

# functions
###############################################################################################
def fast_calibration_main(folder, shmax,shmin,stress_vars,bcs,name):
    import os.path
    
    [shmax_calib,shmin_calib] = load_csv(name,len(shmax),len(shmin),stress_vars)
  
    # If raw stress data directly from the solver in Pascal (instead of MPa) is
    # used, it is converted to the geoscientific convention (compression positive)
    # and to MPa.
    if stress_vars[0] != 'SHmax':
        shmax_calib = shmax_calib*(-1e-06)
        shmin_calib = shmin_calib*(-1e-06)
    [bcx,bcy] = calibration(shmax,shmin,shmax_calib,shmin_calib,bcs)
  
    return [bcx,bcy]

###############################################################################################
def write_macro(shmax,shmin,stress_vars,name,folder):
  # A Tecplot macro to export the stress state at the calibration points
  # is written.
  import numpy as np
  
  # Create one variable with all locations of calibration points.
  coords = []
  for i in range(len(shmax)):
    coords.append(shmax[i][0:3])
  for i in range(len(shmin)):
    coords.append(shmin[i][0:3])
  
  # Start writing the macro file with its header.
  # (The header may vary depending on Tecplot version.)
  macro_name = name + '.mcr'
  fid =  open(macro_name
              ,'w')
  fid.write('#!MC 1410\n\n')
  # Find the number Tecplot assigned to the variables according to their names.
  fid.write('$!GETVARNUMBYNAME |SHMAX|\nNAME = "%s"\n$!GETVARNUMBYNAME |SHMIN|\nNAME = "%s"\n\n' % (stress_vars[0],stress_vars[1]))
  
  # Create 1D zones at the calibration points. At each point 3 zones are created.
  for i in range(len(coords)):
    for j in range(3):
      fid.write('$!CREATERECTANGULARZONE\nIMAX = 1\nJMAX = 1\nKMAX = 1\n')
      fid.write('X1 = %i\nY1 = %i\nZ1 = %i\n' % (coords[i][0],coords[i][1],coords[i][2]))
      fid.write('X2 = %i\nY2 = %i\nZ2 = %i\n\n' % (coords[i][0],coords[i][1],coords[i][2]))
  
  # Interpolate stress state to newly created 1D zones. From each boundary condition
  # scenario at each calibration point the stress state is interpolated to a 1D zone.
  for i in range(len(coords)):
    for x,j in enumerate([1,2,3]):
      fid.write('$!LINEARINTERPOLATE\nSOURCEZONES =  [%i]\n' % j)
      fid.write('DESTINATIONZONE = %i\nVARLIST =  [|SHMAX|,|SHMIN|]\nLINEARINTERPCONST = 0\nLINEARINTERPMODE = DONTCHANGE\n\n' % (3+(i*3)+j))
  
  # Export the two stress components to individually named files in the specified folder.
  for i in range(2):
    fid.write('$!EXTENDEDCOMMAND\nCOMMANDPROCESSORID = \'excsv\'\n')
    if i == 0:
      inst = 'SHMAX'
      id = '_' + stress_vars[0]
    elif i == 1:
      inst = 'SHMIN'
      id = '_' + stress_vars[1]
    
    fid.write('COMMAND = \'FrOp=1:ZnCount=%i:ZnList=[%i-%i]:VarCount=1:VarList=[|%s|]:ValSep=",":FNAME="%s\\data\\%s.csv"\'\n\n' % (len(coords)*3,4,len(coords)*3+3,inst,folder,(name+id)))
  
  # Delet the 1D zones.
  fid.write('$!DELETEZONES [%i-%i]' % (4,len(coords)*3+3))
  
  fid.close()

###############################################################################################
def load_csv(name, leshmax, leshmin, stress_vars):
    # Load the files/stress components that were exported from Tecplot with the macro.
    import csv
    import numpy as np
    # Read the SHmax (or other first variable) file.
    shmax_temp = []
    shmax_calib = []
    with open('data/' + name + '_' + stress_vars[0] + '.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row != []:
                shmax_temp.append(float(row[0]))

    # Read the Shmin (or other second variable) file.
    shmin_temp = []
    shmin_calib = []
    with open('data/' + name + '_' + stress_vars[1] + '.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row != []:
                shmin_temp.append(float(row[0]))

    # Sort the SHmax (or other first variable) according to calibration points.
    for i in range(leshmax):
        temp = []
        for j in range(3):
            temp.append(shmax_temp[j + i * 3])
        shmax_calib.append(temp)

    # Sort the Shmin (or other second variable) according to calibration points.
    for i in range(leshmin):
        temp = []
        for j in range(3):
            temp.append(shmin_temp[j + (leshmax + i) * 3])
        shmin_calib.append(temp)

    # The variables are converted to a numpy array.
    shmax_calib = np.array(shmax_calib)
    shmin_calib = np.array(shmin_calib)

    return [shmax_calib, shmin_calib]

###############################################################################################
def calibration(shmax, shmin, shmax_calib, shmin_calib, bcs):
    # Derive the best-fit boundary conditions for the final model using the
    # FAST Calibration algorithm described in the manual.
    import numpy as np

    # Apply the weighting of the calibration points.
    dshmin = np.array([shmin_calib[i, :] - shmin[i][3] for i in range(len(shmin))])
    weight = [shmin[i][4] for i in range(len(shmin))]
    dshmin = np.average(dshmin, 0, weight)

    dshmax = np.array([shmax_calib[i, :] - shmax[i][3] for i in range(len(shmax))])
    weight = [shmax[i][4] for i in range(len(shmax))]
    dshmax = np.average(dshmax, 0, weight)

    # Transform relevant variables into an array.
    xco = [float(i) for i in bcs[0]]
    yco = [float(i) for i in bcs[1]]
    dshmaxo = [float(i) for i in dshmax]
    dshmino = [float(i) for i in dshmin]

    # Setup 'planes' and derive the equation for the isolines of zero deviation.
    for i in range(2):
        v = [xco]
        v.append(yco)
        if i == 0:
            v.append(dshmaxo)
        else:
            v.append(dshmino)

        v = np.asarray(v)
        v = np.transpose(v)
        r1 = v[1, :] - v[0, :]
        r2 = v[2, :] - v[0, :]

        r2 = np.where(r2 == 0, 0.00001, r2)
        test = [r1[j] / r2[j] for j in range(3)]
        if test[0] == test[1] and test[1] == test[2]:
            print('ERROR! Planes are linearly dependent.')

        n = np.cross(r1, r2)
        d = np.dot(n, v[0, :])

        if i == 0:
            d_x = d
            n1_x = n[0]
            n2_x = n[1]
        else:
            d_i = d
            n1_i = n[0]
            n2_i = n[1]

    # Compute the best-fit boundary conditions.
    if n2_x == 0:
        n2_x = 0.0001
    x = (n2_i * d_x - n2_x * d_i) / (n1_x * n2_i - n2_x * n1_i)
    y = (d_x - n1_x * x) / n2_x

    return [x, y]


###############################################################################################
def comb_number(n1, n, k):
    # Calculates the number of combinations of stress magnitudes 
    # with n1 the number of elements to pick, n the total number of stress magnitudes and k the number of shmin (or shmax)
    from scipy.special import comb
    return comb(n, n1, exact=True) - comb(k, n1, exact=True) - comb(n - k, n1, exact=True)

###############################################################################################
def bc_generator(bcs, filename, stress_vars, name):
    # create a list of all the possible best-fit boundary conditions evaluated with subsets of the complete calibration dataset
    from tqdm import tqdm
    import numpy as np
    import pandas as pd
    import os
    
    def create_matrix(df, col_ind):
        # create a stress magnitudes matrix from the stress magnitudes dataframe
        # df is the pandas dataframe corresponding to the Excel or CSV file
        # col_ind is the list of index of the columns where the stress magnitudes are contained (Shmin and SHMax)
        import pandas as pd
        import warnings
        # Suppress only FutureWarnings
        warnings.simplefilter(action='ignore', category=FutureWarning)
        # Replace the empty cases with text
        df.fillna("nan", inplace=True)
        # creates a matrix of all the stress magnitudes values
        # each line is organized as [type of data (shmin/shmax), x, y, z (bsl), stress magnitude, 0] 
        mat = [[df.columns[j], df.iloc[i, 0], df.iloc[i, 1], df.iloc[i, 2], df.iloc[i, j], 0]
            for j in col_ind for i in range(len(df))
            if df.iloc[i, j] != "nan"]
        return mat
    
    def all_combinations_rows(mat, n):
        # generates all the possible combinations of three elements
        # returns the correct input for fast calibration (list of shmin and list of shmax)
        # each times the weights are kept to zero except for 3 elements where they are set to 1
        # this is being use by fast calibration with the weighted average
        from itertools import combinations
        import numpy as np
        def has_non_zero_last_element(rows):
            # checks if the shmin or shmax list has all their indices set to zero
            return any(row[-1] != 0 for row in rows)
        
        # Generate all combinations of indices
        all_combinations = list(combinations(range(len(mat)), n)) # list of all possible combinations of 3 indices
        
        for combination in all_combinations:
            calib_ls = np.copy(mat)
            random_row_indices = np.array(combination) 
            calib_ls[random_row_indices, -1] = 1  # Mark selected rows with an 1 (weight for the weighted average)
            shmin, shmax = np.array([row[1:] for row in calib_ls if row[0] == 'shmin'], dtype = float), np.array([row[1:] for row in calib_ls if row[0] == 'shmax'], dtype = float)
            # Ensure both shmin and shmax have at least one valid entry
            if has_non_zero_last_element(shmin) and has_non_zero_last_element(shmax):
                yield shmin, shmax
    
    # Determine the file type and read the data
    if filename.endswith('.xlsx') or filename.endswith('.xls'):
        df = pd.read_excel(filename)  # Read Excel file
    elif filename.endswith('.csv'):
        df = pd.read_csv(filename)  # Read CSV file
    else:
        raise ValueError(f"Unsupported file format for file: {filename}")
    
    # find the indices of shmin and shmax columns
    col_names = ["shmin", "shmax"]
    col_ind = [df.columns.get_loc(col) for col in col_names if col in df.columns]
    folder = os.path.dirname(filename) # identify folder for the repository
    
    # Estimate reference boundary conditions
    calib_all = create_matrix(df, col_ind)  # list of all the calibration values,
    calib_ls = np.array(calib_all)
    calib_ls[:, -1] = 1 # set the weight of all the samples to 1
    shmin_ref, shmax_ref = np.array([row[1:] for row in calib_ls if row[0] == 'shmin'], dtype = float), np.array([row[1:] for row in calib_ls if row[0] == 'shmax'], dtype = float)
    all_samples = np.concatenate([shmin_ref.T[0:4], shmax_ref.T[0:4]], axis = 1) # create a list of all the stress magnitudes
    
    write_macro(shmax_ref,shmin_ref,stress_vars,name,folder)
    # A break appears here in order to execute the macro in Tecplot.
    print ('Execute Macro in Tecplot...')
    input("...then press Enter to continue...")
    
    n_size = 3 # size of the subsets

    k = np.sum(np.array(calib_all)[:, 0] == "shmin") # number of shmin data
    n_iterations = comb_number(n_size, len(all_samples[0]), k) # number of iterations of the process
    print(str(n_iterations)+" iterations of the process")
    
    # Create empty list for boundary conditions and stress magnitudes
    boot_ls = np.zeros((n_iterations, n_size, 4))
    bc_list = np.zeros((n_iterations, 2))
    
    # Extract all the possible boundary conditions
    combinations_generator = all_combinations_rows(calib_all, n_size)
    for i, (shmin, shmax) in enumerate(tqdm(combinations_generator, desc="Creating BC list", unit=" iteration")):
        filtered_shmin, filtered_shmax = shmin[shmin[:, -1] != 0], shmax[shmax[:, -1] != 0] # shmin, shmax couples used
        boot_ls[i][:][:] = np.concatenate([filtered_shmin[:, 0:4], filtered_shmax[:, 0:4]], axis = 0)
        bc_list[i][0], bc_list[i][1] = fast_calibration_main(folder, shmax, shmin, stress_vars, bcs, name)
    print('BC list created!')
    return bc_list, calib_ls, boot_ls

###############################################################################################
def crw(data, threshold):
    import numpy as np
    from scipy.stats import norm
    # Center the data to median
    data_centered = data - np.median(data, axis = 0)
    
    # Compute the covariance matrix and its eigen decomposition
    cov_matrix = np.cov(data_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)
    
    # Whiten the data (remove correlation and scale variances)
    whitening_matrix = eigvecs @ np.diag(1/ np.sqrt(eigvals)) @ eigvecs.T
    data_centered = data_centered @ whitening_matrix
    
    #Scale the square so that 95% of points are between [-1, 1]
    x_percentile = np.percentile(data_centered[:, 0], threshold)
    y_percentile = np.percentile(data_centered[:, 1], threshold)

    # Scaling factors to bring 90% of points between -0.5 and 0.5
    x_scaling_factor = 1 / ( abs(norm.ppf(threshold / 100)) * x_percentile + 0.0001)
    y_scaling_factor = 1 / ( abs(norm.ppf(threshold / 100)) * y_percentile + 0.0001)

    # Scale the data
    data_scaled = data_centered * np.array([x_scaling_factor, y_scaling_factor])
    return data_scaled

###############################################################################################
def sample_scores(boot_ls, bc_list, calib_ls, threshold):
    import numpy as np
    all_samples = np.concatenate([np.array([row[1:] for row in calib_ls if row[0] == 'shmin'],dtype = float).T[0:4],
                                  np.array([row[1:] for row in calib_ls if row[0] == 'shmax'],dtype = float).T[0:4]], axis = 1) # create a list of all the stress magnitudes
    printed_extent_message = False  # Flag to ensure the extent message is printed only once
    
    while True:  # Start an iterative loop
        # Compute the transformed square with the current threshold
        bc_list_square = crw(bc_list, threshold)
        
        # Create a mask for the outliers based on the transformed square
        outlier_count = np.zeros(len(all_samples[0]))
        outlier_mask = (np.linalg.norm(bc_list_square, axis=1, ord=np.inf) > 1)
        n_iterations = len(bc_list_square)
        
        # Calculate the range of boundary conditions
        range_bcx = np.max(bc_list.T[0]) - np.min(bc_list.T[0])
        range_bcy = np.max(bc_list.T[1]) - np.min(bc_list.T[1])
        
        # Iterate over the iterations where outliers are detected
        for i in range(n_iterations):
            if outlier_mask[i]:
                # Convert all_samples and boot_ls[i] into comparable formats
                all_samples_flat = all_samples.T.reshape(-1, all_samples.shape[0])
                boot_sample_flat = boot_ls[i].reshape(-1, boot_ls[i].shape[-1])

                # Match rows between all_samples and boot_sample
                matching_indices = []
                for idx, row in enumerate(all_samples_flat):
                    if any(np.array_equal(row, boot_row) for boot_row in boot_sample_flat):
                        matching_indices.append(idx)
                
                # Increment count for outliers proportional to the norm
                outlier_count[matching_indices] += np.linalg.norm(bc_list_square[i], ord=2)
        
        # Check if outliers are detected
        if not np.all(outlier_count == 0):
            # Outliers detected, break the loop
            break

        # If no outliers and range conditions are met, lower the threshold
        if range_bcx > 5 or range_bcy > 5:
            print("\nNo outlier detected")
            if not printed_extent_message:
                print(
                    "The distribution of boundary conditions has an extent of %.2f m in the x direction and %.2f m in the y direction, indicating potential outliers" 
                    % (range_bcx, range_bcy)
                )
                printed_extent_message = True  # Mark message as printed
            threshold -= 5
            print(f"...Re-running with a threshold of {threshold}%")
        else:
            # Stop the process if the range conditions are no longer met
            print("\nNo outlier detected, and range conditions are not met. There are no clear outliers in the dataset")
            break

    return bc_list_square, outlier_count

###############################################################################################
def plot_results(bc_list, bc_list_square, outlier_count, calib_ls, filename, stress_vars):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.widgets import Slider
    import numpy as np
    import os
    import warnings
    
    # Suppress only FutureWarnings
    warnings.simplefilter(action='ignore', category=UserWarning)
    
    def update_plot(val):
        # Updates the plot as the user moves the slider
        detect_thresh = slider.val  # Update threshold value from the slider
        bar_colors = ['red' if count > detect_thresh * count_max / 100 else 'skyblue' for count in outlier_count]  # Recolor based on new threshold

        # Clear previous bar plot in the ax3 subplot
        ax3.clear()

        # Redraw the bar plot with updated colors
        ax3.axhline(detect_thresh_init, linestyle = "--", c = "lightblue", zorder = 0)
        ax3.bar(index_labels, outlier_count/np.max(outlier_count)*100, color=bar_colors, zorder = 1)
        ax3.axhline(detect_thresh, linestyle = "--", c = "r")
        ax3.set_xticks(range(len(index_labels)))
        ax3.set_xticklabels(index_labels, rotation=45, fontsize=7, rotation_mode="anchor", ha='right')
        ax3.set_ylim([0, 100])
        ax3.set_xlabel('Stress magnitude data record', fontsize = 11)
        ax3.set_ylabel('Score (as percentage of the max. score)', fontsize = 11)
        ax3.set_title(f'Outlier Detection by Location (Detection level: {slider.val:.1f}% of the maximum score)')
        
        # Update BC in ax1
        ls_weights = [0 if count > detect_thresh * count_max / 100 else 1 for count in outlier_count]
        calib_ls[:, -1] = ls_weights # set the weight of all the samples that are not labeled as outliers to 1
        shmin_new, shmax_new = np.array([row[1:] for row in calib_ls if row[0] == 'shmin'], dtype = float), np.array([row[1:] for row in calib_ls if row[0] == 'shmax'], dtype = float)
        if all(sublist[-1] == 0 for sublist in shmin_new) or all(sublist[-1] == 0 for sublist in shmax_new):
            #check if the calibration is still possible
            pass # do not plot anything 
        else:
            bcx_new, bcy_new = fast_calibration_main(folder, shmax_new, shmin_new, stress_vars, bcs, name) # boundary conditions without the outlier
            ax1.clear()
            ax1.text(0.99, 0.99, "BC without outlier(s): %.2fm, %.2fm\nReference BC: %.2fm, %.2fm" % (bcx_new, bcy_new, bcx_ref, bcy_ref), ha='right', va='top', transform=ax1.transAxes)
            ax1.scatter(bc_list[:, 0], bc_list[:, 1], color='blue')
            ax1.quiver(*origin, np.sqrt(eigvals[0]) * eigvecs[0, 0], np.sqrt(eigvals[0]) * eigvecs[1, 0],
                   angles='xy', scale_units='xy', scale=1, color='r', label='Eigenvector 1')

            ax1.quiver(*origin, np.sqrt(eigvals[1]) * eigvecs[0, 1], np.sqrt(eigvals[1]) * eigvecs[1, 1],
                   angles='xy', scale_units='xy', scale=1, color='g', label='Eigenvector 2')
            ax1.axhline(0, color='black', linewidth=0.5)
            ax1.axvline(0, color='black', linewidth=0.5)
            ax1.scatter(bcx_ref, bcy_ref, marker = "+", c = "red", linewidth = 1, s = 100, label = "Reference BC")
            ax1.scatter(bcx_new, bcy_new, marker = "+", c = "skyblue", linewidth = 1, s = 100, label = "BC without outlier(s)")
            ax1.set_xlabel('Boundary condition X', fontsize = 11)
            ax1.set_ylabel('Boundary condition Y', fontsize = 11)
            ax1.set_title("Boundary conditions distribution", fontsize = 15)
            ax1.legend()
            calib_results["bcx_new"], calib_results["bcy_new"] = bcx_new, bcy_new
        # Redraw the figure
        fig.canvas.draw_idle()
    
    # Computation of the boundary conditions with the outlier
    folder = os.path.dirname(filename) # identify folder for the repository
    shmin_ref, shmax_ref = np.array([row[1:] for row in calib_ls if row[0] == 'shmin'], dtype = float), np.array([row[1:] for row in calib_ls if row[0] == 'shmax'], dtype = float)
    bcx_ref, bcy_ref = fast_calibration_main(folder, shmax_ref, shmin_ref, stress_vars, bcs, name)
    
    # Eigenvectors
    cov_matrix = np.cov(bc_list, rowvar=False) # Calculate the principal axes using PCA
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)
    
    # Indices for the sample
    origin = np.median(bc_list, axis = 0)
    indexes = [row[:4].tolist() for row in calib_ls] # indexes of the bars
    index_labels = [', '.join(map(str, idx)) for idx in indexes]
    
    # Plot figure
    fig = plt.figure(figsize=(14, 12))  # Adjust size as needed
    
    # Top-left plot (Original Boundary Conditions distribution)
    ax1 = plt.subplot2grid((2, 2), (0, 0))  # Position at (0, 0)
    ax1.scatter(bc_list[:, 0], bc_list[:, 1], color='blue')
    ax1.quiver(*origin, np.sqrt(eigvals[0]) * eigvecs[0, 0], np.sqrt(eigvals[0]) * eigvecs[1, 0],
           angles='xy', scale_units='xy', scale=1, color='r', label='Eigenvector 1')

    ax1.quiver(*origin, np.sqrt(eigvals[1]) * eigvecs[0, 1], np.sqrt(eigvals[1]) * eigvecs[1, 1],
           angles='xy', scale_units='xy', scale=1, color='g', label='Eigenvector 2')
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.axvline(0, color='black', linewidth=0.5)
    ax1.scatter(bcx_ref, bcy_ref, marker = "+", c = "red", linewidth = 1, s = 100, label = "Reference Boundary Conditions")
    
    count_max = np.max(outlier_count)
    detect_thresh_init = 50 # initial threshold for outlier detection
    detect_thresh = detect_thresh_init
    
    # compute new boundary conditions without the outlier
    ls_weights = [0 if count > detect_thresh * count_max / 100 else 1 for count in outlier_count]
    calib_ls[:, -1] = ls_weights # set the weight of all the samples that are not labeled as outliers to 1
    shmin_new, shmax_new = np.array([row[1:] for row in calib_ls if row[0] == 'shmin'], dtype = float), np.array([row[1:] for row in calib_ls if row[0] == 'shmax'], dtype = float)
    bcx_new, bcy_new = fast_calibration_main(folder, shmax_new, shmin_new, stress_vars, bcs, name)
    calib_results = {"bcx_new": bcx_new, "bcy_new": bcy_new}  # Initialize with initial values
    
    ax1.text(0.99, 0.99, "BC without outlier(s): %.2fm, %.2fm\nReference BC: %.2fm, %.2fm" % (bcx_new, bcy_new, bcx_ref, bcy_ref), ha='right', va='top', transform=ax1.transAxes)
    ax1.scatter(bcx_new, bcy_new, marker = "+", c = "skyblue", linewidth = 1, s = 100, label = "Boundary Conditions without outlier(s)")
    ax1.set_xlabel('Boundary condition X', fontsize = 11)
    ax1.set_ylabel('Boundary condition Y', fontsize = 11)
    ax1.set_title("Boundary conditions distribution", fontsize = 15)
    ax1.legend()

    # Top-right plot (Whitened transformed and reduced BC distribution)
    ax2 = plt.subplot2grid((2, 2), (0, 1))  # Position at (0, 1)
    ax2.scatter(bc_list_square[:, 0], bc_list_square[:, 1], color='red')
    lim_value = np.max(np.abs(bc_list_square))
    lim_plot = 2 if lim_value < 2 else lim_value
    ax2.set_xlim([-lim_plot, lim_plot])
    ax2.set_ylim([-lim_plot, lim_plot])
    square = patches.Rectangle((-1, -1), 2, 2, linewidth=1, edgecolor='k', facecolor='none')
    ax2.add_patch(square)
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.axvline(0, color='black', linewidth=0.5)
    ax2.set_xlabel('Centered reduced boundary condition X', fontsize = 11)
    ax2.set_ylabel('Centered reduced boundary condition Y', fontsize = 11)
    ax2.set_title("Centered reduced boundary conditions", fontsize = 15)

    # Bottom plot (Outlier Presence by Location)
    ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)  # Span both columns
    plt.subplots_adjust(bottom=0.17)  # Adjust to leave space for the slider
    bar_colors = ['red' if count > detect_thresh * count_max / 100 else 'skyblue' for count in outlier_count]  # Recolor based on new threshold
    bars = ax3.bar(index_labels, outlier_count/count_max*100, color=bar_colors)
    ax3.axhline(detect_thresh, linestyle = "--", c = "r")
    ax3.set_xticks(range(len(index_labels)))
    ax3.set_xticklabels(index_labels, rotation=45, fontsize=7, rotation_mode="anchor", ha='right')
    ax3.set_xlabel('Stress magnitude data record', fontsize = 11)
    ax3.set_ylim([0, 100])
    ax3.set_ylabel('Score (as percentage of the max. score)', fontsize = 11)
    ax3.set_title(f'Outlier Detection by Location (Detection level: {detect_thresh_init:.1f}% of the maximum score)')



    # Create the slider for the outlier plot
    ax_slider = plt.axes([0.2, 0.02, 0.65, 0.03], facecolor='lightgoldenrodyellow')  # Position of the slider
    slider = Slider(ax_slider, 'Error limit [%]', 0.0, 100, valinit=detect_thresh)  # Slider range 0-2%
    slider.on_changed(update_plot)  # Attach the update function
    plt.savefig("figure_outlier_id.svg")
    plt.show()
    # Clear the current figure if you want to continue plotting
    plt.clf()  # Clear the current figure
    
    print("\nBoundary conditions using all data records: BCX = %.2fm, BCY = %.2fm" % (bcx_ref, bcy_ref))
    print("Boundary conditions excluding data records labelled as outliers: BCX = %.2fm, BCY = %.2fm" % (calib_results["bcx_new"], calib_results["bcy_new"]))

###############################################################################################
def main(filename, bcs, name):
    stress_vars = ['SHmax', 'Shmin']
    bc_list, calib_ls, boot_ls = bc_generator(bcs, filename, stress_vars, name)
    bc_list_square, outlier_count = sample_scores(boot_ls, bc_list, calib_ls, threshold)
    plot_results(bc_list, bc_list_square, outlier_count, calib_ls, filename, stress_vars)

###############################################################################################
if __name__ == '__main__':
    print('Running DOuGLAS v1.1')
    main(filename, bcs, name)
