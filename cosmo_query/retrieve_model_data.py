#!/usr/bin/env python2
"""
Author Daniel Wolfensberger - EPFL-LTE
Created on Thu Mar 23 11:27:32 2017
"""


# General imports
import numpy as np
import paramiko
import os
import inspect
import datetime
import netCDF4
import re
import uuid
import shutil
import fnmatch

# Module imports
from metaarray import MetaArray
import config as cfg # all constants

        
class SSH(object):
    """
    CLASS:
        connection = SSH(address, username, password = None)
    
    PURPOSE:
         Creates a connection with a given server, using specified ssh
         identifiers
         
         IMPORTANT: don't make too many connection or you will get banned.
         Create one and reuse it for all you queries...
    
    INPUTS:
        adress : adress of the server, either a IP or a name, ex. ela.cscs.ch
        username : the username to be used, for example "wolfensb"
        password : (optional) the corresponding password. If not set, it will
                   be assumed that a ssh key has been setup on the server. THIS
                   IS THE RECOMMENDED WAY.
    
    OUTPUTS:
        connection : a ssh connection that can be used to send commands remote-
                     ly to the distant server
    """    
    def __init__(self, adress, username, password = None):
        # Let the user know we're connecting to the server
        print("Connecting to server...")
        # Create a new SSH client
        self.jump = paramiko.SSHClient()
        self.jump_adress = adress
        self.target = None
        self.target_adress = None
        self.username = username
        # The following line is required if you want the script to be able to 
        # access a server that's not yet in the known_hosts file
        self.jump.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        # Make the connection
        
        if not password: # If no password set, use ssh key
            privatekeyfile = os.path.expanduser('~/.ssh/id_rsa')
            mykey = paramiko.RSAKey.from_private_key_file(privatekeyfile)
            self.jump.connect(adress, username=username, look_for_keys=True, 
                              pkey = mykey)            
        else:
            self.jump.connect(adress, username=username, password = password,
                allow_agent = True)
        
        # Open ftp
        self.ftp = self.jump.open_sftp()
        
    def open_channel(self,adress, username, password = None):
        """
        FUNCTION:
            open_channel(adress, username, password = None)
        
        PURPOSE:
             Opens a channel to a new remote server, from the server 
             you are connected to
        
        INPUTS:
            adress : adress of the server, either a IP or a name, ex. 
                kesch.cscs.ch
            username : the username to be used, for example "wolfensb"
            password : (optional) the corresponding password. If not set, it will
                be assumed that a ssh key has been setup on the server. THIS
                IS THE RECOMMENDED WAY.
        OUTPUTS:
            None (but you will be able to send command to new remote)
        
        """    
            
        transport = self.jump.get_transport()
        channel = transport.open_channel('direct-tcpip', [adress,22],
                                         (self.jump_adress, 0))
        
        self.target = paramiko.SSHClient()
        self.target_adress = adress
        self.target.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        if not password: # If no password set, use ssh key
            privatekeyfile = os.path.expanduser('~/.ssh/id_rsa')
            mykey = paramiko.RSAKey.from_private_key_file(privatekeyfile)
            self.target.connect(adress, username=username,
                look_for_keys=True, sock = channel, pkey = mykey)            
        else:
            self.target.connect(adress, username=username, password = password,
                sock = channel, allow_agent = True)
        
    def send_command(self, command, client = 'jump'):
        """
        FUNCTION:
            send_command(command, client = 'jump')
        
        PURPOSE:
             Sends a command to a remote server
        
        INPUTS:
            command : the command in the form of a python string
            client : either 'jump' or 'target', if 'jump' the command will
                 be sent to the original server (the one defined when 
                 creating a ssh instance), if 'target', the command will
                 be sent to the new remote to which a channel must have
                 been previously opened with the open_channel function.
        OUTPUTS:
            None (but the output of the command on the remote will be printed)
        
        """    
        
        # Check if connection is made previously
        if client == 'jump':
            client = self.jump
        elif client == 'target' and self.target:
            client = self.target
        else:
            raise ValueError("Invalid client, must be either 'jump' or 'target'")
            
        stdin, stdout, stderr = client.exec_command(command)
        while not stdout.channel.exit_status_ready():
            # Print stdout data when available
            if stdout.channel.recv_ready():
                
                # Retrieve the first 1024 bytes
                alldata = stdout.channel.recv(1024)
                while stdout.channel.recv_ready():
                    # Retrieve the next 1024 bytes
                    alldata += stdout.channel.recv(1024)

                # Print as string with utf8 encoding
                print(alldata)


class Query(object):
    """
    CLASS:
        query_instance = Query(connection)
    
    PURPOSE:
         Creates an interface that allows to retrieve COSMO data from the
         MCH servers
    
    INPUTS:
        connection : the connection to the MCH servers as an ssh class instance
            connected to ela.cscs.ch. You must also have opened a 
            channel to kesch.cscs.ch
    
    OUTPUTS:
        query_instance : an interface that allows to make queries of COSMO data
    """    
    
    def __init__(self,connection):
        self.connection = connection
        self.data = None
        self.valid_vars = None
        self.temp_folder_r = None
        self.temp_folder_l = None
        
    def get_available_vars(self, date, model_res = 'fine', 
                           mode = 'analysis'):
        """
        FUNCTION:
            list_vars = get_available_vars(time, model_res = 'fine', 
                           mode = 'analysis')
        
        PURPOSE:
             Returns the list of variables that are available in the COSMO
             files at a given time and for a given resolution and mode
        
        INPUTS:
            date : the date of the desired simulation in the YYYY-mm-DD HH:MM 
                format, for example 2017-02-27 13:46
            model_res : the resolution of the model, either 'fine' (COSMO-2 or
                COSMO-1, since 2017) or 'coarse' (COSMO-7)
            mode : the simulation mode can be either 'analysis' or 'forecast'
            
        OUTPUTS:
            list_vars : the list of available variables
        
        """    
        
        date = datetime.datetime.strptime(date,'%Y-%m-%d %H:%M')
        
        if model_res == 'coarse':
            res_name = '7'
        else:
            if date < cfg.COSMO_2_END:
                res_name = '1'
            else:
                res_name = '2'
        
        current_folder = inspect.getfile(inspect.currentframe()) 
        current_folder = os.path.dirname(current_folder)
        fname = current_folder + '/cosmo_info/list_vars/listvar_cosmo' + \
            res_name + '_' + mode + '.txt'
                
        list_vars = [line.rstrip('\n') for line in open(fname)]
        
        return list_vars
    
    def get_COSMO_fname(self,date, model_res = 'fine', mode = 'analysis',
                        forecast_start = None):
        """
        FUNCTION:
            get_COSMO_fname = get_available_vars(self,date, model_res = 'fine',
                 mode = 'analysis', forecast_start = None)
        
        PURPOSE:
             Returns the list of variables that are available in the COSMO
             files at a given time and for a given resolution and mode
        
        INPUTS:
            date : the date of the desired simulation in the YYYY-mm-DD HH:MM 
                format, for example 2017-02-27 13:46
            model_res : the resolution of the model, either 'fine' (COSMO-2 or
                COSMO-1, since 2017) or 'coarse' (COSMO-7)
            mode : the simulation mode can be either 'analysis' or 'forecast'
            
        OUTPUTS:
            list_vars : the list of available variables
        
        """   
        
        if mode == 'analysis': # Analysis
            if date < cfg.COSMO_2_END: # Old simulations
                fname = cfg.COSMO_ROOT_DIR + '/osm/LA'+str(date.year)[2:] + '/' + \
                    datetime.datetime.strftime(date,'%Y%m%d')+'/'+model_res+'/' + \
                    'laf' + datetime.datetime.strftime(date,'%Y%m%d%H')         
            else: # New simulations
                if model_res == 'fine':
                    model_name = 'COSMO-1'
                elif model_res == 'coarse':
                    model_name = 'COSMO-7'
                    
                fname = cfg.COSMO_ROOT_DIR + '/owm/'+model_name+'/ANA'+str(date.year)[2:] + '/laf' + \
                    datetime.datetime.strftime(date,'%Y%m%d%H')
                    
        elif mode == 'forecast': # Forecast
            # Get start time of forecast
            if not forecast_start: # If no forecast time, get the closest one
                # This gets the closest starting hour (before the actual time)
                closest_starthour = max(filter(lambda x: x <= date.hour,
                                           cfg.COSMO_FORECAST_STARTS))
                forecast_start = datetime.datetime(date.year,date.month,date.day,
                                               closest_starthour)
            if forecast_start.hour not in [0,3,6,9,12,15,18,21]:
                raise ValueError('Invalid forecast start time, COSMO runs start every 3 hours')
            
            # Hour of forecasting
            forecast_hours = date.hour - forecast_start.hour 
            
            if forecast_hours > cfg.FORECAST_DURATION:
                msg = """
                Forecast start time is located too much in the past, compared
                with the desired simulation time, the maximal forecast duration
                is {:d} hours
                """.format(cfg.FORECAST_DURATION)
                
                raise ValueError(msg)
                
            if forecast_start < cfg.COSMO_2_END: # Old simulations
                if model_res == 'fine':
                    model_name = 'f'
                elif model_res == 'coarse':
                    model_name = 'c'
                        
                fname = cfg.COSMO_ROOT_DIR + '/osm/LM'+str(forecast_start.year)[2:] + '/' + \
                    datetime.datetime.strftime(forecast_start,'%y%m%d%H')+'*/grib/' + \
                    'verif_'+model_name+str(forecast_hours).zfill(2)
            else:  # Newer simulations
                if model_res == 'fine':
                    model_name = 'COSMO-1'
                    prefix = 'c1ffsurf'
                elif model_res == 'coarse':
                    model_name = 'COSMO-7'
                    prefix = 'c7ffsurf'
                    
                fname = cfg.COSMO_ROOT_DIR + '/owm/'+model_name+'/FCST'+str(forecast_start.year)[2:] + '/' + \
                    datetime.datetime.strftime(forecast_start,'%y%m%d%H')+'*/grib/' + \
                    prefix + str(forecast_hours).zfill(3)
        else:
            raise ValueError("Invalid mode, must be 'forecast' or 'analysis'")
        
        # Stat the file
        try:
            print(self.connection.ftp.stat(fname))
        except:
            msg = """
            File {:s} was not found on server, we will try to find in the MCH
            archives...
            """.format(fname)
            
            # Try to guess the .tar filename in the archives
            part_1,la_name,part_2 = re.split(r'(LA\d{2})',fname)
            arch_name = re.findall(r'\/(\d{8})\/',part_2)[0] + '.tar'
            
            arch_name_full = cfg.COSMO_ARCHIVE_DIR + la_name + '/' + arch_name
            
            try:
                print(self.connection.ftp.stat(arch_name_full))
            except:
                print('No archive file could be found')
                print('Sorry this is the end, I am helpless')
                print('Aborting...')
                raise
            
            print('Trying to untar the requested file...')
            
            fname_bname = os.path.basename(fname)
            
            untar_cmd = 'tar -xvf ' + arch_name_full + ' ' + part_2[1:] + \
            ' -O > ' + self.temp_folder_r + '/' + fname_bname
            print(untar_cmd)
            self.connection.send_command(untar_cmd, client = 'target')
            
            fname = self.temp_folder_r + '/' + fname_bname
        return fname
    
      
    def retrieve_data(self, variables, date, model_res = 'fine', 
                          mode = 'analysis', forecast_start = None, 
                          coord_bounds = None, vert_levels = None, 
                          assign_heights = True):
        
        """
        FUNCTION:
            data = retrieve_data(time, model_res = 'fine', 
                           mode = 'analysis', forecast_start = None, 
                           coord_bounds = None)
        
        PURPOSE:
             Retrieves COSMO variables from the cscs server for a given time
             (by interpolating in time if the time falls between two outputs)
        
        INPUTS:
            date : the date of the desired simulation in the YYYY-mm-DD HH:MM 
                format, for example 2017-02-27 13:46
            model_res : the resolution of the model, either 'fine' (COSMO-2 or
                COSMO-1, since 2017) or 'coarse' (COSMO-7)
            mode : the simulation mode can be either 'analysis' or 'forecast'
            forecast_start : (optional) only needed for mode = 'forecast'.
                Specifies the start of the forecast, for exemple
                if time = 2017-02-27 13:46, you could for ex. set 
                forecast_start = 2017-02-27 12:00, which correspond
                to 1h36 of forecast or forecast_start = 2017-02-27 
                9:00 which would correspond to 3h36 of forecast.
                Note that the forecast_start time must agree
                with the parameters COSMO_FORECAST_STARTS and
                FORECAST_DURATION in the config file. 
            coord_bounds : (optional) the bounds of the domain on which the 
                data should be retrieved. This will make the 
                query faster and further data extraction faster.
                When not provided the whole COSMO domain will be 
                retrieved. Must be a tuple of the format
                ([lon_llc,lat_llc],[lon_upr,lat_upr]) where llc
                is the lower left corner and upr the upper right 
                corner
            vert_levels : (optional) the vertical levels to be retrieved (for
                3D variables), if not provided all levels will be retrieved
            assign_heights : (optional)
        OUTPUTS:
            data : a dictionary containing the data, every retrieved variable
                is a special numpy array with an addiitional field called
                "metadata" which contains attributes about the variable which
                are obtained from the original grib file. Note that besides the
                retrieved variables this dictionary contains variables with 
                names starting by x_. y_, z_, lat_, lon_ and followed by an 
                integer. These variables correspond to the
                system of coordinates used by COSMO: x and y are the coordinates
                in the local COSMO rotated coordinates, z_ are the altitude levels
                and lat_ and lon_ are the real WGS84 coordinates corresponding
                to the x_ and y_ coordinates. The integer defines the
                grid_mapping (the COSMO variables are defined on different
                coordinates system). You can obtain the grid mapping number of
                a given variable with myvar.metdata['mapping']
        
        """ 
        # Show warning converning vertical levels
        if vert_levels != None:
            msg = """
            You have specified vertical levels
            Please note that all 2D variables will be ignored when using this
            option...
            """
            print(msg)
            
        # First create remote temporary folder
        folder_code = '/tmp_' + str(uuid.uuid4()).replace('-','_') # unique id
        self.temp_folder_r = '/users/'+self.connection.username + '/' + \
            folder_code + '/'
        cmd_mdkir  = 'mkdir ' + self.temp_folder_r
        print('Creating temporary folder in home folder :'+self.temp_folder_r)
        self.connection.send_command(cmd_mdkir, client = 'target')

        # Then we create local temporary folder
        self.temp_folder_l = cfg.LOCAL_FOLDER +'/' + folder_code +'/'
        print('Creating local temporary folder in :' + self.temp_folder_l)
        os.makedirs(self.temp_folder_l)

        # Check coordinates
        if coord_bounds:
            try:
                llc = np.array(coord_bounds[0]) - 0.1 # Take some security margin
                upr =  np.array(coord_bounds[1]) + 0.1
            except:
                raise ValueError('Invalid format for coord_bounds, must be a tuple '+\
                                 'of the form ([lon_llc,lat_llc],[lon_upr,lat_upr])')  
                   
        # Check variables
        valid_vars = []
        available_vars = self.get_available_vars(date, model_res, mode)
        
        for v in variables:
            if v in available_vars:
                valid_vars.append(v)
            else:
                msg = """
                {:s} is not a valid variable name, ignoring it...
                Tip : use the "get_available_vars class function to get all
                available variables or check the file logs in the
                cosmo_info/file_logs directory
                """
                print(msg)

        t = datetime.datetime.strptime(date,'%Y-%m-%d %H:%M')
        
        if forecast_start:
            forecast_start = datetime.datetime.strptime(forecast_start,
                                                       '%Y-%m-%d %H:%M')
            
        if t < cfg.COSMO_2_END:
            is_COSMO1 = False
        else:
            is_COSMO1 = True
            
        t0 = t - datetime.timedelta(minutes = t.minute)
        t1 = t + datetime.timedelta(minutes = 60 - t.minute)
                            
        if mode == 'analysis':
            cosmo_f0 = self.get_COSMO_fname(t0, mode = mode)
            cosmo_f1 = self.get_COSMO_fname(t1, mode = mode)
        elif mode == 'forecast':          
            t1 = t + datetime.timedelta(minutes = 60 - t.minute)

                
            cosmo_f1 = self.get_COSMO_fname(t1, mode = mode,
                                        forecast_start = forecast_start)
            if t0 < forecast_start: # if t0 is after forecast start
                # we set forecast_start = None so the appropriate forecast
                # start time is automatically computed in get_COSMO_fname
                forecast_start = None
                
            cosmo_f0 = self.get_COSMO_fname(t0, mode = mode,
                            forecast_start = forecast_start)     
                        

        interpolate_time = False
        if t == t0:
            files_to_get = [cosmo_f0]
        elif t == t1:
            files_to_get = [cosmo_f1]
        else:
            files_to_get = [cosmo_f0, cosmo_f1]
            interpolate_time = True
            
        # Generate PMSL field to get local coordinates
        if model_res == 'fine':
            if is_COSMO1:
                name_PMSL_file = 'PMSL_cosmo_1.nc'
            else:
                name_PMSL_file = 'PMSL_cosmo_2.nc'   
        else:
            name_PMSL_file = 'PMSL_cosmo_7.nc'
            
        
        # Here is why things might go wrong so we put a try/except and keep 
        # a list of commands that might have failed
        
        serv_cmds = []
        try:
            if not os.path.exists(cfg.LOCAL_FOLDER + name_PMSL_file):
                cmd_filter = cfg.FX_DIR + 'fxfilter --force -s PMSL -o ' + \
                    self.temp_folder_r + '/PMSL.grb ' + cosmo_f0 
                cmd_convert = cfg.FX_DIR + 'fxconvert --force -o '+ self.temp_folder_r \
                    + name_PMSL_file +' nc ' + self.temp_folder_r + '/PMSL.grb'
                    
                serv_cmds.append(cmd_filter)
                serv_cmds.append(cmd_convert)
                
                self.connection.send_command(cmd_filter, client = 'target')
                self.connection.send_command(cmd_convert, client = 'target')        
                
                # Download corresponding data to localhost
                self.connection.ftp.get(str(self.temp_folder_r + name_PMSL_file),
                                        str(cfg.LOCAL_FOLDER + name_PMSL_file))            
    
            pmsl_f0  = netCDF4.Dataset(cfg.LOCAL_FOLDER+name_PMSL_file)
            
            # Get latitude and longitude of files
            lat = pmsl_f0.variables['lat_1'][:]
            lon = pmsl_f0.variables['lon_1'][:]
            
            if not coord_bounds:
                j_min = 1
                i_min = 1
                j_max = lat.shape[0]
                i_max = lat.shape[1] 
                
            else:
                # Get indexes in i and j local model indexes to crop
                domain = np.logical_and(np.logical_and(lat < upr[1], 
                                                       lon < upr[0]),
                                                       np.logical_and(lat > llc[1],
                                                                      lon > llc[0]))
                idx = np.where(domain == 1)
                
                j_min = np.min(idx[0])
                j_max = np.max(idx[0])
                i_min = np.min(idx[1])
                i_max = np.max(idx[1])
                
            pmsl_f0.close()
    
            # Now treat files
            for i,f in enumerate(files_to_get):
                # (1) FILTER
                if vert_levels != None:
                    cmd_filter = cfg.FX_DIR + 'fxfilter --force -s ' + ','.join(valid_vars) + \
                        ' -l '+','.join(str(i) for i in vert_levels)+\
                        ' -o '+self.temp_folder_r + '/filtered' + str(i) + '.grb ' + f 
                else:
                    cmd_filter = cfg.FX_DIR + 'fxfilter --force -s ' + ','.join(valid_vars) + \
                        ' -o '+self.temp_folder_r + '/filtered' + str(i) + '.grb ' + f 
                self.connection.send_command(cmd_filter, client = 'target')
                # (2) CROP
                cmd_crop = cfg.FX_DIR + 'fxcrop --force -i ' + \
                    ','.join([str(i_min),str(i_max)]) + \
                    ' -j ' + ','.join([str(j_min),str(j_max)]) + \
                    ' -o '+self.temp_folder_r + '/crop' + str(i) + '.grb '+ \
                    self.temp_folder_r +'/filtered'+str(i)+'.grb'
                self.connection.send_command(cmd_crop, client = 'target')  
    
                # (3) CONVERT TO NETCDF
                cmd_convert = cfg.FX_DIR + 'fxconvert --force -o '+ self.temp_folder_r \
                    +'/convert' + str(i) + '.nc' + ' nc ' + self.temp_folder_r + \
                    '/crop'+str(i)+'.grb'
                self.connection.send_command(cmd_convert, client = 'target')
    
                serv_cmds.append(cmd_filter)
                serv_cmds.append(cmd_crop)
                serv_cmds.append(cmd_convert)
                
                # (4) DOWNLOAD
                fname = 'convert' + str(i)+'.nc'
                # Download corresponding data to localhost
                print('Retrieving file ' + self.temp_folder_r + '/' + str(fname))
                self.connection.ftp.get(str(self.temp_folder_r + '/' + str(fname)),
                                        str(self.temp_folder_l + fname))
                
                

                
        except:
            print('Something has failed on the server side...')
            print('Please check manually if any of these commands fails :')
            for cmd in serv_cmds:
                print(cmd)
            print('---')
            print('The query will now abort')

            # Delete local and remote temp folders
            cmd_rm  = 'rm -r ' + self.temp_folder_r
            self.connection.send_command(cmd_rm, client = 'target')
            shutil.rmtree(self.temp_folder_l, ignore_errors=True)
            
            os.remove(cfg.LOCAL_FOLDER+name_PMSL_file)
            
            return
        
        f0 = netCDF4.Dataset(self.temp_folder_l + '/convert0.nc')

        if interpolate_time:
            f1 = netCDF4.Dataset(self.temp_folder_l + '/convert1.nc')
            delta = float((t - t0).seconds)

        variables = {}
        for var in f0.variables.keys():
            if var in valid_vars:
                
                mapping = int(re.findall(r'\d+',f0.variables[var].grid_mapping)[0])
                if interpolate_time:
                    offset = (f1.variables[var][:] - f0.variables[var][:])/3600. * delta
                    data = np.squeeze(f0.variables[var][:] + offset)
                else:
                    data = np.squeeze(f0.variables[var] [:])
                    
                variables[var] = MetaArray(data,mapping = mapping)
                
                # Assign all variable attributes to metadata
                for key in f0.variables[var].__dict__.keys():
                    if 'grid_mapping' in key:
                        continue # We ignore this key as it is redundant with
                                 # the "mapping" key
                    variables[var].metadata[key] = getattr(f0.variables[var],key)

            elif 'x_' in var or 'y_' in var :
                # Other variables that we need to add to our output
                
                coord = f0.variables[var][:]
                # For whatever reason fieldextra sometimes write 360 + x/y
                # instead of x or y, for example 357 instead of -3
                # This check should fix that
                if np.mean(coord) > 180:
                    coord = coord - 360
                variables[var] = MetaArray(coord)
                for key in f0.variables[var].__dict__.keys():
                    variables[var].metadata[key] = getattr(f0.variables[var],key)     
                    
            elif 'lat_' in  var or 'lon_' in var :
                variables[var] = MetaArray(f0.variables[var])
                for key in f0.variables[var].__dict__.keys():
                    variables[var].metadata[key] = getattr(f0.variables[var],key)      
                    
            elif 'z_' in  var and not assign_heights :
                variables[var] = MetaArray(f0.variables[var])
                for key in f0.variables[var].__dict__.keys():
                    variables[var].metadata[key] = getattr(f0.variables[var],key)      
                    
            elif 'grid_mapping' in var:
                variables[var] = MetaArray(np.array(['nodata']))
                # Add all info about the grid mapping to our output
                for key in f0.variables[var].__dict__.keys():
                    variables[var].metadata[key] = getattr(f0.variables[var],key)
            
        for var in valid_vars:
            if var not in variables.keys():
                print('Variable '+var+' could not be found in COSMO file')
                print('It might not be present in MCH standard output')
                valid_vars.remove(var)

        # Assign ztypes to 3D variables
        for  var in valid_vars:
            # Some variables are on full levels (ex. W) some on half-levels
            # (ex. T, P)
            if variables[var].ndim == 3:
                all_dim = f0.variables[var].dimensions
                z_dim = fnmatch.filter(all_dim, 'z_*')[0]
                z_dim_no = z_dim[-1]
                if 'z_bnds_' + str(z_dim_no) in f0.variables.keys():
                    variables[var].metadata['ztype'] = 'half'
                    variables[var].metadata['zbnds'] = \
                        f0.variables['z_bnds_' + str(mapping)][:]
                else:
                    variables[var].metadata['ztype'] = 'full'
                
        f0.close()
        if interpolate_time:
            f1.close()
        
        # Finally, we also assign the appropriate heights if at least one var 
        # is 3D
        
        if np.any([variables[v].ndim == 3 for v in valid_vars]) and assign_heights:
            current_folder = inspect.getfile(inspect.currentframe()) 
            current_folder = os.path.dirname(current_folder)
            
            if model_res == 'fine':
                if is_COSMO1:
                    cosmo_heights = np.load(current_folder + 
                                            '/cosmo_info/cosmo_1_heights.npz')
                else:
                    cosmo_heights = np.load(current_folder + 
                                            '/cosmo_info/cosmo_2_heights.npz')
            else:
                cosmo_heights = np.load(current_folder + 
                                            '/cosmo_info/cosmo_7_heights.npz')
                
            heights = cosmo_heights['arr_0'] # arr_0 =  heights, arr_1 = x, arr_2 = y
            x_heights = cosmo_heights['arr_1']
            y_heights = cosmo_heights['arr_2']
    
            # Cut heights to the same domain as our variables
            heights_cut = heights[:,j_min-1:j_max,i_min-1:i_max]
            x_heights = x_heights[i_min-1:i_max]
            y_heights = y_heights[j_min-1:j_max]
    
            # Now assign to every mapping a height
            all_mappings = [variables[k].metadata['mapping'] for k in valid_vars]
            
            for mapping in np.unique(all_mappings):
                x = variables['x_'+str(mapping)] 
                y = variables['y_'+str(mapping)] 
                
                # Tolerance is set to 0.002 degrees = 100 m, should be enough
                cond1 = np.allclose(x, x_heights, atol = 0.002) # condition on x
                cond2 = np.allclose(y, y_heights, atol = 0.002) # condition on y
                
                if cond1 and cond2:
                    # W for example
                    final_heights = heights_cut
                    variables['z_'+str(mapping)] = heights_cut
                elif cond1 and not cond2:
                    # U for example
                    if np.allclose(y[0:-1], 0.5 * (y_heights[1:] + y_heights[0:-1])):
                        final_heights = 0.5 * (heights_cut[:,0:-1,:] + heights_cut[:,1:,:])
                        # Pad with NaN to account for the last column which is unknown
                        final_heights = np.pad(final_heights,((0,0),(0,1),(0,0)),'constant',
                                          constant_values=np.nan)
                
                elif not cond1 and cond2:
                     # V for example
                    if np.allclose(x[0:-1], 0.5 * (x_heights[1:] + x_heights[0:-1])):
                        final_heights = 0.5 * (heights_cut[:,:,0:-1] + heights_cut[:,:,1:])
                        # Pad with NaN to account for the last column which is unknown
                        final_heights = np.pad(final_heights,((0,0),(0,0),(0,1)),'constant',
                                          constant_values=np.nan)
                # This checks if at least one of the variables with this mapping
                # is defined on half levels
                
                half_lvl_present = np.any([variables[k].metadata['ztype'] == 'half' \
                                           for k in valid_vars if \
                                           variables[k].metadata['mapping'] == mapping])
    
                # This checks if at least one of the variables with this mapping
                # is defined on full levels
                full_lvl_present = np.any([variables[k].metadata['ztype'] == 'full' \
                                           for k in valid_vars if \
                                           variables[k].metadata['mapping'] == mapping])
                                
                if full_lvl_present:
                    if vert_levels != None:
                        full_heights = final_heights[vert_levels,:,:]
                    else:
                        full_heights = final_heights
                    variables['heights_'+str(mapping)+'_full'] = full_heights
                    
                if half_lvl_present:
                    z_bnds = [variables[k].metadata['zbnds'] \
                        for k in valid_vars if \
                        variables[k].metadata['mapping'] == mapping and \
                        variables[k].metadata['ztype'] == 'half' ]
                    
                    z_bnds = z_bnds[0].astype(int)
                    
                    half_heights = 0.5 * (final_heights[z_bnds[:,0]-1,:,:] + \
                           final_heights[z_bnds[:,1]-1,:,:])
                    
                    variables['heights_'+str(mapping)+'_half'] = half_heights
                    
        variables['retrieved_variables'] = np.array(valid_vars )
        self.data = variables
        
        
        # Finally delete temporary folder
        cmd_rm  = 'rm -r ' + self.temp_folder_r
        self.connection.send_command(cmd_rm, client = 'target')
        
        
        # Local temp folder
        shutil.rmtree(self.temp_folder_l, ignore_errors=True)

    
        return variables

def save_netcdf(data, fname, write_heights = True, compress = True):
    """
    FUNCTION:
        save_netcdf(data, fname)
    
    PURPOSE:
        Saves a data structure (output of a COSMO query) to a netCDF file
    
    INPUTS:
        data : data structure as obtained with the retrieve_data function of 
               the COSMO_query class
        fname : the full name (path) of the netCDF file to be written
        write_heights : if set to False, the altitudes corresponding to the
            model levels will not be written to the file
    """
    if type(data) != dict:
        raise ValueError('Invalid data format, must be dictionary, as obtained ',
                         'with the retrieve_data function')
    
    nc = netCDF4.Dataset(fname,'w')
    for var in data['retrieved_variables']:
        # Get mapping
        mapping = data[var].metadata['mapping']
        
        # Create dimensions if not created already
        x_key = 'x_'+str(mapping)
        if x_key not in nc.dimensions.keys():
            nc.createDimension(x_key, len(data[x_key]))
        if x_key not in nc.variables.keys():
            nc_x = nc.createVariable(x_key,
                                       'f4', (x_key))          
        nc_x[:] = data[x_key]
        
        y_key = 'y_'+str(mapping)            
        if y_key not in nc.dimensions.keys():
            nc.createDimension(y_key, len(data[y_key]))
        if y_key not in nc.variables.keys():
            nc_y = nc.createVariable(y_key,
                                       'f4', (y_key))          
        nc_y[:] = data[y_key]
            
        
        if data[var].ndim == 3:
            ztype = data[var].metadata['ztype']
            z_key =  'heights_'  +str(mapping) + '_' + ztype
        
            if z_key not in nc.dimensions.keys():
                len_z = len(data[var])
                nc.createDimension( z_key , len_z)           
                
            if (z_key not in nc.variables.keys()) and write_heights :
                nc_z = nc.createVariable(z_key,     
                                         'f4', (z_key,y_key,x_key), zlib=True)          
                nc_z[:] = [data[z_key]]
            # Now create variable
            nc_var = nc.createVariable(var, 'f4', (z_key,y_key, x_key), zlib=True)            
        
        elif data[var].ndim == 2:            
            nc_var = nc.createVariable(var, 'f4', (y_key,x_key), zlib=True)

        nc_var[:] = data[var]
        
        # Also assign variable attributes
        for key in data[var].metadata.keys(): 
            if key != '_FillValue': # FillValue cannot be added a posteriori
                setattr(nc_var,key,data[var].metadata[key])
        
        # Also assign lat/lon variables if not assigned already
        lon_key = 'lon_'+str(mapping)
        if lon_key not in nc.variables.keys():
            nc_lon = nc.createVariable(lon_key,
                                       'f4', (y_key,x_key))
            nc_lon[:] = data[lon_key]
            for key in data[lon_key].metadata.keys(): 
                setattr(nc_lon,key,data[lon_key].metadata[key])
                
        lat_key = 'lat_'+str(mapping)
        if lat_key not in nc.variables.keys():
            nc_lat = nc.createVariable(lat_key,
                                       'f4', (y_key,x_key))                
            nc_lat[:] = data[lat_key]
            for key in data[lat_key].metadata.keys(): 
                setattr(nc_lat,key,data[lat_key].metadata[key])
                
        # Also assign grid_mapping variables with their attributes if not 
        # assigned already
        grid_mapping_key = 'grid_mapping_' + str(mapping)
        if grid_mapping_key not in nc.variables.keys():
            nc_grid = nc.createVariable(grid_mapping_key,'c')
            for key in data[grid_mapping_key].metadata.keys(): 
                setattr(nc_grid,key,data[grid_mapping_key].metadata[key])

    # Finally put retrieved_variables array (= list of variables we got) into
    # global attribute
    nc.retrieved_variables = data['retrieved_variables']
    
    nc.close()

def load_netcdf(fname):
    """
    FUNCTION:
        data = load_netcdf(name)
    
    PURPOSE:
         Loads a netCDF file as written by the save_netcdf function to a data
         structure similar to the one obtained with the retrieve_data function
         of, that can be used for data extraction 
    
    INPUTS:
        fname : the full name (path) of the netCDF file to be read
    
    OUTPUTS:
        data : the data structure corresponding to the loaded netCDF file
    """
    
    data = {}
    nc = netCDF4.Dataset(fname,'r')
    
    for var in nc.variables.keys():
        data[var] = MetaArray(nc.variables[var])
        for att in nc.variables[var].__dict__.keys():
            data[var].metadata[att] = getattr(nc.variables[var],att)
            
    # Finally get retrieved_variables array (= list of variables we got) from
    # global attribute
    data['retrieved_variables'] = nc.retrieved_variables
        
    return data
     
if __name__ == '__main__':
    from cosmo_query import config, SSH, Query
    from cosmo_query import save_netcdf, load_netcdf
    from cosmo_query import extract, coords_profile
    ##
    # We initiate a connection to ela.cscs.ch, with username and password
    # specified in the config.py file (password is not needed if ssh key is 
    # defined)
    connection = SSH(config.ELA_ADRESS,'wolfensb')
    #
    # We also need to open a channel to kesch.cscs.ch since it 
    connection.open_channel(config.KESCH_ADRESS,'wolfensb')
    
    # Now that the connection is setup we can create a Query instance
    query = Query(connection)
    
    # And use this query to retrieve some data
    
    variables = ['T','P','QV'] # we want temperature and pressure
    date = '2015-05-31 12:00' # for the 31th May 2016 at 12h30
    model_res = 'fine' # at high resolution
    mode = 'analysis' # In analysis mode
    coord_bounds = ([6.6,45.8],[8.4,46.6]) # Over an area covering roughly the Valais

    
    data = query.retrieve_data(variables, date, model_res = 'fine', 
                              mode = 'analysis', coord_bounds = coord_bounds)
    
    options_RHI = {}
    options_RHI['beamwidth'] = 1.5 # 1.5 deg 3dB beamwidth, as MXPol
    options_RHI['azimuth'] = 47
    options_RHI['rrange'] = np.arange(200,10000,75) # from 0.2 to 10 km with a res of 75 m
    options_RHI['npts_quad'] = [3,3] # 3 quadrature points in azimuthal, 3 in elevational directions
    options_RHI['rpos'] = [7.0923,46.1134,500] # Radar position in lon/lat/alt
    options_RHI['refraction_method'] = 1 # 1 = standard 4/3, 2 = ODE refraction 
    
    rhi_T = extract(data,['T'],'RHI',options_RHI) 