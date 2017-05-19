import datetime
import numpy as np


ELA_ADRESS = 'ela.cscs.ch'
KESCH_ADRESS = 'kesch.cscs.ch'
USERNAME = 'Enter username here'
PASSWORD = 'Enter password here'
LOCAL_FOLDER = '/tmp/'
FX_DIR = '/oprusers/osm/opr/bin/'
COSMO_ROOT_DIR = '/store/s83/'
COSMO_FORECAST_STARTS = np.array([0,3,6,9,12,15,18,21]) # all hours were a forecast is started
FORECAST_DURATION = 48
COSMO_2_END = datetime.datetime.strptime('2016/09/27 09:00','%Y/%m/%d %H:%M')

