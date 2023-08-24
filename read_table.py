import pandas as pd
import obspy
from obspy.core import UTCDateTime

windows = 20    # length of cut window
before_arrival = 5  # the start of the cut window
data_path = './continue_data'

df = pd.read_excel('./excel_info/data_list.xlsx')
hr_min_sec = df['arrival_time(hr:min:sec.msec)'].values      # hour, minute, second
yr_mth_day = df['Date'].values                               # year, month, day
recv_num = df['Receiver'].values                             # station number

for i in range(len(hr_min_sec)):
    tt = (hr_min_sec[i])[0: 8]                               # record hour, minute & second
    day = (yr_mth_day[i])[2: 4] + (yr_mth_day[i])[5: 7] + (yr_mth_day[i])[8: 10]   # record year, month & day
    recv = recv_num[i]                # record station number
    
    # Index file name
    indicate_filename = (yr_mth_day[i])[2: 4] + (yr_mth_day[i])[5: 7] + (yr_mth_day[i])[8: 10] + \
        '.' + (hr_min_sec[i])[0: 2] + '0000.EB000' + str(recv_num[i])
        
    # Recorded vehicle arrival time
    vehicle_arri_tt = UTCDateTime((yr_mth_day[i])[0: 4] + '-' + (yr_mth_day[i])[5: 7] + '-' + \
                                  (yr_mth_day[i])[8: 10] + 'T' + (hr_min_sec[i])[0: 8] + '+08:00')
    
    for component in ['E', 'N', 'Z']:
        input = obspy.read(data_path + '/' + indicate_filename + '.EH' + component + '.sac')[0]
        
        # Cut data
        input_copy = input.copy()
        input_copy.trim(vehicle_arri_tt - before_arrival, vehicle_arri_tt - before_arrival + windows)
        
        input_copy.data = input_copy.data[: -1]
        
        output_name = (yr_mth_day[i])[2: 4] + (yr_mth_day[i])[5: 7] + (yr_mth_day[i])[8: 10] + \
            '.' + (hr_min_sec[i])[0: 2] + (hr_min_sec[i])[3: 5] + (hr_min_sec[i])[6: 8] + \
                '.EB000' + str(recv_num[i]) + '.EH' + component + '.sac'
        output_path = './events/' + output_name
        
        if input_copy.data.shape[0] == 20000:
            input_copy.write(output_path, format='SAC')

print('Data clipping is complete.')