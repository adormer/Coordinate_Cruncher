# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 12:43:48 2024

@author: kacper
"""

#%%
import h5py
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path
import cv2
import os
#%%
#path and file names

video = ('1_SC162-F-L0_d14')
file_path = 'E:/august/videos/14dUS-G1/OFT_20240716_Day14/'
file_name = video + 'DLC_resnet50_OFT-pylonJul8shuffle1_150000'

name = file_path + file_name + '.h5'

#print('file name:',name)

#center boundaries
x_min = 460
x_max = 833
y_min = 298
y_max = 678

#scale modifier for the external boundaries given the center bounds. added 10 to minimums to compensate for the off-center camera positioning.
exp = 155
ext_x_min = int(x_min - (exp+10))
ext_x_max = int(x_max + exp)
ext_y_min = int(y_min - (exp+10))
ext_y_max = int(y_max + exp)

#generate an array containing all the H5 files in a given folder (path)
def fn_directory_path(path):
    filenames = []
    #stores the directory path as a string
    directory_in_str = path
    
    #encode the directory as desired data type which allows it to be iterable (???) 
    directory = os.fsencode(directory_in_str)
    
    for file in os.listdir(directory):
        
        #decodes the filename from a byte back qinto a str
        filename = os.fsdecode(file)
        if filename.endswith(".h5"):
            
            filenames.append(np.array(directory_in_str+filename))
            
            continue 
    return(filenames)

#filenames = fn_directory_path('C:/Users/kacper/Desktop/shaorong_sample_data/')
#print(filenames)

#%%
# determining the H5 file shape/ organization

def list_datasets(group, path=""):
    """
    Recursively list all datasets in the given HDF5 group.
    
    Parameters:
    - group: h5py.Group or h5py.File
    - path: str, current path in the HDF5 file structure
    
    Returns:
    - list of paths to datasets
    """
    datasets = []
    for key in group:
        item = group[key]
        if isinstance(item, h5py.Dataset):
            datasets.append(f"{path}/{key}")
        elif isinstance(item, h5py.Group):
            datasets.extend(list_datasets(item, path=f"{path}/{key}"))
    return datasets

#for files in filenames:
    
with h5py.File(name, 'r') as f:
    f.attrs.get('model_config')
    ls=list(f.keys())
    datasets= list_datasets(f)
    #print('list of datasets in this file:', ls)
    datasets = list_datasets(f)
    #print(datasets)
    
    #root group
    data_root = f.get('/df_with_missing', default=None, getclass=False,getlink=False)
    df_with_missing = np.array(data_root)
    #print('what is in df with missing:', df_with_missing, )
    
    #2nd tear data
    data_i_table = f.get('/df_with_missing/_i_table/', default=None, getclass=False,getlink=False)
    _i_table = np.array(data_i_table)
    #print('what is in _i_table:', _i_table)
    
    data_table = f.get('/df_with_missing/table/', default=None, getclass=False,getlink=False)
    table = np.array(data_table)
    #print('what is in table:', table)
    
    #3rd tear data
  #  data_index = f.get('/df_with_missing/_i_table/index/', default=None, getclass=False, getlink=False)
   # index = np.array(data_index)
   # print ('what is in index:', index)
    
    #4th tear data
    #data_abounds = f.get()
    
#%% parsing the array containing the position data

#try to capture the nose only!
nose = []
nose_x = []
nose_y = []
nose_p = []

head = []
head_x = []
head_y = []
head_p = []

neck = []
neck_x = []
neck_y = []
neck_p = []

leftear = []
leftear_x = []
leftear_y = []
leftear_p = []

rightear = []
rightear_x = []
rightear_y = []
rightear_p = []

body = []
body_x = []
body_y = []
body_p = []

tailbase = []
tailbase_x = []
tailbase_y = []
tailbase_p = []

count=0

for n in table:
    """
    all of the position data is stored in a matrix in the variable: table
    this for loop goes into the table array and chooses the nested array based on the current frame (n), this new array is stored as frame
    the x y data for each body part for each frame is stored in a second one row array in frame at [1] and this array is stored to frame_data
    The first two data points are the x and y position of the nose and are added to respective lists called nose_x and nose_y.
    The loop iterates through all frames to create lists of the position data for the nose at every frame
    
    """
 
    frame = table[count]
    frame_data = frame[1]
    
    nose_x.extend(frame_data[0:1])
    nose_y.extend(frame_data[1:2])
    nose_p.extend(frame_data[2:3])
    
    head_x.extend(frame_data[3:4])
    head_y.extend(frame_data[4:5])
    head_p.extend(frame_data[5:6])
    
    neck_x.extend(frame_data[6:7])
    neck_y.extend(frame_data[7:8])
    neck_p.extend(frame_data[8:9])
    
    leftear_x.extend(frame_data[9:10])
    leftear_y.extend(frame_data[10:11])
    leftear_p.extend(frame_data[11:12])
    
    rightear_x.extend(frame_data[12:13])
    rightear_y.extend(frame_data[13:14])
    rightear_p.extend(frame_data[14:15])
    
    body_x.extend(frame_data[15:16])
    body_y.extend(frame_data[16:17])
    body_p.extend(frame_data[17:18])
    
    tailbase_x.extend(frame_data[18:19])
    tailbase_y.extend(frame_data[19:20])
    tailbase_p.extend(frame_data[20:21])
    
    count += 1
    
#making the lists into arrays
nose_x_array = np.array(nose_x)
nose_y_array = np.array(nose_y)
nose_p_array = np.array(nose_p)

head_x_array = np.array(head_x)
head_y_array = np.array(head_y)
head_p_array = np.array(head_p)

neck_x_array = np.array(neck_x)
neck_y_array = np.array(neck_y)
neck_p_array = np.array(neck_p)

leftear_x_array = np.array(leftear_x)
leftear_y_array = np.array(leftear_y)
leftear_p_array = np.array(leftear_p)

rightear_x_array = np.array(rightear_x)
rightear_y_array = np.array(rightear_y)
rightear_p_array = np.array(rightear_p)

body_x_array = np.array(body_x)
body_y_array = np.array(body_y)
body_p_array = np.array(body_p)

tailbase_x_array = np.array(tailbase_x)
tailbase_y_array = np.array(tailbase_y)
tailbase_p_array = np.array(tailbase_p)


#vertically stacks arrays and then transposes them 
nose = np.vstack((nose_x_array, nose_y_array, nose_p_array)).T
head = np.vstack((head_x_array, head_y_array, head_p_array)).T
neck = np.vstack((neck_x_array, neck_y_array, neck_p_array)).T
leftear = np.vstack((leftear_x_array, leftear_y_array, leftear_p_array)).T
rightear = np.vstack((rightear_x_array, rightear_y_array, rightear_p_array)).T
body = np.vstack((body_x_array, body_y_array, body_p_array)).T
tailbase = np.vstack((tailbase_x_array, tailbase_y_array, tailbase_p_array)).T


#print('what is in nose :', nose)
#print('what is in head :', head)
#print('what is in neck :', neck)
#print('what is in leftear :', leftear)
#print('what is in rightear :', rightear)
#print('what is in body :', body)
#print('what is in tailbase :', tailbase)


#print('nose shape!:', np.shape(nose))
#print('head shape!:', np.shape(head))
#print('neck shape!:', np.shape(neck))
#print('leftear shape!:', np.shape(leftear))
#print('rightear shape!:', np.shape(rightear))
#print('body shape!:', np.shape(body))
#print('tailbase shape!:', np.shape(tailbase))

#%% line plot of the mouse movement

#plt.figure(figsize=(7,7))
#plt.plot(body_x_array[:18000], 1000-body_y_array[:18000], 'r')

#THESE ARE THE ONES THAT WORK WITH SHOU_RONG PREVIOUS PROGRAM
#plt.figure(figsize=(5,5))
#plt.plot(body_x_array[:18000], body_y_array[:18000], 'r')
#plt.savefig(f'C:/Users/kacper/Desktop/shaorong_sample_data/OFT_PSI_7dUS-G2/{video}_plot.svg', format = 'svg', dpi = 300)

#plt.figure(figsize=(7,7))
#plt.plot(1000-body_x_array[:18000], 1000-body_y_array[:18000], 'r')
#plt.figure(figsize=(7,7))
#plt.plot(1000-body_x_array[:18000], body_y_array[:18000], 'r')


#plt.plot(tailbase_x_array[:18000], tailbase_y_array[:18000], 'r')

#%% defining global variables

#the length of one pixel in centimeters
len_of_pixel = 0.060066847727576

#the duration of one frame in seconds
frame_duration = 1/30


video_name = file_path + video + '.avi'

#capturing the video
cap = cv2.VideoCapture(video_name)
fps = int(cap.get(cv2.CAP_PROP_FPS))



#%%
def fn_inst_velocity(frame, body_part):
    
    if frame >= len(body_x_array)-1:
        frame += -1
        
    if body_part == 'nose':
        x_1 = nose_x_array[frame]
        y_1 = nose_y_array[frame]
        
        x_2 = nose_x_array[frame+1]
        y_2 = nose_y_array[frame+1]
    elif body_part == 'head':
        x_1 = head_x_array[frame]
        y_1 = head_y_array[frame]
        
        x_2 = head_x_array[frame+1]
        y_2 = head_y_array[frame+1]
    elif body_part == 'neck':
        x_1 = neck_x_array[frame]
        y_1 = neck_y_array[frame]
        
        x_2 = neck_x_array[frame+1]
        y_2 = neck_y_array[frame+1]
    elif body_part == 'leftear':
        x_1 = leftear_x_array[frame]
        y_1 = leftear_y_array[frame]
        
        x_2 = leftear_x_array[frame+1]
        y_2 = leftear_y_array[frame+1]
    elif body_part == 'rightear':
        x_1 = rightear_x_array[frame]
        y_1 = rightear_y_array[frame]
        
        x_2 = rightear_x_array[frame+1]
        y_2 = rightear_y_array[frame+1]
    elif body_part == 'body':
        x_1 = body_x_array[frame]
        y_1 = body_y_array[frame]
        
        x_2 = body_x_array[frame+1]
        y_2 = body_y_array[frame+1]
    elif body_part == 'tailbase':
        x_1 = tailbase_x_array[frame]
        y_1 = tailbase_y_array[frame]
        
        x_2 = tailbase_x_array[frame+1]
        y_2 = tailbase_y_array[frame+1]
        
        ##pythagoreans theorem to determine distance between the two points 
    distance = np.sqrt(((x_1-x_2)**2)+((y_1-y_2)**2))
    velocity = (distance*len_of_pixel)/frame_duration
    
    #return print('velocity at frame', frame,'is equal to:', f'{velocity: .05f}', 'cm/sec')
    #return(distance)
    return float(f'{velocity: .03f}')


# Average velocity over 90 frames (3 senconds for 30 fps)

def fn_avg_velocity (frame, body_part):
    
    velocity_frames = np.empty((1,45))
    count = 0
    total_velocity = 0
    
    for f in range(len(velocity_frames)):
        if f == len(velocity_frames)-90:
            break
        total_velocity += fn_inst_velocity(frame+count, body_part)
        total_velocity += fn_inst_velocity(frame-count, body_part)
        count += 1
        
    final_velocity = total_velocity/(len(velocity_frames)*2)
    return float(f'{final_velocity: .01f}')

#overall average velocity

def fn_overall_velocity (body_part):
    
    count = 0
    total_velocity = 0
    
    for f in range(len(body_x_array)):
        if f == fps*600:
            break
        total_velocity += fn_inst_velocity(count, 'body')
        count += 1
        
    final_velocity = total_velocity/(fps*600)
    return float(f'{final_velocity: .01f}')


#calculating the total distance travelled over specified time range

def fn_distance_travelled (time_beginning, time_end):
    
    #truncating the position arrays to the desired time ranges
    body_x_1 = body_x_array[time_beginning: time_end+1]
    body_y_1 = body_y_array[time_beginning: time_end+1]
    
    
    counter = 0
    distance = []
    
#calculates the 
    for n in range(len(body_x_1)):
        if n == fps*600:
            break
        x_1 = body_x_1[counter]
        y_1 = body_y_1[counter]
            
        x_2 = body_x_1[counter-1]
        y_2 = body_y_1[counter-1]
       
        distance.append(np.sqrt(((x_1-x_2)**2)+((y_1-y_2)**2)))
        counter += 1
        
    distance_1 = np.cumsum(distance[1:])
    total_distance = distance_1[-1]
    t_distance_cm = total_distance * len_of_pixel
    #return(total_distance)
    return float(f'{t_distance_cm:.02f}')
    

def fn_time_center ():
    
    frame_number = 0
    center_time_frame = 0
    center_time_sec = 0
    for n in range(len(body_x_array)):
        if n == fps*600:
            break
        
        #defining the x and y coords of the body as an integer at the given frame, taken from the array         
        x_body = int(body_x_array[frame_number])
        y_body = int(body_y_array[frame_number])
        
        if x_body >= x_min and x_body <= x_max and y_body >= y_min and y_body <= y_max:
            
            center_time_frame += 1
            center_time_sec += frame_duration
        frame_number +=1
    
    return float(f'{center_time_sec: .02f}')

def fn_percent_time_center(): 
    frame_number = 0
    center_time_frame = 0
    for n in range(len(body_x_array)):
        if n == fps*600:
            break
        
        #defining the x and y coords of the body as an integer at the given frame, taken from the array         
        x_body = int(body_x_array[frame_number])
        y_body = int(body_y_array[frame_number])
        
        if x_body >= x_min and x_body <= x_max and y_body >= y_min and y_body <= y_max:
            
            center_time_frame += 1
        frame_number +=1
    
    #frame number is +1 because frame number is 0 for the first iteration while the center_time_frame is 1, and then 1 and 2, etc. 
    percent_time_center = ((center_time_frame)/(frame_number+1))*100
    
    return float(f'{percent_time_center: .02f}')
    
    
    
    
    return(float(f'{percent_time_center: .02f}'))
    
def fn_time_immoble ():
    frame_number = 0
    time_immoble1 = 0
    time_immoble = 0
    
    for n in range(len(body_x_array)):
        if n == fps*600: ### the 600 is the amount of seconds in 10 min, will stop counting after ten minutes
        
            break
        if fn_avg_velocity(frame_number, 'body')<= 5.0: 
    
            #time_immoble1 is a placeholder variable that only increases in value when the mouse's average velocity over a span of 3 seconds is below 5 cm/sec.
            time_immoble1 += frame_duration
        
            #if the mouse is immoble for .5 sec, time_immoble begins to count up. 
            if time_immoble1 >= .5 :
                time_immoble += frame_duration
            
        elif fn_avg_velocity(frame_number, 'body')<= 10.0:
                    
            #if mouse velocity >5 and <=10 then time_immoble1 only resets to .25, meaning it only takes .25 seconds to be considered immoble again
            time_immoble1 = .25
        else:
            time_immoble1 = 0
            
        frame_number+=1
    return float(f'{time_immoble: .03f}')

def fn_percent_time_immoble ():
    return float(f'{(fn_time_immoble()*100)/(fps*600): .03f}')

def fn_time_towards_wall ():
     
    frame_number = 0
    wall_time_frame = 0
    wall_time_sec = 0
    for n in range(len(body_x_array)):
        if n == fps*600:
            break
        
        #defining the x and y coords of the nose as an integer at the given frame, taken from the array         
        x_nose = int(nose_x_array[frame_number])
        y_nose = int(nose_y_array[frame_number])
        
        if x_nose <= ext_x_min or x_nose >= ext_x_max or y_nose <= ext_y_min or y_nose >= ext_y_max:
            
            wall_time_frame += 1
            wall_time_sec += frame_duration
        frame_number +=1
    
    return float(f'{wall_time_sec: .02f}')

def fn_times_on_wall ():
    frame_number = 0
    time_since_on_wall = 0
    times_on_wall = 0
    len_mouse = 0
    max_len_mouse = 0
    
    for n in range(len(body_x_array)):
        if n == fps*600:
            break
        
        #defining the x and y coords of the body as an integer at the given frame, taken from the array         
        x_body = int(body_x_array[frame_number])
        y_body = int(body_y_array[frame_number])
        
        x_nose = int(nose_x_array[frame_number])
        y_nose = int(nose_y_array[frame_number])
        
        o_x_body = int(body_x_array[frame_number-1])
        o_y_body = int(body_y_array[frame_number-1])
        
        
        time_since_on_wall += frame_duration
        
        len_mouse = int(np.sqrt(((x_body-x_nose)**2)+((y_body-y_nose)**2)))
        
        if len_mouse > max_len_mouse:
            
            max_len_mouse = len_mouse
        #The length of the mouse at the time of on_wall being True is considered to decrease the amount of false positives when the mouse is simply parallel to the boundary and crosses it. 
        if (x_body <= ext_x_min or x_body >= ext_x_max or y_body <= ext_y_min or y_body >= ext_y_max) and (x_nose <= ext_x_min or x_nose >= ext_x_max or y_nose <= ext_y_min or y_nose >= ext_y_max) and (o_x_body >= ext_x_min and o_x_body <= ext_x_max and o_y_body >= ext_y_min and o_y_body <= ext_y_max) and time_since_on_wall >= .7 and len_mouse <= .80*max_len_mouse :
            
            times_on_wall += 1
            time_since_on_wall = 0
            
        frame_number +=1
        
    return float(f'{times_on_wall: .02f}')

def fn_time_scrunch ():
    
    len_mouse = 0
    max_len_mouse = 0
    frame_number = 0
    time_scruntch = 0
    
    for n in range(len(body_x_array)):
        
        x_body = int(body_x_array[frame_number])
        y_body = int(body_y_array[frame_number])
        
        x_nose = int(nose_x_array[frame_number])
        y_nose = int(nose_y_array[frame_number])
        
        len_mouse = int(np.sqrt(((x_body-x_nose)**2)+((y_body-y_nose)**2)))
        
        if len_mouse > max_len_mouse:
            
            max_len_mouse = len_mouse
    
   # for n in range(len(body_x_array)):
    #    
     #   x_body = int(body_x_array[frame_number])
      #  y_body = int(body_y_array[frame_number])
       # 
        #x_nose = int(nose_x_array[frame_number])
        #y_nose = int(nose_y_array[frame_number])
        
        #len_mouse = int(np.sqrt(((x_body-x_nose)**2)+((y_body-y_nose)**2)))
        
        
        #this is True if :  The len_mouse is less than or equal to 60% of max_len_mouse. AND Neither the body nor the nose are completely outside the defined boundaries (i.e., if one or both are within the boundaries).
        #therefore the mouse will not be considered scruntched if on the wall.
        if len_mouse <= (max_len_mouse*.60) and not ((x_body <= ext_x_min or x_body >= ext_x_max or y_body <= ext_y_min or y_body >= ext_y_max) and (x_nose <= ext_x_min or x_nose >= ext_x_max or y_nose <= ext_y_min or y_nose >= ext_y_max)) :
                
            time_scruntch += frame_duration
            
        frame_number +=1
            
    return float(f'{time_scruntch: .02f}')
        
        

#def fn_gen_excel ():
    
    #define the data frame
    
    #df1 = pd.


#%% Creating video

def fn_gen_video ():
    
    video_name = file_path + video + '.avi'
 
    #capturing the video
    cap = cv2.VideoCapture(video_name)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    #determining the video dimentions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(f'{video}_motion_tracking.avi',0,fourcc, fps, (width,height))

    #print('fps:', fps)
    
    #defining some variables
    frame_number = 0
    center_time_sec = 0
    center_time_frame = 0
    time_immoble1 = 0
    time_immoble = 0
    wall_time_frame = 0
    wall_time_sec = 0
    time_since_on_wall = 0
    times_on_wall = 0
    len_mouse = 0
    max_len_mouse = 0
    time_scruntch = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
    
        if not ret:
            break  # Exit the loop if the video has ended or cannot be read

        # Get the x, y positions for the current frame
        x_nose = int(nose_x_array[frame_number])
        y_nose = int(nose_y_array[frame_number])
        
        x_head = int(head_x_array[frame_number])
        y_head = int(head_y_array[frame_number])
    
        x_neck = int(neck_x_array[frame_number])
        y_neck = int(neck_y_array[frame_number])
    
        x_leftear = int(leftear_x_array[frame_number])
        y_leftear = int(leftear_y_array[frame_number])
    
        x_rightear = int(rightear_x_array[frame_number])
        y_rightear = int(rightear_y_array[frame_number])
    
        x_body = int(body_x_array[frame_number])
        y_body = int(body_y_array[frame_number])
    
        x_tailbase = int(tailbase_x_array[frame_number])
        y_tailbase = int(tailbase_y_array[frame_number])

        #these are the body positions one frame previous. Used to calculate times climbed on wall
        o_x_body = int(body_x_array[frame_number-1])
        o_y_body = int(body_y_array[frame_number-1])
        
        # Draw a circle (dot) on the current frame
        cv2.circle(frame, (x_nose, y_nose), radius=3, color=(0, 0, 255), thickness=-1)
        cv2.circle(frame, (x_head, y_head), radius=3, color=(0, 0, 255), thickness=-1)
        cv2.circle(frame, (x_neck, y_neck), radius=3, color=(0, 0, 255), thickness=-1)
        cv2.circle(frame, (x_leftear, y_leftear), radius=3, color=(0, 0, 255), thickness=-1)
        cv2.circle(frame, (x_rightear, y_rightear), radius=3, color=(0, 0, 255), thickness=-1)
        cv2.circle(frame, (x_body, y_body), radius=3, color=(0, 255, 255), thickness=-1)
        cv2.circle(frame, (x_tailbase, y_tailbase), radius=3, color=(0, 0, 255), thickness=-1)

        #draw a line to make internal boundaries
        #top line
        cv2.line(frame,(x_min,y_min),(x_max,y_min),color=(255,0,0), thickness=2)
        #bottom line
        cv2.line(frame,(x_min,y_max),(x_max,y_max),color=(255,0,0), thickness=2)
        #left line
        cv2.line(frame,(x_min,y_min),(x_min,y_max),color=(255,0,0), thickness=2)
        #right line
        cv2.line(frame,(x_max,y_min),(x_max,y_max),color=(255,0,0), thickness=2)
    
        #draw lines to make external boundaries
        #top line
        cv2.line(frame,(ext_x_min,ext_y_min),(ext_x_max,ext_y_min),color=(255,0,0), thickness=2)
        #bottom line
        cv2.line(frame,(ext_x_min,ext_y_max),(ext_x_max,ext_y_max),color=(255,0,0), thickness=2)
        #left line
        cv2.line(frame,(ext_x_min,ext_y_min),(ext_x_min,ext_y_max),color=(255,0,0), thickness=2)
        #right line
        cv2.line(frame,(ext_x_max,ext_y_min),(ext_x_max,ext_y_max),color=(255,0,0), thickness=2)
        
    
        #line to calculate the scale of the cage (diagonal)
        #cv2.line(frame,(1020,140),(300,845),color=(255,0,0), thickness=2)
        
        #displays the frame number
        cv2.putText(frame,'frame:'+str(frame_number), (0,75), cv2.FONT_HERSHEY_SIMPLEX, fontScale = .75, color=(100, 200, 0), thickness = 2)
        #determining the time scruntched 
        
        len_mouse = int(np.sqrt(((x_body-x_nose)**2)+((y_body-y_nose)**2)))
    
        if len_mouse > max_len_mouse:
        
            max_len_mouse = len_mouse
            
        if len_mouse <= (max_len_mouse*.60) and not ((x_body <= ext_x_min or x_body >= ext_x_max or y_body <= ext_y_min or y_body >= ext_y_max) and (x_nose <= ext_x_min or x_nose >= ext_x_max or y_nose <= ext_y_min or y_nose >= ext_y_max)) :
                
            time_scruntch += frame_duration
            
        cv2.putText(frame,f'time scruntched:{time_scruntch: .02f}', (0,225), cv2.FONT_HERSHEY_SIMPLEX, fontScale = .75, color=(100, 200, 0), thickness = 2)
        cv2.putText(frame,f'max_len_mouse:{max_len_mouse: .02f}', (0,250), cv2.FONT_HERSHEY_SIMPLEX, fontScale = .75, color=(100, 200, 0), thickness = 2)
        
        #on the condition that the mouse nose is near the wall,  time goes up 1/30 of a second.
        if x_nose <= ext_x_min or x_nose >= ext_x_max or y_nose <= ext_y_min or y_nose >= ext_y_max:
            
            wall_time_frame += 1
            wall_time_sec += frame_duration
        #display the time spent towards the wall 
        cv2.putText(frame,f'time towards wall:{wall_time_sec: .02f}', (0,200), cv2.FONT_HERSHEY_SIMPLEX, fontScale = .75, color=(100, 200, 0), thickness = 2)
        
        #on the condition that the mouse body is near the wall, the times climbed on the wall goes up
        time_since_on_wall += frame_duration
       
        if (x_body <= ext_x_min or x_body >= ext_x_max or y_body <= ext_y_min or y_body >= ext_y_max) and (x_nose <= ext_x_min or x_nose >= ext_x_max or y_nose <= ext_y_min or y_nose >= ext_y_max) and (o_x_body >= ext_x_min and o_x_body <= ext_x_max and o_y_body >= ext_y_min and o_y_body <= ext_y_max) and time_since_on_wall >= .7 and len_mouse <= .80*max_len_mouse :
           
            times_on_wall += 1
            time_since_on_wall = 0
            
        #display the numbe rof times on the wall 
        cv2.putText(frame,f'times on wall:{times_on_wall: .1f}', (0,150), cv2.FONT_HERSHEY_SIMPLEX, fontScale = .75, color=(100, 200, 0), thickness = 2)
        
        #On the condition that the mouse is in the center time goes up (1/30 of a second)
        if x_body >= x_min and x_body <= x_max and y_body >= y_min and y_body <= y_max:
        
            center_time_frame += 1
            center_time_sec += frame_duration
        
        #displaying the center counter. 
        cv2.putText(frame,'time in center:'+f'{center_time_sec: .03f}'+' sec', (0,100), cv2.FONT_HERSHEY_SIMPLEX, fontScale = .75, color=(100, 200, 0), thickness = 2) 
        cv2.putText(frame,'time in center:'+str(center_time_frame)+' frames', (0,125), cv2.FONT_HERSHEY_SIMPLEX, fontScale = .75, color=(100, 200, 0), thickness = 2)
   
        #displaying mouse velocity (of the body)
        
        #displays the average velocity of the mouse, color depends on the magnitude
        
        if fn_avg_velocity(frame_number, 'body')<= 5.0: 
            #cv2.circle(frame, (50,50), 3, (255,0,0), 3)
            cv2.putText(frame,'velocity (cm/s):'+str(fn_avg_velocity(frame_number, 'body')), (0,50), cv2.FONT_HERSHEY_SIMPLEX, fontScale = .75, color=(255, 0, 0), thickness = 2) 
            
            #time_immoble1 is a placeholder variable that only increases in value when the mouse's average velocity over a span of 3 seconds is below 5 cm/sec.
            time_immoble1 += frame_duration
            
            #if the mouse is immoble for .5 sec, time_immoble begins to count up. 
            if time_immoble1 >= .5 :
                    time_immoble += frame_duration
            
        elif fn_avg_velocity(frame_number, 'body')<= 10.0:
            cv2.putText(frame,'velocity (cm/s):'+str(fn_avg_velocity(frame_number, 'body')), (0,50), cv2.FONT_HERSHEY_SIMPLEX, fontScale = .75, color=(100, 200, 0), thickness = 2)
            #cv2.circle(frame, (50,50), 3, (100,200,0), 3)
    
            #if mouse velocity >5 and <=10 then time_immoble1 only resets to .25, meaning it only takes .25 seconds to be considered immoble again
            time_immoble1 = .25
        else:
            #cv2.circle(frame, (50,50), 3, (0,0,255), 3)
            cv2.putText(frame,'velocity (cm/s):'+str(fn_avg_velocity(frame_number, 'body')), (0,50), cv2.FONT_HERSHEY_SIMPLEX, fontScale = .75, color=(0, 0, 255), thickness = 2)
            time_immoble1 = 0
    
        #display the time spend immoble 
        cv2.putText(frame,f'time immoble: {time_immoble: .03f}sec', (0,25), cv2.FONT_HERSHEY_SIMPLEX, fontScale = .75, color=(100, 200, 0), thickness = 2)
        #write the frame (save to the new video file)
        out.write(frame)
    
        # Display the frame
        cv2.imshow('Mouse Nose Tracking', frame)

        # Wait for a short period, exit if 'q' is pressed
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    
        frame_number += 1
        
        if frame_number >= fps*600:
            break  # Exit if we run out of position data

        # Release the video capture object and close all OpenCV windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    #percent_time_center = (center_time_frame/(frame_number+1))*100
    #print('frames that were played',frame_number+1)
    #print(f'time in center {center_time_sec:.03} sec')
    #print('percent of time in center', percent_time_center)
    return()

#percent_time_center = (center_time_frame/(frame_number+1))*100


#print('video dimentions, width x height:', width,'x', height)

#%% This is a video generator that only displays the time in center. Simplified to reduce render time (does not really help)
def fn_gen_simple_video ():
    
    video_name = file_path + video + '.avi'
 
    #capturing the video
    cap = cv2.VideoCapture(video_name)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    #determining the video dimentions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(f'{video}_motion_tracking.avi',0,fourcc, fps, (width,height))

    #print('fps:', fps)
    
    #defining some variables
    frame_number = 0
    center_time_sec = 0
    center_time_frame = 0
    


    while cap.isOpened():
        ret, frame = cap.read()
    
        if not ret:
            break  # Exit the loop if the video has ended or cannot be read

        # Get the x, y positions for the current frame
       
        x_body = int(body_x_array[frame_number])
        y_body = int(body_y_array[frame_number])
    
   
       
        cv2.circle(frame, (x_body, y_body), radius=3, color=(0, 255, 255), thickness=-1)
        

        #draw a line to make the boundaries
        #top line
        cv2.line(frame,(x_min,y_min),(x_max,y_min),color=(255,0,0), thickness=2)
        #bottom line
        cv2.line(frame,(x_min,y_max),(x_max,y_max),color=(255,0,0), thickness=2)
        #left line
        cv2.line(frame,(x_min,y_min),(x_min,y_max),color=(255,0,0), thickness=2)
        #right line
        cv2.line(frame,(x_max,y_min),(x_max,y_max),color=(255,0,0), thickness=2)
    
        #line to calculate the scale of the cage (diagonal)
        #cv2.line(frame,(1020,140),(300,845),color=(255,0,0), thickness=2)
        
        #displays the frame number
        cv2.putText(frame,'frame:'+str(frame_number), (50,75), cv2.FONT_HERSHEY_SIMPLEX, fontScale = .75, color=(100, 200, 0), thickness = 2)
        cv2.putText(frame,'time (sec):'+str(frame_number/30), (50,50), cv2.FONT_HERSHEY_SIMPLEX, fontScale = .75, color=(100, 200, 0), thickness = 2)

        #On the condition that the mouse is in the center time goes up (1/30 of a second)
        if x_body >= x_min and x_body <= x_max and y_body >= y_min and y_body <= y_max:
        
            center_time_frame += 1
            center_time_sec += frame_duration
        
        #displaying the center counter. 
        cv2.putText(frame,'time in center:'+f'{center_time_sec: .03f}'+' sec', (50,100), cv2.FONT_HERSHEY_SIMPLEX, fontScale = .75, color=(100, 200, 0), thickness = 2) 
        cv2.putText(frame,'time in center:'+str(center_time_frame)+' frames', (50,125), cv2.FONT_HERSHEY_SIMPLEX, fontScale = .75, color=(100, 200, 0), thickness = 2)
        
        #save the generated frame to the video
        #out.write(frame)
        
        # Display the frame
        cv2.imshow('Mouse Nose Tracking', frame)

        # Wait for a short period, exit if 'q' is pressed
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    
        frame_number += 1
        
        if frame_number >= fps*600:
            break  # Exit if we run out of position data

        # Release the video capture object and close all OpenCV windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()
#%% outputs

#just some sample velocities


#print(f'the percent of time spent in the center is {fn_percent_time_center():.02f}%')
#print (f'the time in the center is {fn_time_center():.02f} seconds')
#print(f'the time spent immoble is: {fn_time_immoble():.02f}')
#print
#fn_directory_path('C:/Users/kacper/Desktop/shaorong_sample_data/')


#fn_avg_velocity(0, 'body')
#fn_inst_velocity(0, 'body')
#print('the instantaneous velocity at frame 0 is', fn_inst_velocity(0, 'body'))
#inst_velocity(2, 'body')
#print('the avg velocity at frame 4243 is:', fn_avg_velocity(4212, 'body'), 'cm/s')

#troubleshooting functions
#print('the distance travelled from 0 to 1 is: ', fn_distance_travelled(0,1))
#print('the ins velocity at frame 0 is:', fn_inst_velocity(0, 'body'))


#IMPORTANT OUTPUTS FOR FINAL DATA COLLECTION BELOW




fn_gen_video()
#fn_gen_simple_video()
print(fn_time_scrunch())
print(fn_times_on_wall())
print(fn_time_towards_wall())
print(video)
print(fn_time_center())
print(fn_percent_time_center())
print(fn_time_immoble())
print(fn_distance_travelled(0,(fps*600)))
print(fn_overall_velocity('body'))
print(fn_percent_time_immoble())

print(x_min)
print(x_max)
print(y_min)
print(y_max)


print('working as intended 😎')



