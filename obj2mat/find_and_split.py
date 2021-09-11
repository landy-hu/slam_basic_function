import os
import sys
import subprocess
from math import ceil
import math

task_dir = "tasks-all"
task_prefix = 'task'
task_size = 40
obj_dir = '/home/mpl/lily/data/trash-bin_mitsuba/trash-bin'

#for test only
test_dir = '/data/vision/billf/object-properties/shapenet/obj/val'

working_dir = obj_dir     # for work
#working_dir = test_dir   # for test

overwrite = True

#####################################################
##    find obj files

input_ext = 'obj'
filt_ext = 'mat'

print ("Looking for ."+input_ext+' files')
find_result =  subprocess.check_output('find '+working_dir+' -type f -name "*.'+input_ext+'"', shell=True)
# print(find_result)
objs_path = find_result.decode().split('\n')
objs_path = sorted(objs_path)
print(len(objs_path))
objs_path = filter(lambda x: x!='', objs_path)
print (input_ext + ' files found:', len(list(objs_path)))


# filt
if not overwrite:
    print ("Looking for ."+filt_ext+' files')
    filt_result = subprocess.check_output('find '+working_dir+' -type f -name "*.'+filt_ext+'"', shell=True)

    filt_list = filt_result.split('\n')
    filt_list = filter(lambda x:x!='', filt_list)

    print (filt_ext + ' files found:', len(filt_list))

    objs_path = filter(lambda x: (x[:-len(input_ext)]+filt_ext) not in filt_list, objs_path)
    print(objs_path)
######################################################3
# split
if os.path.isdir(task_dir):
    print ("task directory already exists! Are you sure you want to override existing task partition?")
    print ("If yes, please run this in command:")
    print ('    rm -rf '+task_dir)
    sys.exit(1)

os.system('mkdir -p '+task_dir)

print('num',int(ceil(len(list(objs_path))/float(task_size))))
for file_ind in range(int(ceil(len(list(objs_path))/float(task_size)))):
    task_file = open(task_dir+'/'+task_prefix+str(file_ind), 'w')
    print('hello:',task_file)
    ind_start = file_ind * task_size
    ind_end = min((file_ind+1) * task_size, len(list(objs_path)))
    for ind in range(ind_start, ind_end):
        task_file.write(objs_path[ind]+'\n')
    task_file.close()

print ('total number of files:', len(list(objs_path)))
print ('task_size:', task_size)
print ('total number fo tasks:', int(ceil(len(list(objs_path)))/float(task_size)))
print ('size of the last task:', len(list(objs_path)) - (int(ceil(len(list(objs_path))/float(task_size)))-1)*task_size)




