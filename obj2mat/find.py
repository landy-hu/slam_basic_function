import os
import sys
import subprocess
from math import ceil

task_dir = "tasks-all_view"
task_prefix = 'task'
task_size = 40
obj_dir = '/home/lan/Downloads/shapenet/data/04379243/'

#for test only
test_dir = '/data/vision/billf/object-properties/shapenet/obj/val'

working_dir = obj_dir     # for work
#working_dir = test_dir   # for test

overwrite = False

#####################################################
##    find obj files

input_ext = 'obj'
filt_ext = 'mat'

print "Looking for ."+input_ext+' files'
find_result =  subprocess.check_output('find '+working_dir+' -type f -name "*.'+input_ext+'"', shell=True)

objs_path = find_result.split('\n')
objs_path = sorted(objs_path)
objs_path = filter(lambda x: x!='', objs_path)
print input_ext + ' files found:', len(objs_path)

# filt
if not overwrite:
    print ("Looking for ."+filt_ext+' files')
    filt_result = subprocess.check_output('find '+working_dir+' -type f -name "*.'+filt_ext+'"', shell=True)

    filt_list = filt_result.split('\n')
    filt_list = filter(lambda x:x!='', filt_list)

    print filt_ext + ' files found:', len(filt_list)

    objs_path = filter(lambda x: (x[:-len(input_ext)]+filt_ext) not in filt_list, objs_path)
######################################################3
# split
if os.path.isdir(task_dir):
    print "task directory already exists! Are you sure you want to override existing task partition?"
    print "If yes, please run this in command:"
    print '    rm -rf '+task_dir
    sys.exit(1)

os.system('mkdir -p '+task_dir)

for file_ind in xrange(int(ceil(len(objs_path)/float(task_size)))):
    task_file = open(task_dir+'/'+task_prefix+str(file_ind), 'w')
    ind_start = file_ind * task_size
    ind_end = min((file_ind+1) * task_size, len(objs_path))
    for ind in xrange(ind_start, ind_end):
        task_file.write(objs_path[ind]+'\n')
    task_file.close()

print 'total number of files:', len(objs_path)
print 'task_size:', task_size
print 'total number fo tasks:', int(ceil(len(objs_path)/float(task_size)))
print 'size of the last task:', len(objs_path) - (int(ceil(len(objs_path)/float(task_size)))-1)*task_size




