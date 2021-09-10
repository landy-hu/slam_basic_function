import scipy.io as scio
from skimage import measure
import numpy as np
import h5py
import struct


def meshwrite(filename, verts, faces, norms, colors):
    # Write header
    ply_file = open(filename, 'w')
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n" % (verts.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("property float nx\n")
    ply_file.write("property float ny\n")
    ply_file.write("property float nz\n")
    ply_file.write("property uchar red\n")
    ply_file.write("property uchar green\n")
    ply_file.write("property uchar blue\n")
    ply_file.write("element face %d\n" % (faces.shape[0]))
    ply_file.write("property list uchar int vertex_index\n")
    ply_file.write("end_header\n")

    # Write vertex list
    for i in range(verts.shape[0]):
        ply_file.write("%f %f %f %f %f %f %d %d %d\n" % (
        verts[i, 0], verts[i, 1], verts[i, 2], norms[i, 0], norms[i, 1], norms[i, 2], colors[i, 0], colors[i, 1],
        colors[i, 2]))

    # Write face list
    for i in range(faces.shape[0]):
        ply_file.write("3 %d %d %d\n" % (faces[i, 0], faces[i, 1], faces[i, 2]))

    ply_file.close()


object_txt = '/home/lan/Desktop/iccv2019/data/scene.txt'
f = open(object_txt)
line = f.readline()
R_new = np.eye(3)
R_new[[0,1],:] =R_new[[1,0],:]
ves=[]
nos=[]
col=[]
fac=[]
index_vert=[]
index_vert.append(0)
index_face=[]
index_face.append(0)
count_vert =[]
count_face=[]
idx = 0

thresh = 0.025
step=thresh
while line:
    print(line)
    idx +=1
    line = line[:-1]
    print(line)

    pt = scio.loadmat(line)
    likelyhood_value = pt['dist']

    likelyhood_value[likelyhood_value>0.75*thresh]=1
    likelyhood_value[likelyhood_value<=0.75*thresh]=0

    likelyhood_value = likelyhood_value-1
    likelyhood_value[likelyhood_value==-1]=1

    if idx ==8:
        likelyhood_value = likelyhood_value-0.1
    else:
        likelyhood_value = likelyhood_value-0.4

    x_1 = likelyhood_value[1:-1, 1: -1, 1: -1] + likelyhood_value[1: -1, 1: -1, 0:-2] + likelyhood_value[1: -1, 0:-2, 1: -1] + likelyhood_value[0:-2, 1: -1, 1: -1]
    x_2 = likelyhood_value[0:-2 , 0:-2, 0:-2] + likelyhood_value[0:-2 , 0:-2, 1:-1] + likelyhood_value[0:-2 , 1:-1, 0:-2] + likelyhood_value[1:-1 , 0:-2, 0:-2]
    likelyhood_value[1: -1 , 1: -1, 1: -1]= (x_1 + x_2) / 8
    x_1 = likelyhood_value[1:-1, 1: -1, 1: -1] + likelyhood_value[1: -1, 1: -1, 0:-2] + likelyhood_value[1: -1, 0:-2, 1: -1] + likelyhood_value[0:-2, 1: -1, 1: -1]
    x_2 = likelyhood_value[0:-2 , 0:-2, 0:-2] + likelyhood_value[0:-2 , 0:-2, 1:-1] + likelyhood_value[0:-2 , 1:-1, 0:-2] + likelyhood_value[1:-1 , 0:-2, 0:-2]
    likelyhood_value[1: -1 , 1: -1, 1: -1]= (x_1 + x_2) / 8


    verts, faces, normals,values = measure.marching_cubes_lewiner(likelyhood_value, level=0)

    verts = verts*step + step/2
    if idx<=2:
        verts = verts + np.array([-6.2549,-2.7934,-4.5600]).reshape(1,-1)
    else:
        verts = verts + np.array([-4.1113,   -0.6338,   -1.5328]).reshape(1,-1)

    # verts = (R_new.dot(verts.transpose())).transpose()
    # normals = (R_new.dot(normals.transpose())).transpose()
    if idx%2==0:
        color = np.array([77,141,180]).reshape(1,-1)
    else:
        color = np.array([191,39,35]).reshape(1,-1)

    if idx>10:
        color = np.array([244,51,8]).reshape(1,-1)
    colors = np.tile(color,[verts.shape[0],1])
    name = '/home/lan/Desktop/iccv2019/data/curtain/cma_mesh/scene_' +str(idx)+'.ply'
    meshwrite(name, verts, faces, normals, colors)
    line = f.readline()

f.close()

# ve = np.zeros((index_vert[-1],3))
# no = np.zeros((index_vert[-1],3))
# co = np.zeros((index_vert[-1],3))
# fa = np.zeros((index_face[-1],3))
#
# for i in range(len(count_face)):
#     ve[index_vert[i]:index_vert[i+1],:] = ves[i]
#     no[index_vert[i]:index_vert[i+1],:] = nos[i]
#     co[index_vert[i]:index_vert[i+1],:] = col[i]
#     fa[index_face[i]:index_face[i+1],:] = fac[i]+index_vert[i]
# name = '/home/lan/Desktop/iccv2019/data/curtain/cma_mesh/cma_mesh_before.ply'
# meshwrite(name, ve, fa, no, co)
