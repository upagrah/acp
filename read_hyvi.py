import numpy as np
from osgeo import gdal,ogr,osr
import os
from sklearn.decomposition import PCA, IncrementalPCA
from tabulate import tabulate
import math




#----- ASSUMPS ----------------------
#1) ALL BANDS HAVE SAME Geometric Representation ************** HAVE TO WORK ON RESHAPING TRANSFORMED ARRAY




def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()





#--------------------- CONFIG ---------------------------------------

EXCLUDED_BANDS=[x for x in range(1,8)]+[x for x in range(58,77)]+[x for x in range(225,243)]
#print(EXCLUDED_BANDS)

BANDS_RANGE=[x for x in range(1,243)]

VALID_BANDS=[]

for x in BANDS_RANGE:
    if(x not in EXCLUDED_BANDS):
        VALID_BANDS.append(x)



BOUNDARY_SHP='/home/cloud/rpros/data/hyperion/boundary.shp'

datadirloc='/home/cloud/rpros/data/hyperion/'

outdatadirloc='/home/cloud/rpros/data/hyviouts/'


print('\n\n\n')

print(' ---------- INCREMENTAL PRINCIPLE COMPONENT ANALYSIS ROUTINE FOR HYPERSPECTRAL IMAGERY -----------')

print('\n\n\n')



band_name_list=[]
#------------------------
def pixel2coord(x, y,transform):
    xp = transform[1]* x + transform[2]* y + transform[0]
    yp = transform[4] * x + transform[5] * y + transform[3]
    return(xp, yp)
#------------------------
def seperator():
    print('\n---------------------------------------------------------\n')
#--------------------------------------------------------------------

#------------------------------------PREP BANDS----------------------
for x in VALID_BANDS:
    str_att=''
    if(x<10):
        str_att='00'+str(x)
    elif(x>=10 and x<100):
        str_att='0'+str(x)
    else:
        str_att=str(x)
    file_name_temp=datadirloc+'EO1H1440482003116110PZ_B'+str_att+'_L1T.TIF'
    band_name_list.append(file_name_temp)
#print(band_name_list)



#---------------------------- LOAD THE CONFIG AND INITIAL INFO -------------------

work_img=gdal.Open(band_name_list[len(band_name_list)-1])
#work_img_1=gdal.Open(band_name_list[78])
#working on band1 of black and whit image
#work_img_array=np.array(work_img.GetRasterBand(1).ReadAsArray())
cols = work_img.RasterXSize
rows = work_img.RasterYSize
transform = work_img.GetGeoTransform()
# cols1 = work_img_1.RasterXSize
# rows1 = work_img_1.RasterYSize
print('-- THE CONFIG IS AS FOLLOWS --')

print(tabulate([['DATA LOCATION',datadirloc],['OUTPUT LOCATION',outdatadirloc]],tablefmt='psql'))
#--------------------------------------------------------------------
seperator()
print('-- IMG DETAILS ARE AS FOLLOWS --')

print(tabulate([
    ['coloumns',cols],
    ['rows',rows],
    ['left_top_corx',transform[0]],
    ['horizontal scale',transform[1]],
    ['horizontal skew',transform[2]],
    ['left_top_cory',transform[3]],
    ['vertical scale',transform[5]],
    ['vertical skew',transform[4]],
    
    
    ],tablefmt='psql'))


seperator()

#print(str(transform)+' \n : GEOTRANSFORM \n xleft_top0,scalex1,skewx2,yleft_top3,skewy4,scaley5')#(xleft_top0,scalex1,skewx2,yleft_top3,skewy4,scaley5)

        
#---- READ THE IMAGE IN CHUNKS OF COL_WIN AND ROW_WIN AND PERFORM 0 CHECK AND INC_PCA
n_components=(len(BANDS_RANGE)-len(EXCLUDED_BANDS))
seperator()
print(str(n_components) + ' : ARE THE TOTAL NUMBER OF COMPONENTS')


COL_WIN=500
ROW_WIN=500
xoff=0
yoff=0
xend=0
yend=0
grid_count=0
ipca = IncrementalPCA(n_components=n_components)
print('-- Computing the Information content across components --')
printProgressBar(0, math.ceil(cols/(COL_WIN))*math.ceil(rows/ROW_WIN), prefix = 'Processing the Imagery', suffix = 'Processing Complete', length = 50)
#-------------
while(xend !=1 or yend !=1):
    #print('XOFF AND YOFF'+str(xoff)+':'+str(yoff))
    #horiz_grid_move
    xwin=COL_WIN
    ywin=ROW_WIN

    if(cols-xoff<=COL_WIN):
        xwin=cols-xoff
        xend=1
    else:
        xend=0
    if(rows-yoff<=ROW_WIN):
        ywin=rows-yoff
        yend=1
    else:
        yend=0
    
    #print('EXECUTE')
    #if(grid_count==1):#toberemoved
    data_ipca=np.ndarray(shape=(((xwin)*(ywin)),len(VALID_BANDS)))
    datarasters=[]
    
    count=0
    for imgdata in band_name_list:
        
        work_img=gdal.Open(imgdata)
        
        dataraster = work_img.GetRasterBand(1).ReadAsArray(xoff, yoff, xwin, ywin).astype(np.float)
        
        datarasters.append((dataraster))
        
    datarasters=np.array(datarasters)
    
    #print(datarasters[:,300,180])
    
    res_data=[]
    for y in range(0,ywin):
        for x in range(0,xwin):
            #for b in range(0,n_components):
            res_data.append(datarasters[:,y,x])
    res_data=np.array(res_data)
    #print(res_data[180+300*xwin])
    res_nz_data=res_data[~(res_data==0).all(1)]
    #print(res_nz_data)
    if(len(res_nz_data)>=1):
        partial_x_transform = ipca.partial_fit(res_nz_data)

    # if(len(nz_data_ipca>2)):
    #     print( nz_data_ipca[2])
    if(xend==1 and yend!=1):
        xoff=0
        yoff=yoff+ROW_WIN
    elif(xend!=1):
        xoff=xoff+COL_WIN
    #print('XEND AND YEND'+str(xend)+':'+str(yend))
    grid_count=grid_count+1
    #print (grid_count)
    printProgressBar(grid_count,math.ceil(cols/(COL_WIN))*math.ceil(rows/ROW_WIN) , prefix = 'Processing the Imagery', suffix = 'Processing Complete', length = 50)




# data_ipca = np.ndarray(shape=(5000, 200))
# data_ipca[0]=[x for x in range(1,201)]
# data_ipca[1]=[x for x in range(1,201)]
#----------------------------
#
#print(data_ipca)

# partial_x_data = ipca.transform(data_ipca)
# print(partial_x_data)
# seperator()
# print(str(partial_x_transform.components_) +'Are the Components')
# seperator()
# print(str(partial_x_transform.explained_variance_) +'Are the Variances explained')
# seperator()
# print(str(partial_x_transform.explained_variance_ratio_) +'Are the Variance Ratios explained')


#----------- REV WRITE--------------------------------


COL_WIN=500
ROW_WIN=500
xoff=0
yoff=0
xend=0
yend=0
grid_count=0
#ipca = IncrementalPCA(n_components=n_components)
#-------------

output_raster = gdal.GetDriverByName('GTiff').Create(outdatadirloc+'IPCA_fin_1.tif',cols, rows,1  ,gdal.GDT_Float32)  # Open the file
output_raster.SetGeoTransform(transform)  # Specify its coordinates

output_raster.SetProjection( work_img.GetProjection())   # Exports the coordinate system 
                                                   # to the file
# Writes my array to the raster
#output_raster.GetRasterBand(1).WriteArray(array)
print('-- COMPUTING AND TRANSFORMING THE ORIGINAL IMAGERY --')
printProgressBar(0, math.ceil(cols/(COL_WIN))*math.ceil(rows/ROW_WIN), prefix = 'Computing  Components', suffix = 'Computing  Components Complete', length = 50)
#-------------
grid_count=0
while(xend !=1 or yend !=1):
    #print('XOFF AND YOFF'+str(xoff)+':'+str(yoff))
    #horiz_grid_move
    xwin=COL_WIN
    ywin=ROW_WIN

    if(cols-xoff<=COL_WIN):
        xwin=cols-xoff
        xend=1
    else:
        xend=0
    if(rows-yoff<=ROW_WIN):
        ywin=rows-yoff
        yend=1
    else:
        yend=0
    
    #print('EXECUTE')
    data_ipca=np.ndarray(shape=(((xwin)*(ywin)),len(VALID_BANDS)))
    datarasters=[]
    #print('Processing Grids')
    for imgdata in band_name_list:
        ##print()
        work_img=gdal.Open(imgdata)
        
        dataraster = work_img.GetRasterBand(1).ReadAsArray(xoff, yoff, xwin, ywin).astype(np.float)
        datarasters.append((dataraster))
    datarasters=np.array(datarasters)
    datarasters_n=[]
    res_data=[]
    for y in range(0,ywin):
        for x in range(0,xwin):
            #for b in range(0,n_components):
            res_data.append(datarasters[:,y,x])
    res_data=np.array(res_data)
    x_transform = ipca.transform(res_data)


    for y in range(0,ywin):
        data_buffer_x=[]
        for x in range(0,xwin):
            data_buffer_x.append(x_transform[y*xwin+x])
        datarasters_n.append(data_buffer_x)
    datarasters_n=np.array(datarasters_n)
    #print(datarasters_n.shape)
    # import scipy.misc
    # scipy.misc.imsave(outdatadirloc+'outfile_dr.jpg', datarasters[1,:,:])
    # scipy.misc.imsave(outdatadirloc+'outfile_drn.jpg', datarasters_n[:,:,1])

    #print (datarasters.shape)
    #data_ipca=datarasters.reshape(ywin*xwin,n_components)
    #nz_data_ipca= data_ipca[~(data_ipca==0).all(1)]
    # x_transform = ipca.transform(datarasters)
    # print(x_transform.shape)
    # x_transform_reshape=x_transform.reshape((198,ywin,xwin))
    # #print(x_transform[1].shape)
    output_raster.GetRasterBand(1).WriteArray(datarasters_n[:,:,0],xoff=xoff,yoff=yoff)
    # if(len(nz_data_ipca>2)):
    #     print( nz_data_ipca[2])
    if(xend==1 and yend!=1):
        xoff=0
        yoff=yoff+ROW_WIN
    elif(xend!=1):
        xoff=xoff+COL_WIN
    grid_count=grid_count+1
    printProgressBar(grid_count,math.ceil(cols/(COL_WIN))*math.ceil(rows/ROW_WIN) , prefix = 'Computing  Components', suffix = 'Computing  Components Complete', length = 50)
output_raster.FlushCache()


np.savetxt(outdatadirloc+"components.txt",partial_x_transform.components_)


# text_file = open(outdatadirloc+"components.txt", "w")
# text_file.write(str(partial_x_transform.components_))
# text_file.close()


# text_file = open(outdatadirloc+"variance.txt", "w")
# text_file.write(str(partial_x_transform.explained_variance_))
# text_file.close()


# text_file = open(outdatadirloc+"percentage_variance.txt", "w")
# text_file.write(str(partial_x_transform.explained_variance_ratio_))
# text_file.close()