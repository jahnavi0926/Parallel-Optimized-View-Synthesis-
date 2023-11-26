import numpy as np
from PIL import Image,ImageOps, ImageFile
from scipy.stats import entropy
from scipy.optimize import minimize
import sys
import skimage.measure
from paraview.simple import *
from mpi4py import MPI
import time
import random
import logging
import threading
import queue
import time
import os
from csv import writer

ImageFile.LOAD_TRUNCATED_IMAGES = True


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

num_threads = 2


def generate_image(viewpoint,img_ind):

	paraview.simple._DisableFirstRenderCameraReset()

	# create a new 'Legacy VTK Reader'
	example_large_it9vtk = LegacyVTKReader(registrationName='example_large_it=9.vtk', FileNames=['datasets/example_large_it=9.vtk'])

	# get active view
	renderView1 = GetActiveViewOrCreate('RenderView')

	# show data in view
	example_large_it9vtkDisplay = Show(example_large_it9vtk, renderView1, 'UnstructuredGridRepresentation')

	# get color transfer function/color map for 'Scalars10'
	scalars10LUT = GetColorTransferFunction('Scalars10')

	# get opacity transfer function/opacity map for 'Scalars10'
	scalars10PWF = GetOpacityTransferFunction('Scalars10')

	# trace defaults for the display properties.
	example_large_it9vtkDisplay.Representation = 'Surface'
	example_large_it9vtkDisplay.ColorArrayName = ['POINTS', 'Scalars10']
	example_large_it9vtkDisplay.LookupTable = scalars10LUT
	example_large_it9vtkDisplay.SelectTCoordArray = 'None'
	example_large_it9vtkDisplay.SelectNormalArray = 'None'
	example_large_it9vtkDisplay.SelectTangentArray = 'None'
	example_large_it9vtkDisplay.OSPRayScaleArray = 'Scalars10'
	example_large_it9vtkDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
	example_large_it9vtkDisplay.SelectOrientationVectors = 'None'
	example_large_it9vtkDisplay.ScaleFactor = 9.9
	example_large_it9vtkDisplay.SelectScaleArray = 'Scalars10'
	example_large_it9vtkDisplay.GlyphType = 'Arrow'
	example_large_it9vtkDisplay.GlyphTableIndexArray = 'Scalars10'
	example_large_it9vtkDisplay.GaussianRadius = 0.495
	example_large_it9vtkDisplay.SetScaleArray = ['POINTS', 'Scalars10']
	example_large_it9vtkDisplay.ScaleTransferFunction = 'PiecewiseFunction'
	example_large_it9vtkDisplay.OpacityArray = ['POINTS', 'Scalars10']
	example_large_it9vtkDisplay.OpacityTransferFunction = 'PiecewiseFunction'
	example_large_it9vtkDisplay.DataAxesGrid = 'GridAxesRepresentation'
	example_large_it9vtkDisplay.PolarAxes = 'PolarAxesRepresentation'
	example_large_it9vtkDisplay.ScalarOpacityFunction = scalars10PWF
	example_large_it9vtkDisplay.ScalarOpacityUnitDistance = 77.2841842661301
	example_large_it9vtkDisplay.OpacityArrayName = ['POINTS', 'Scalars10']
	example_large_it9vtkDisplay.SelectInputVectors = [None, '']
	example_large_it9vtkDisplay.WriteLog = ''

	# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
	example_large_it9vtkDisplay.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 799999.0, 1.0, 0.5, 0.0]

	# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
	example_large_it9vtkDisplay.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 799999.0, 1.0, 0.5, 0.0]

	# reset view to fit data
	renderView1.ResetCamera(False)

	# get the material library
	materialLibrary1 = GetMaterialLibrary()

	# show color bar/color legend
	example_large_it9vtkDisplay.SetScalarBarVisibility(renderView1, True)

	# update the view to ensure updated data information
	renderView1.Update()

	# get 2D transfer function for 'Scalars10'
	scalars10TF2D = GetTransferFunction2D('Scalars10')

	# get layout
	layout1 = GetLayout()

	# layout/tab size in pixels
	layout1.SetSize(1538, 789)

	# current camera placement for renderView1
	renderView1.CameraPosition = [49.5, 49.5, 350.0597994267811]
	renderView1.CameraFocalPoint = [49.5, 49.5, 39.5]
	renderView1.CameraParallelScale = 80.37879073486985
	
	camera=GetActiveCamera()
	renderView1.ResetCamera()
	camera.Elevation(viewpoint[0]) 
	camera.Azimuth(viewpoint[1])
	renderView1.Update()
	# save screenshot
	SaveScreenshot(f'InitImages/image_{img_ind}.png', renderView1, ImageResolution=[1538, 789])
	SaveScreenshot(f'OptiImages/image_{img_ind}.png', renderView1, ImageResolution=[1538, 789])




#objective function
def objfunc(viewpoint,img_ind):
	#get image from viewpoint
	generate_image(viewpoint,img_ind)
	image_path = 'Optimages/image_' + str(img_ind) + '.png'


	image = Image.open(image_path)
	image = ImageOps.grayscale(image)
	image_array = np.array(image)

	overall_entropy = skimage.measure.shannon_entropy(image_array)
	return -1*(overall_entropy) # we minimise this


#optimiser
def optim(x0,img_ind):
	bounds = [(x0[0] -20,  x0[0] + 20), (x0[1] -20 , x0[1] + 20)]
	result = minimize(objfunc, x0 = x0, bounds = bounds, method= 'Nelder-Mead', options={ 'fatol' : 0.01, 'maxiter': 20}, args = {img_ind})
	return result.x

#main

if rank != 0:
	#take initial viewpoints
	[view,img_ind] = comm.recv()
	print('Process with rank=' + str(rank) + ' initialised.',flush = True)

	while(1):	

		optview = optim(view,img_ind)
		generate_image(view, img_ind)
		print("Image " + str(img_ind) + " optimised. Viewpoint: ", view, "->", optview,". Entropy: ",round(-1*objfunc(view,0),3),"->",round(-1*objfunc(optview,0),3),flush = True)
		#ask for new viewpoint
		comm.send(rank, dest=0)
		
		#get new viewpoint
		[view,img_ind] = comm.recv()
		if (view == 1):
			os._exit(0)
		print(str(rank) + ' got new viewpoint, for image ' + str(img_ind) + '.',flush = True)



elif rank == 0:

	start = time.time()

	#get data from csv
	img_ind = 1
	my_data = np.loadtxt(r'data.csv', delimiter=',')
	n = len(my_data)
	my_data = my_data.tolist()
	my_data = my_data[:int(sys.argv[1])]
	print("Viewpoints: ",my_data,flush = True)

	#initialise all processes
	for rrank in range (1,size):
		comm.send([my_data[0],img_ind], dest= rrank)
		img_ind = img_ind + 1
		my_data = my_data[1:] #remove first element

	#while list isnt empty, keep receieving and sending
	while (len(my_data) > 0):
		rrec = comm.recv()
		print('Receieved viewpoint request from rank ' + str(rrec) + '.',flush = True)
		comm.send([my_data[0],img_ind],dest= rrec)
		img_ind = img_ind + 1
		my_data = my_data[1:]
		
	n = size
	while (n > 1):
		comm.recv()
		n = n - 1
	print('All images saved.')
	n = 1
	while (n < size):
		comm.send([1,img_ind], dest = n)
		n = n + 1
	MPI.Finalize()
	end = time.time()	
	print("Time taken :", end-start)
	
	#writing optimisarion time results to csv
	with open('results.csv','a') as file:
		wobj = writer(file)
		wobj.writerow([size,sys.argv[1],end - start,0])
		file.close()

