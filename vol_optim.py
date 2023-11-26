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

	# create a new 'XML Image Data Reader'
	more_reduced_pressure_datavti = XMLImageDataReader(registrationName='more_reduced_pressure_data.vti', FileName=['datasets/more_reduced_pressure_data.vti'])
	more_reduced_pressure_datavti.PointArrayStatus = ['ImageScalars']

	# Properties modified on more_reduced_pressure_datavti
	more_reduced_pressure_datavti.TimeArray = 'None'

	# get active view
	renderView1 = GetActiveViewOrCreate('RenderView')

	# show data in view
	more_reduced_pressure_datavtiDisplay = Show(more_reduced_pressure_datavti, renderView1, 'UniformGridRepresentation')

	# trace defaults for the display properties.
	more_reduced_pressure_datavtiDisplay.Representation = 'Outline'
	more_reduced_pressure_datavtiDisplay.ColorArrayName = ['POINTS', '']
	more_reduced_pressure_datavtiDisplay.SelectTCoordArray = 'None'
	more_reduced_pressure_datavtiDisplay.SelectNormalArray = 'None'
	more_reduced_pressure_datavtiDisplay.SelectTangentArray = 'None'
	more_reduced_pressure_datavtiDisplay.OSPRayScaleArray = 'ImageScalars'
	more_reduced_pressure_datavtiDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
	more_reduced_pressure_datavtiDisplay.SelectOrientationVectors = 'None'
	more_reduced_pressure_datavtiDisplay.ScaleFactor = 9.142180443925234
	more_reduced_pressure_datavtiDisplay.SelectScaleArray = 'ImageScalars'
	more_reduced_pressure_datavtiDisplay.GlyphType = 'Arrow'
	more_reduced_pressure_datavtiDisplay.GlyphTableIndexArray = 'ImageScalars'
	more_reduced_pressure_datavtiDisplay.GaussianRadius = 0.4571090221962617
	more_reduced_pressure_datavtiDisplay.SetScaleArray = ['POINTS', 'ImageScalars']
	more_reduced_pressure_datavtiDisplay.ScaleTransferFunction = 'PiecewiseFunction'
	more_reduced_pressure_datavtiDisplay.OpacityArray = ['POINTS', 'ImageScalars']
	more_reduced_pressure_datavtiDisplay.OpacityTransferFunction = 'PiecewiseFunction'
	more_reduced_pressure_datavtiDisplay.DataAxesGrid = 'GridAxesRepresentation'
	more_reduced_pressure_datavtiDisplay.PolarAxes = 'PolarAxesRepresentation'
	more_reduced_pressure_datavtiDisplay.ScalarOpacityUnitDistance = 0.8184745782579901
	more_reduced_pressure_datavtiDisplay.OpacityArrayName = ['POINTS', 'ImageScalars']
	more_reduced_pressure_datavtiDisplay.ColorArray2Name = ['POINTS', 'ImageScalars']
	more_reduced_pressure_datavtiDisplay.IsosurfaceValues = [-1580.5731811523438]
	more_reduced_pressure_datavtiDisplay.SliceFunction = 'Plane'
	more_reduced_pressure_datavtiDisplay.Slice = 25
	more_reduced_pressure_datavtiDisplay.SelectInputVectors = [None, '']
	more_reduced_pressure_datavtiDisplay.WriteLog = ''

	# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
	more_reduced_pressure_datavtiDisplay.ScaleTransferFunction.Points = [-4930.23046875, 0.0, 0.5, 0.0, 1769.0841064453125, 1.0, 0.5, 0.0]

	# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
	more_reduced_pressure_datavtiDisplay.OpacityTransferFunction.Points = [-4930.23046875, 0.0, 0.5, 0.0, 1769.0841064453125, 1.0, 0.5, 0.0]

	# init the 'Plane' selected for 'SliceFunction'
	more_reduced_pressure_datavtiDisplay.SliceFunction.Origin = [112.19961176635513, 120.51068489719626, 8.221492788590604]

	# reset view to fit data
	renderView1.ResetCamera(False)

	# get the material library
	materialLibrary1 = GetMaterialLibrary()

	# update the view to ensure updated data information
	renderView1.Update()

	# set scalar coloring
	ColorBy(more_reduced_pressure_datavtiDisplay, ('POINTS', 'ImageScalars'))

	# rescale color and/or opacity maps used to include current data range
	more_reduced_pressure_datavtiDisplay.RescaleTransferFunctionToDataRange(True, True)

	# change representation type
	more_reduced_pressure_datavtiDisplay.SetRepresentationType('Volume')

	# rescale color and/or opacity maps used to include current data range
	more_reduced_pressure_datavtiDisplay.RescaleTransferFunctionToDataRange(True, False)

	# get color transfer function/color map for 'ImageScalars'
	imageScalarsLUT = GetColorTransferFunction('ImageScalars')

	# get opacity transfer function/opacity map for 'ImageScalars'
	imageScalarsPWF = GetOpacityTransferFunction('ImageScalars')

	# get 2D transfer function for 'ImageScalars'
	imageScalarsTF2D = GetTransferFunction2D('ImageScalars')

	# get layout
	layout1 = GetLayout()

	# layout/tab size in pixels
	layout1.SetSize(1538, 789)

	# current camera placement for renderView1
	renderView1.CameraPosition = [66.05031099881897, 70.23731645548888, -211.83514119706007]
	renderView1.CameraFocalPoint = [112.19960784912105, 120.5106811523438, 8.221492619567494]
	renderView1.CameraViewUp = [0.6421691699444195, 0.7070260097916965, -0.29619753316144803]
	renderView1.CameraParallelScale = 59.63074581337485

	camera=GetActiveCamera()
	renderView1.ResetCamera()
	camera.Elevation(viewpoint[0]) 
	camera.Azimuth(viewpoint[1])
	renderView1.Update()

	# save screenshot
	SaveScreenshot(f'InitImages/image_{img_ind}.png', renderView1, ImageResolution=[1538, 789])
	SaveScreenshot(f'OptiImages/volimage_2_{img_ind}.png', renderView1, ImageResolution=[1538, 789])


#objective function
def objfunc(viewpoint,img_ind):
	#get image from viewpoint
	generate_image(viewpoint,img_ind)

	image_path = 'OptiImages/volimage_2_' + str(img_ind) + '.png'


	image = Image.open(image_path)
	image = ImageOps.grayscale(image)
	image_array = np.array(image)

	overall_entropy = skimage.measure.shannon_entropy(image_array)
	print(f'Overall entropy of the image: {overall_entropy} ' + str(img_ind) + " "  + str(rank))
	return -1*(overall_entropy) # we minimise this


#optimiser
def optim(x0,img_ind):
	bounds = [(x0[0] -20,  x0[0] + 20), (x0[1] -20 , x0[1] + 20)]
	#minimize objective function using Nelder-Mead within bounds with max iterations and change in function values
	result = minimize(objfunc, x0 = x0, bounds = bounds, method= 'Nelder-Mead', options={ 'fatol' : 0.01, 'maxiter': 20}, args = {img_ind})
	return result.x

#main

if rank != 0:
	#take initial viewpoints
	[view,img_ind] = comm.recv()
	print('Process with rank=' + str(rank) + ' initialised.',flush = True)


#creating new threads to replace completed ones
	while(1):	
		print(view)
		optview = optim(view,img_ind)
		print("Image " + str(img_ind) + " optimised. Entropy: ",round(-1*objfunc(view,88),3),"->",round(-1*objfunc(optview,0),3),flush = True)
		#ask for new viewpoint
		comm.send(rank, dest=0)
		
		#get new viewpoint
		[view,img_ind] = comm.recv()
		if (view == 1):
			os._exit(0)
		print(str(rank) + ' got new viewpoint, for image ' + str(img_ind) + '.',flush = True)
		#start new thread with new viewpoint and same name as dead thread
		


elif rank == 0:

	start = time.time()

	#get data from csv
	img_ind = 1
	my_data = np.loadtxt(r'data.csv', delimiter=',')
	n = len(my_data)
	my_data = my_data.tolist()
	my_data = my_data[1:]
	my_data = my_data[:int(sys.argv[1])]
	print(my_data,flush = True)

	#initialise all processes
	for rrank in range (1,size):
		comm.send([my_data[0],img_ind], dest= rrank)
		img_ind = img_ind + 1
		my_data = my_data[1:] #remove first 1 element

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
	end = time.time()	
	print("Time taken :", end-start)
	n = 1
	while (n < size):
		comm.send([1,img_ind], dest = n)
		n = n + 1
	MPI.Finalize()
	end = time.time()	
	print("Time taken :", end-start)
	
	#writing optimisarion time results to csv
	with open('results2.csv','a') as file:
		wobj = writer(file)
		wobj.writerow([size,sys.argv[1],end - start,1])
		file.close()

