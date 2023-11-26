import numpy as np
from PIL import Image, ImageOps
from scipy.stats import entropy
from scipy.optimize import minimize
import sys
import skimage.measure
from zoopt import Dimension, Objective, Parameter, Opt
from paraview.simple import *


def generate_image(viewpoint):

	#### disable automatic camera reset on 'Show'
	paraview.simple._DisableFirstRenderCameraReset()

	# create a new 'NetCDF Reader'
	output_weather_simnc = NetCDFReader(registrationName='output_weather_sim.nc', FileName=['/users/misc/psogra20/Desktop/Project/output_weather_sim.nc'])
	output_weather_simnc.Dimensions = '(x, y, z)'

	# get active view
	renderView1 = GetActiveViewOrCreate('RenderView')

	# show data in view
	output_weather_simncDisplay = Show(output_weather_simnc, renderView1, 'UniformGridRepresentation')

	# trace defaults for the display properties.
	output_weather_simncDisplay.Representation = 'Outline'
	output_weather_simncDisplay.ColorArrayName = [None, '']
	output_weather_simncDisplay.SelectTCoordArray = 'None'
	output_weather_simncDisplay.SelectNormalArray = 'None'
	output_weather_simncDisplay.SelectTangentArray = 'None'
	output_weather_simncDisplay.OSPRayScaleArray = 'field'
	output_weather_simncDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
	output_weather_simncDisplay.SelectOrientationVectors = 'None'
	output_weather_simncDisplay.ScaleFactor = 6.300000000000001
	output_weather_simncDisplay.SelectScaleArray = 'None'
	output_weather_simncDisplay.GlyphType = 'Arrow'
	output_weather_simncDisplay.GlyphTableIndexArray = 'None'
	output_weather_simncDisplay.GaussianRadius = 0.315
	output_weather_simncDisplay.SetScaleArray = ['POINTS', 'field']
	output_weather_simncDisplay.ScaleTransferFunction = 'PiecewiseFunction'
	output_weather_simncDisplay.OpacityArray = ['POINTS', 'field']
	output_weather_simncDisplay.OpacityTransferFunction = 'PiecewiseFunction'
	output_weather_simncDisplay.DataAxesGrid = 'GridAxesRepresentation'
	output_weather_simncDisplay.PolarAxes = 'PolarAxesRepresentation'
	output_weather_simncDisplay.ScalarOpacityUnitDistance = 1.8966608180553592
	output_weather_simncDisplay.OpacityArrayName = ['POINTS', 'field']
	output_weather_simncDisplay.ColorArray2Name = ['POINTS', 'field']
	output_weather_simncDisplay.SliceFunction = 'Plane'
	output_weather_simncDisplay.Slice = 31
	output_weather_simncDisplay.SelectInputVectors = [None, '']
	output_weather_simncDisplay.WriteLog = ''

	# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
	output_weather_simncDisplay.ScaleTransferFunction.Points = [-4.3200000000000007e-05, 0.0, 0.5, 0.0, 123039.0, 1.0, 0.5, 0.0]

	# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
	output_weather_simncDisplay.OpacityTransferFunction.Points = [-4.3200000000000007e-05, 0.0, 0.5, 0.0, 123039.0, 1.0, 0.5, 0.0]

	# init the 'Plane' selected for 'SliceFunction'
	output_weather_simncDisplay.SliceFunction.Origin = [15.5, 31.5, 31.5]

	# reset view to fit data
	renderView1.ResetCamera(False)

	# get the material library
	materialLibrary1 = GetMaterialLibrary()

	# update the view to ensure updated data information
	renderView1.Update()

	# set scalar coloring
	ColorBy(output_weather_simncDisplay, ('POINTS', 'field'))

	# rescale color and/or opacity maps used to include current data range
	output_weather_simncDisplay.RescaleTransferFunctionToDataRange(True, True)

	# change representation type
	output_weather_simncDisplay.SetRepresentationType('Volume')

	# rescale color and/or opacity maps used to include current data range
	output_weather_simncDisplay.RescaleTransferFunctionToDataRange(True, False)

	# get color transfer function/color map for 'field'
	fieldLUT = GetColorTransferFunction('field')

	# get opacity transfer function/opacity map for 'field'
	fieldPWF = GetOpacityTransferFunction('field')

	# get 2D transfer function for 'field'
	fieldTF2D = GetTransferFunction2D('field')

	# get layout
	layout1 = GetLayout()

	# layout/tab size in pixels
	layout1.SetSize(1538, 789)

	# current camera placement for renderView1
	renderView1.CameraPosition = [15.5, 31.5, 213.7402813226413]
	renderView1.CameraFocalPoint = [15.5, 31.5, 31.5]
	renderView1.CameraParallelScale = 47.167255591140766
	
	camera=GetActiveCamera()
	renderView1.ResetCamera()
	camera.Elevation(viewpoint[0]) 
	camera.Azimuth(viewpoint[1])
	renderView1.Update()
	# save screenshot
	SaveScreenshot('/users/misc/psogra20/Desktop/Project/6.png', renderView1, ImageResolution=[1538, 789])

#objective function
def objfunc(viewpoint):
	generate_image(viewpoint)
	print(viewpoint)
	image_path = './6.png'


	image = Image.open(image_path)
	image = ImageOps.grayscale(image)
	image_array = np.array(image)

	overall_entropy = skimage.measure.shannon_entropy(image_array)
	print(f'Overall entropy of the image: {overall_entropy}')
	return -1*(overall_entropy) # we minimise this


#optimiser
def optim(x0):
	bounds = [(max(-90,x0[0] -20), min(90, x0[0] + 20)), (max(-180,x0[1] -20), min(180, x0[1] + 20))]
	result = minimize(objfunc, x0 = x0, bounds = bounds, method= 'Nelder-Mead', options={ 'maxfev' : 200})
	return result.x

#main
x0 = [0,0]
x0[0] = float(sys.argv[1])
x0[1] = float(sys.argv[2])
final_view = optim(x0)

