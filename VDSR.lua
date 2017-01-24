require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'hdf5'
local nninit = require 'nninit'
--Use FloatTensor for faster but less precise calculations
--Default is DoubleTensor
local dtype = 'torch.FloatTensor'

local useOpenCl = true;

--If we are using opencl, we change the tensor dtype to "ClTensor" using :cl();
if (useOpenCl) then
	require 'cltorch'
	require 'clnn'
	dtype = torch.FloatTensor():cl():type()
end
--torch.setdefaulttensortype(dtype)
--cltorch.setDevice(1)

--Create Loss Function
local criterion = nn.MSECriterion():type(dtype)

--Create VDSR conv neural network
--http://cv.snu.ac.kr/research/VDSR/VDSR_CVPR2016.pdf
vdsrcnn = nn.Sequential()

vdsrcnn:add(nn.SpatialConvolution(3, 64, 3, 3, 1, 1, 1, 1))
vdsrcnn:add(nn.ELU(0.6, false)) --ReLU
vdsrcnn:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1))
vdsrcnn:add(nn.ELU(0.6, false))
vdsrcnn:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1))
vdsrcnn:add(nn.ELU(0.6, false))
vdsrcnn:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1))
vdsrcnn:add(nn.ELU(0.6, false))
vdsrcnn:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1))
vdsrcnn:add(nn.ELU(0.6, false))
vdsrcnn:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1))
vdsrcnn:add(nn.ELU(0.6, false))
vdsrcnn:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1))
vdsrcnn:add(nn.ELU(0.6, false))
vdsrcnn:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1))
vdsrcnn:add(nn.ELU(0.6, false))
vdsrcnn:add(nn.SpatialConvolution(64, 3, 3, 3, 1, 1, 1, 1))

local getBias = function(module)
  return module.bias
end

local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m:init('weight', nninit.eye)
       :init('weight', nninit.mulConstant, 1/2)
       :init('weight', nninit.addNormal, 0, 0.01)
       :init(getBias, nninit.constant, 0)
   end
end

vdsrcnn:apply(weights_init)

--Set the network to the dtype
vdsrcnn:type(dtype)

--Create training data
local x = image.load("LR.png", 3, "float"):type(dtype)


local y = image.load("Resid.png", 3, "float"):type(dtype)

--Test data

--local z = image.load("Test.png", 3, "float"):type(dtype)




--Initialise training variables
params, gradParams = vdsrcnn:getParameters()
local optimState = {learningRate = 0.1}
local loss = 1;
local showlossevery = 1;
local cnorm = 0.02 * 0.1 --Gradient Clipping


--Training function
function f(params)
	vdsrcnn:zeroGradParameters();

	--Forward the x values
	local out = vdsrcnn:forward(x)
	--Compute the loss
	loss = criterion:forward(out, y)
	--Compute the gradient
	local grad_out = criterion:backward(out, y)
	--local grad_out = criterion:backward(out, y):clamp(-0.1, 0.1)

	--Zero the previous gradient, and backpropagate the new gradient

	vdsrcnn:backward(x, grad_out)
	
	local lrate = optimState.learningRate

	gradParams:clamp(-cnorm/lrate, cnorm/lrate) --Clip the gradients

	--Return the loss and new gradient parameters to the optim.adam() function
	return loss, gradParams
end

--For 500 epochs
for epoch = 1, 300 do
	--Use torch-optim
	if (epoch%20 == 0) then
		optimState.learningRate = optimState.learningRate * 0.7
	end
	optim.adam(f, params, optimState)
	print(epoch);
	if (epoch%showlossevery == 0) then
		print(loss)
	end

end

--Validate the result

image.save("orig.png", image.load("LR.png", 3, "float"))
image.save("TrainOut.png", vdsrcnn:forward(image.load("LR.png", 3, "float"):type(dtype)))
image.save("TestOut.png", vdsrcnn:forward(image.load("Test.png", 3, "float"):type(dtype)))
