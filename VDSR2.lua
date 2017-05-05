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
vdsrcnn:add(nn.ReLU(true)) --ReLU
vdsrcnn:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1))
vdsrcnn:add(nn.ReLU(true))
vdsrcnn:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1))
vdsrcnn:add(nn.ReLU(true))
vdsrcnn:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1))
vdsrcnn:add(nn.ReLU(true))
vdsrcnn:add(nn.SpatialConvolution(64, 3, 3, 3, 1, 1, 1, 1))

local getBias = function(module)
  return module.bias
end

local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m:init('weight', nninit.kaiming, {dist = 'normal', gain = {'relu'}})
      --m:init('weight', nninit.kaiming, {dist = 'uniform', gain = {'relu'}})
       --:init('weight', nninit.mulConstant, 5/6)
       --:init('weight', nninit.addNormal, 0, 0.01)
       :init(getBias, nninit.constant, 0)
   end
end

vdsrcnn:apply(weights_init)

--Set the network to the dtype
vdsrcnn:type(dtype)

--Create training data

function TableToTensor(table)
  local tensorSize = table[1]:size()
  local tensorSizeTable = {-1}
  for i=1,tensorSize:size(1) do
    tensorSizeTable[i+1] = tensorSize[i]
  end
  merge=nn.Sequential()
    :add(nn.JoinTable(1))
    :add(nn.View(unpack(tensorSizeTable)))

  return merge:forward(table)
end

local hr = {}
local lr = {}

for k=1, 48 do
	hr[k] = image.load("train/" .. k .. "HR.png", 3, "float")
	lr[k] = image.load("train/" .. k .. "LR.png", 3, "float")

end


local x;
local y;

function setBatch(iter)
	local batch = iter%4
	print(batch)
	local start = batch*12+1
	local eloc = batch*12+12
	
	x = TableToTensor({table.unpack(lr, start, eloc)}):type(dtype)
	y = TableToTensor({table.unpack(hr, start, eloc)}):type(dtype)

end

setBatch(0)

--local gimg = image.load("grad.png", 3, "float")
--print(gimg)
--local diff = hr:csub(lr):type(dtype)

--local tlr = lr:type(dtype)
--local thr = hr:type(dtype)

--local x = tlr
--local y = thr
print(x[1][1][1][30])
print(y[1][1][1][30])



--Initialise training variables
params, gradParams = vdsrcnn:getParameters()
local optimState = {learningRate = 0.1, weightDecay = 0.0001, momentum = 0.9}
local loss = 1;
local showlossevery = 1;
--0.02
local cnorm = 0.02 * optimState.learningRate --Gradient Clipping


--Training function
function f(params)
	vdsrcnn:zeroGradParameters();

	--Forward the x values
	local out = vdsrcnn:forward(x)
	--Compute the loss, add the network output to LR image
	out:add(x)
	loss = criterion:forward(out, y)
	--Compute the gradient
	local grad_out = criterion:backward(out, y)

	local lrate = optimState.learningRate

	--local grad_out = criterion:backward(out, y):clamp(-cnorm/lrate, cnorm/lrate)

	--Zero the previous gradient, and backpropagate the new gradient

	vdsrcnn:backward(x, grad_out)
	


	gradParams:clamp(-cnorm/lrate, cnorm/lrate) --Clip the gradients
	--gradParams:clamp(-cnorm/lrate, cnorm/lrate) --Clip the gradients

	--Return the loss and new gradient parameters to the optim.adam() function
	return loss, gradParams
end

local decreaseRate = 0.1

--local firstdiff = vdsrcnn:forward(x)
--image.save("test/Epoch" .. 0 .. "resid.png", firstdiff:add(0.5))

local imagesn = 48
local batchsize = 12
local minibatch = imagesn/batchsize

local epoch = 0;

--For 500 epochs
for iter = 1, 500 do
	--Use torch-optim
	if (iter%200 == 0) then
		optimState.learningRate = optimState.learningRate * decreaseRate
		print("Reducing learning rate by a factor of " .. decreaseRate .. ". New learning rate: " .. optimState.learningRate)
	end
	optim.sgd(f, params, optimState)
	
	if (iter%showlossevery == 0) then
		print("Epoch " .. epoch .. " Iteration " .. iter .. " Training Loss " .. loss)
		--local epochdiff = vdsrcnn:forward(x)
		--image.save("test/Epoch" .. epoch .. "resid.png", epochdiff:add(0.5))
	end
	
	if (iter%minibatch == minibatch-1) then
		epoch = epoch+1
	end
	setBatch(iter)

end

--Validate the result

local trainLR = image.load("LR.png", 3, "float"):type(dtype)
local trainDiff = vdsrcnn:forward(trainLR)

local trainHR = torch.add(trainLR, trainDiff)
print(trainLR[1][1][30])
print(trainDiff[1][1][30])
print(trainHR[1][1][30])


local testLR = image.load("TestLR.png", 3, "float"):type(dtype)
local testDiff = vdsrcnn:forward(testLR)
local testHR = testLR:add(testDiff)

image.save("TrainResid.png", trainDiff:add(0.5))
image.save("TrainOut.png", trainHR)
image.save("TestOut.png", testHR)
