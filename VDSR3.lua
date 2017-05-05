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
criterion.sizeAverage = true

--Create VDSR conv neural network
--http://cv.snu.ac.kr/research/VDSR/VDSR_CVPR2016.pdf
vdsrcnn = nn.Sequential()

vdsrcnn:add(nn.SpatialConvolutionMM(1, 64, 3, 3, 1, 1, 1, 1))
vdsrcnn:add(nn.ReLU(true)) --ReLU
vdsrcnn:add(nn.SpatialConvolutionMM(64, 64, 3, 3, 1, 1, 1, 1))
vdsrcnn:add(nn.ReLU(true))
vdsrcnn:add(nn.SpatialConvolutionMM(64, 64, 3, 3, 1, 1, 1, 1))
vdsrcnn:add(nn.ReLU(true))
vdsrcnn:add(nn.SpatialConvolutionMM(64, 64, 3, 3, 1, 1, 1, 1))
vdsrcnn:add(nn.ReLU(true))
vdsrcnn:add(nn.SpatialConvolutionMM(64, 64, 3, 3, 1, 1, 1, 1))
vdsrcnn:add(nn.ReLU(true))
vdsrcnn:add(nn.SpatialConvolutionMM(64, 64, 3, 3, 1, 1, 1, 1))
vdsrcnn:add(nn.ReLU(true))
vdsrcnn:add(nn.SpatialConvolutionMM(64, 64, 3, 3, 1, 1, 1, 1))
vdsrcnn:add(nn.ReLU(true))
vdsrcnn:add(nn.SpatialConvolutionMM(64, 64, 3, 3, 1, 1, 1, 1))
vdsrcnn:add(nn.ReLU(true))
vdsrcnn:add(nn.SpatialConvolutionMM(64, 64, 3, 3, 1, 1, 1, 1))
vdsrcnn:add(nn.ReLU(true))
vdsrcnn:add(nn.SpatialConvolutionMM(64, 64, 3, 3, 1, 1, 1, 1))
vdsrcnn:add(nn.ReLU(true))
vdsrcnn:add(nn.SpatialConvolutionMM(64, 64, 3, 3, 1, 1, 1, 1))
vdsrcnn:add(nn.ReLU(true))
vdsrcnn:add(nn.SpatialConvolutionMM(64, 1, 3, 3, 1, 1, 1, 1))
vdsrcnn:add(nn.CAdd(1))

local getBias = function(module)
  return module.bias
end

local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m:init('weight', nninit.kaiming, {dist = 'normal', gain = {'lrelu', leakiness = 0.1}})
      --m:init('weight', nninit.kaiming, {dist = 'uniform', gain = {'relu'}})
       --:init('weight', nninit.mulConstant, 9/10)
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

function swap(array, index1, index2)
    array[index1], array[index2] = array[index2], array[index1]
end

function shuffle(array, array2)
    local counter = #array
    while counter > 1 do
        local index = math.random(counter)
        swap(array, index, counter)
        swap(array2, index, counter)
        counter = counter - 1
    end
end

function subrange(t, first, last)
  local subt = {}
  for i=first,last do
    subt[#subt + 1] = t[i]
  end
  return subt
end

local hr = {}
local lr = {}

local imagesn = 80


for k=1, imagesn do
	hr[k] = image.rgb2y(image.crop(image.load("train/" .. k .. ".png", 3, "float"),0,0,100,100))
	hr[k+imagesn] = image.hflip(hr[k])
	hr[k+2*imagesn] = image.vflip(hr[k])
	hr[k+3*imagesn] = image.rotate(hr[k], 90)
	--lr[k] = image.rgb2y(image.load("train/" .. k .. ".png", 3, "float"))
	lr[k] = image.rgb2y(image.scale(image.scale(image.crop(image.load("train/" .. k .. ".png", 3, "float"),0,0,100,100), "*1/2"), "*2"))
	lr[k+imagesn] = image.hflip(lr[k])
	lr[k+2*imagesn] = image.vflip(lr[k])
	lr[k+3*imagesn] = image.rotate(lr[k], 90)

end

--shuffle(hr, lr)

--checkhr = image.hflip(hr[1])
--checklr = image.hflip(lr[1])
--checkresid = checkhr:csub(checklr):add(0.5)

--image.save("CheckResid.png", checkresid)

local x;
local y;

local batchsize = 10
local minibatch = (imagesn*4)/batchsize --#Of iterations before going through entire batch

function setBatch(iter)
	local batch = iter%minibatch
	local start = batch*batchsize+1
	local eloc = batch*batchsize+batchsize
	
	x = TableToTensor(subrange(lr, start, eloc)):type(dtype)
	y = TableToTensor(subrange(hr, start, eloc)):type(dtype)

end



setBatch(0)


--Initialise training variables
params, gradParams = vdsrcnn:getParameters()
local optimState = {learningRate = 0.1, weightDecay = 0.0001, momentum = 0.9}
local loss = 1;
local showlossevery = 100;
--0.02
local cnorm = 0.01 * optimState.learningRate --Gradient Clipping


--Training function
function f(params)

	vdsrcnn:zeroGradParameters();

	--Forward the x values
	--local out = torch.add(vdsrcnn:forward(x), x)
	--Compute the loss, add the network output to LR image
	
	local out = vdsrcnn:forward(x)
	local diff = y:clone():csub(x)
	
	loss = criterion:forward(out, diff)
	--Compute the gradient

	local lrate = optimState.learningRate

	local grad_out = criterion:backward(out, diff)
	grad_out:clamp(-cnorm/lrate, cnorm/lrate)

	--Zero the previous gradient, and backpropagate the new gradient
	gradParams:zero()
	vdsrcnn:backward(x, grad_out)
	
		-- normalize gradients and f(X)
	--gradParams:div(batchsize) -- division by batch_size
	--loss = loss/batchsize -- division by batch_size
	
	
	gradParams:clamp(-cnorm/lrate, cnorm/lrate) --Clip the gradients
	
	

	--Return the loss and new gradient parameters to the optim.adam() function
	return loss, gradParams
end

local decreaseRate = 0.1

--local firstdiff = vdsrcnn:forward(x)
--image.save("test/Epoch" .. 0 .. "resid.png", firstdiff:add(0.5))



local trainLRY = image.rgb2y(image.load("test.png", 3, "float")):type(dtype)
local trainLR = image.rgb2yuv(image.load("test.png", 3, "float")):type(dtype)
local trainDiff = vdsrcnn:forward(trainLRY)

local trainSR = trainLR

trainSR[1]:add(trainDiff[1])

image.save("SR.png", image.yuv2rgb(trainSR))



local epoch = 0;

--For 500 iters
for iter = 1, 40000 do
	--Use torch-optim
	if (iter%500 == 0) then
		optimState.learningRate = optimState.learningRate * decreaseRate
		print("Reducing learning rate by a factor of " .. decreaseRate .. ". New learning rate: " .. optimState.learningRate)
	end
	optim.sgd(f, params, optimState)
	
	if ((iter%showlossevery == 0) or (iter%20 == 0 and iter < 200) or (iter < 20)) then
		print("Epoch " .. epoch .. " Iteration " .. iter .. " Training Loss " .. loss)
		--local epochdiff = vdsrcnn:forward(trainLR)
		--image.save("test/Iter" .. iter .. "resid.png", image.yuv2rgb(epochdiff):add(0.5))
	end
	
	if (iter%100 == 0) then
	
		torch.save("save/nn" .. iter .. ".cv", vdsrcnn)
		
	end
	
	if (iter%minibatch == minibatch-1) then
		epoch = epoch+1
	end
	setBatch(iter)

end

--Validate the result
local trainLRY = image.rgb2y(image.load("test.png", 3, "float")):type(dtype)
local trainLR = image.rgb2yuv(image.load("test.png", 3, "float")):type(dtype)
local trainDiff = vdsrcnn:forward(trainLRY)

local trainSR = trainLR

trainSR[1]:add(trainDiff[1])

image.save("SR.png", image.yuv2rgb(trainSR))




