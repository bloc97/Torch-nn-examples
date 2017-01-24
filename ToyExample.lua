require 'torch'
require 'nn'
require 'optim'
--Use FloatTensor for faster but less precise calculations
--Default is DoubleTensor
local dtype = 'torch.FloatTensor'

local useOpenCl = true;

--If we are using opencl, we change the tensor dtype to "ClTensor" using :cl();
if (useOpenCl) then
	require 'cltorch'
	require 'clnn'
	dtype = torch.Tensor():cl():type()
end
--torch.setdefaulttensortype(dtype)
--cltorch.setDevice(1)

--Create Loss Function
local criterion = nn.MSECriterion():type(dtype)

--Create toy neural network
mlp = nn.Sequential()

mlp:add(nn.Linear(2,5))
mlp:add(nn.Sigmoid()) --Activation layer
mlp:add(nn.Linear(5,1))

--Set the network to the dtype
mlp:type(dtype)

--Create training data. In this case, we will train a XOR Gate.
x = torch.Tensor({
    {0,0},
    {0,1},
    {1,0},
    {1,1}
}):type(dtype)


y = torch.Tensor({
    {0},
    {1},
    {1},
    {0}
}):type(dtype)

--Initialise training variables
params, gradParams = mlp:getParameters()
local optimState = {learningRate = 0.1}
local loss = 0;
local showlossevery = 100;

--Training function
function f(params)

	--Forward the x values
	local out = mlp:forward(x)
	--Compute the loss
	loss = criterion:forward(out, y)
	--Compute the gradient
	local grad_out = criterion:backward(out, y)

	--Zero the previous gradient, and backpropagate the new gradient
	mlp:zeroGradParameters();
	mlp:backward(x, grad_out)
	
	--Return the loss and new gradient parameters to the optim.adam() function
	return loss, gradParams
end

--For 500 epochs
for epoch = 1, 500 do
	--Use torch-optim
	optim.adam(f, params, optimState)
	if (epoch%showlossevery == 0) then
		print(loss)
	end

end

--Validate the result
local out = mlp:forward(x)
print(out)
