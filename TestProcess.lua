require 'torch'
require 'image'
require 'hdf5'

local x = image.load("LR.png", 3, "float")
local y = image.load("Resid.png", 3, "float")
local z = image.load("Test.png", 3, "float")


local myFile = hdf5.open('test.h5', 'w')
myFile:write('tin', x)
myFile:write('tout', y)
myFile:write('valid', z)
myFile:close()