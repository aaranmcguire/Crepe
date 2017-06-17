--[[
Main Driver for Crepe
By Xiang Zhang @ New York University
]]

-- Necessary functionalities
require("nn")
require("torch")
require("cutorch")
require("cunn")
require("cudnn")

-- Local requires
require("data")
require("network")
require("train")

-- Configurations
dofile("config.lua")

-- Prepare random number generator
math.randomseed(os.time())
torch.manualSeed(os.time())

-- Create namespaces
main = {}

-- The main program
function main.main()
   -- Setting the device
   if config.main.device then
      cutorch.setDevice(config.main.device)
      print("Device set to "..config.main.device)
   end
   
   --cudnn.fastest = true

   main.new()
   main.run()
end


-- Train a new experiment
function main.new()
   -- Load the model
   print("Loading the model...")
   
   print("Model: ")
   local model = Network:model()
   print(model:__tostring())

   -- Initiate the trainer
   print("Loading the trainer...")
   main.train_data = Data(config.train_data)
   main.train = Train(main.train_data, Network)

   collectgarbage()
end

-- Start the training
function main.run()
   --Run for this number of era
   for i = 1,config.main.eras do
      
      print("Training for era "..i)
      main.train:run()
      collectgarbage()
   end
end


-- Utility function: find files with the specific 'ls' pattern
function main.findFiles(pattern)
   require("sys")
   local cmd = "ls "..pattern
   local str = sys.execute(cmd)
   local files = {}
   for file in str:gmatch("[^\n]+") do
      files[#files+1] = file
   end
   return files
end

-- Execute the main program
main.main()
