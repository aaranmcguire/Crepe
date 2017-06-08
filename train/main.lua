--[[
Main Driver for Crepe
By Xiang Zhang @ New York University
]]

-- Necessary functionalities
require("nn")
require("cutorch")
require("cunn")
require("cudnn")

-- Local requires
require("data")
require("network")
require("train")
require("test")

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
   -- Load the data
   print("Loading datasets...")
   main.train_data = Data(config.train_data)
   main.val_data = Data(config.val_data)

   -- Load the model
   print("Loading the model...")
   
   print("Model: ")
   local model = Network:model()
   print(model:__tostring())

   
   -- Initiate the trainer
   print("Loading the trainer...")
   main.train = Train(main.train_data, Network)

   -- Initiate the tester
   --print("Loading the tester...")
   --main.test_val = Test(main.val_data, main.model, config.loss(), config.test)

   collectgarbage()
end

-- Start the training
function main.run()
   --Run for this number of era
   for i = 1,config.main.eras do
      
      print("Training for era "..i)
      main.train:run(config.main.epoches)

      --if config.main.test == true then
	-- print("Disabling dropouts")
	-- print("Testing on test data for era "..i)
	-- main.test_val:run(main.testlog)
      --end

      --print("Saving data")
      --main.save()
      collectgarbage()
   end
end

-- Save a record
function main.save()
   -- Record necessary configurations
   config.train.epoch = main.train.epoch

   -- Make the save
    local filename
    local modelObjectToSave
    
    if main.model.sequential.clearState then
        -- save the full model
        filename = paths.concat(config.main.save, '_' .. (main.train.epoch-1) .. '_Model.t7')
        modelObjectToSave = main.model.sequential:clearState()
    else
        -- this version of Torch doesn't support clearing the model state => save only the weights
        local Weights,Gradients = main.model.sequential:getParameters()
        filename = paths.concat(config.main.save, '_' .. (main.train.epoch-1) .. '_Weights.t7')
        modelObjectToSave = Weights
    end
    print('Snapshotting to ' .. filename)
    torch.save(filename, modelObjectToSave)
    print('Snapshot saved - ' .. filename)
   
   collectgarbage()
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
