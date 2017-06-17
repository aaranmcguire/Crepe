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
   cutorch.setDevice(1)

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

   main.train:run()
   collectgarbage()
end

-- Execute the main program
main.main()
