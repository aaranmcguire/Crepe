
-- Necessary functionalities
require("nn")
require("torch")
require("cutorch")
require("cunn")
require("cudnn")

-- Local requires
require("data")
require("network")

-- Configurations
dofile("config.lua")

-- Create Namespace
Train = {}


-- The Main Program
function Train.main()
	print("Loading Module...")
	
	module = torch.load(paths.concat("/data/", "TestModel.t7"))
	module:evaluate()
	
	print("Loading Test Data...")
	train_data = Data(config.val_data)
	
	--print(module:forward(input))
end

-- Execute the main program
Train.main()
