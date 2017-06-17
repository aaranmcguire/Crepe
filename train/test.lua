
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
	train_data = Data(config.val_data).data
	
	print("Testing..")
	for i = 1, #train_data do	
		print( "Prediction: " )
		print( module:forward(train_data[i]["data"]) )
		print( "Fact: "..train_data[i]["label"] )
		print( "---" )
	end
	
	--print(module:forward(input))
end

-- Execute the main program
Train.main()
