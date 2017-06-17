
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
	
	cutorch.setDevice(1)
	
	print("Loading Module...")
	
	module = torch.load(paths.concat("/data/", "TestModel.t7"))
	module = module:cuda()
	module:evaluate()
	
	print("Loading Test Data...")
	data = Data(config.val_data)
	train_data = data.data
	
	print("Testing..")
	for i = 1, #train_data do	
		print( "Prediction: " )
		
		local input = data:stringToTensor(train_data[i]["data"], 1024, torch.Tensor(1, 1024, 69))
		local output = module:forward(input)
		print( "Fact: "..train_data[i]["label"] )
		print( "---" )
	end
	
	--print(module:forward(input))
end

-- Execute the main program
Train.main()
