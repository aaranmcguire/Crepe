
-- Necessary functionalities
require("nn")
require("torch")
require("cutorch")
require("cunn")
require("cudnn")

-- Local requires
require("data")
require("network")


-- Create Namespace
Train = {}


-- The Main Program
function Train.main()
	print("testing...")
	
	module = torch.load(paths.concat("/data/", "TestModel.t7"))
	
	module:evaluate()
	
	
	--print(module:forward(input))
end

-- Execute the main program
Train.main()
