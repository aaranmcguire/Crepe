
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
	tensor = torch.Tensor(1024, 69);
	
	print("Testing..")
	for batch = 1, #self.data.batches do
           print("Batch:"..batch)
           local trainset = self.data:loadBatch(batch)
      
           trainset.data = trainset.data:cuda()
           trainset.label = trainset.label:cuda()
           
           for t = 1,trainset:size() do
		local prediction = module:forward(trainset.data[t])
		local confidences, indices = torch.sort(prediction, true)
		print("Prediction: "..indices[1])
		print("Fact "..trainset.label[t])
			
	   end
      
      collectgarbage()
   end
	
	--print(module:forward(input))
end

-- Execute the main program
Train.main()
