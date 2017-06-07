--[[
Main Driver for Crepe
By Xiang Zhang @ New York University
]]

-- Necessary functionalities
require("nn")
require("cutorch")
require("cunn")

-- Local requires
require("data")
require("model")
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
   main.model = Model(config.model)
   if config.main.randomize then
      main.model:randomize(config.main.randomize)
      print("Model randomized.")
   end
   main.model:type(config.main.type)
   print("Current model type: "..main.model:type())
   collectgarbage()
   
   print("Model: ")
   print(main.model.sequential)

   -- Initiate the trainer
   print("Loading the trainer...")
   main.train = Train(main.train_data, main.model, config.loss(), config.train)

   -- Initiate the tester
   print("Loading the tester...")
   main.test_val = Test(main.val_data, main.model, config.loss(), config.test)

   collectgarbage()
end

-- Start the training
function main.run()
   --Run for this number of era
   for i = 1,config.main.eras do
   
      if config.main.dropout then
	 print("Enabling dropouts")
	 main.model:enableDropouts()
      else
	 print("Disabling dropouts")
	 main.model:disableDropouts()
      end
      
      print("Training for era "..i)
      main.train:run(config.main.epoches, main.trainlog)

      if config.main.test == true then
	 print("Disabling dropouts")
	 print("Testing on test data for era "..i)
	 main.test_val:run(main.testlog)
      end

      print("Saving data")
      main.save()
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

-- The training logging function
function main.trainlog(train)
   if config.main.collectgarbage and math.fmod(train.epoch-1,config.main.collectgarbage) == 0 then
      print("Collecting garbage at epoch = "..(train.epoch-1))
      collectgarbage()
   end
   
      local msg = ""
      
      if config.main.details then
	 msg = msg.."epo: "..(train.epoch-1)..
	    ", rat: "..string.format("%.2e",train.rate)..
	    ", err: "..string.format("%.2e",train.error)..
	    ", obj: "..string.format("%.2e",train.objective)
      end
      
      if config.main.debug then
	 msg = msg..", bmn: "..string.format("%.2e",train.batch:mean())..
	    ", bsd: "..string.format("%.2e",train.batch:std())..
	    ", bmi: "..string.format("%.2e",train.batch:min())..
	    ", bmx: "..string.format("%.2e",train.batch:max())..
	    ", pmn: "..string.format("%.2e",train.params:mean())..
	    ", psd: "..string.format("%.2e",train.params:std())..
	    ", pmi: "..string.format("%.2e",train.params:min())..
	    ", pmx: "..string.format("%.2e",train.params:max())..
	    ", gmn: "..string.format("%.2e",train.grads:mean())..
	    ", gsd: "..string.format("%.2e",train.grads:std())..
	    ", gmi: "..string.format("%.2e",train.grads:min())..
	    ", gmx: "..string.format("%.2e",train.grads:max())..
	    ", omn: "..string.format("%.2e",train.old_grads:mean())..
	    ", osd: "..string.format("%.2e",train.old_grads:std())..
	    ", omi: "..string.format("%.2e",train.old_grads:min())..
	    ", omx: "..string.format("%.2e",train.old_grads:max())
      end
      
      if config.main.details or config.main.debug then
	 print(msg)
      end

end

function main.testlog(test)
   if config.main.collectgarbage and math.fmod(test.n,config.train_data.batch_size*config.main.collectgarbage) == 0 then
      print("Collecting garbage at n = "..test.n)
      collectgarbage()
   end
   if not config.main.details then return end
   print("n: "..test.n..
	", e: "..string.format("%.2e",test.e)..
	", l: "..string.format("%.2e",test.l)..
	", err: "..string.format("%.2e",test.err)..
	", obj: "..string.format("%.2e",test.objective))
      
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
