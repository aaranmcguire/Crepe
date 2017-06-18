local ffi = require("ffi")

local Train = torch.class("Train")

function Train:__init(data, network)
   -- Set vars
   self.data = data;

   self.module = network:model();
   self.criterion = network:loss();

   -- Load Network into GPU
   self.module = self.module:cuda();
   self.criterion = self.criterion:cuda();
   
   print("Ready to train...")
end


function Train:run()
   
   trainer = nn.StochasticGradient(self.module, self.criterion)
   trainer.learningRate = 0.001
   trainer.maxIteration = 500 -- just do 5 epochs of training.
   trainer.shuffleIndices = false
   
   print("Number of batches:"..#self.data.batches)
   
   for batch = 1, #self.data.batches do
      print("Batch:"..batch)
      local trainset = self.data:loadBatch(batch)
      
      trainset.data = trainset.data:cuda()
      trainset.label = trainset.label:cuda()
      
      trainer:train(trainset)
      
      if batch % 10 == 0 or batch == #self.data.batches then
         filename = paths.concat('/data', 'Train_B'..batch..'_Model.t7')
         torch.save(filename, self.module:clearState())
         print("Saving batch "..batch)
      end
      
      collectgarbage()
   end
   
end
