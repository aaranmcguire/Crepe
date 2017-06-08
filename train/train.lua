local Train = torch.class("Train")

function Train:__init(data, network)
   -- Set vars
   self.module = network:model();
   self.criterion = network:loss();
   
   self:formatData(data);

   -- Load Network into GPU
   self.module = self.module:cuda();
   self.criterion = self.criterion:cuda();
   
   print("Ready to train...")
end

function Train:formatData(data)
   local formatedData = ''
   
   local randomClass = torch.random(#data.data.index) -- Select from one of the classifications.
   local randomDataFromClass = torch.random(data.data.index[randomClass]:size(1))
   
   print(randomDataFromClass)
   
   return formatedData
end


function Train:run()
   
   
end
