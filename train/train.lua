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
   
   print(data.data[2][300])
   
   return formatedData
end


function Train:run()
   
   
end
