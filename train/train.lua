local Train = torch.class("Train")

function Train:__init(data, network)
   -- Set vars
   self.module = network:model();
   self.criterion = network:loss();
   self.data = self:formatData(data);

   -- Load Network into GPU
   self.module = self.module:cuda();
   self.criterion = self.criterion:cuda();
   
   print("Ready to train...")
end

function Train:formatData(data)
   local formatedData = ''
   
   for batch,labels,n in data:iterator() do
   
      print(batch)
      
      
   end
   
   return formatedData
end


function Train:run()
   
   
end
