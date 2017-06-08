local Train = torch.class("Train")

function Train:__init(data, network)
   -- Set vars
   self.module = network:model();
   self.criterion = network:loss();
   self.data = data;

   -- Load Network into GPU
   self.module = self.module:cuda();
   self.criterion = self.criterion:cuda();
   
   print("Ready to train...")
end

function Train:run()
   
   print(self.data[1])
   
end
