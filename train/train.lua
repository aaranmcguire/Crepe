local Train = torch.class("Train")

function Train:__init(data, network)
   -- Set vars
   self.module = network:model();
   self.criterion = network:loss();
   self.data = data;

   -- Load Network into GPU
   self.module = self.model:cuda();
   self.criterion = self.loss:cuda();
   
   print("Ready to train...")
end

function Train:train()
   
   
   
   print(data[1])
   
end
