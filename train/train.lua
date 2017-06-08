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
   
   local randomClass = torch.random(#data.data.index)
   --^^ Random select of one of the classifications.
   
   local randomDataFromClass = torch.random(data.data.index[randomClass]:size(1))
   --^^ Random select or one of the data inputs from teh selected Class.
   
   local dataString = data.data.index[randomClass][randomDataFromClass][ data.data.index[randomClass][randomDataFromClass]:size(1) ]
   --^^  ¯\_(ツ)_/¯ -- No clue what this is doing.
   
   
   print(dataString)
   
   return formatedData
end


function Train:run()
   
   
end
