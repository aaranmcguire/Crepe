local ffi = require("ffi")

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
   
   local dataString = ffi.string(
      torch.data(
         data.data.content:narrow(
            1, data.data.index[randomClass][randomDataFromClass][( data.data.index[randomClass][randomDataFromClass]:size(1) )], 1
         )
      )
   )
   --^^  ¯\_(ツ)_/¯ -- No clue what this is doing, but this is the string of the input
   local s = '';
   for l = data.data.index[randomClass][randomDataFromClass]:size(1) - 1, 1, -1 do
	   s = s.." "..ffi.string(torch.data(data.data.content:narrow(1, data.data.index[randomClass][randomDataFromClass][l], 1)))
   end
   
   
   print(dataString)
   
   print('--')
   print(s)
   
   return formatedData
end


function Train:run()
   
   
end
