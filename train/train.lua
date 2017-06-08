local ffi = require("ffi")
local tds = require ("tds")

local Train = torch.class("Train")

function Train:__init(data, network)
   -- Set vars
   self.data = data;
   self.module = network:model();
   self.criterion = network:loss();
   self.dataset = self:formatData(data);

   -- Load Network into GPU
   self.module = self.module:cuda();
   self.criterion = self.criterion:cuda();
   
   self.dataset.data = self.dataset.data:cuda();
   --self.dataset.label = self.dataset.label:cuda();
   
   print("Ready to train...")
end

function Train:formatData(data)
   local i = 1;
   local formatedData = tds.Hash()
   
   for class = 1, #data.data.index do
      print('Class #:'..class);
      print('# of data in class: '..data.data.index[class]:size(1));
      
      for dataID = 1, data.data.index[class]:size(1) do
       
         formatedData[data][i] = self:toTensor(
            ffi.string(
               torch.data(
                  data.data.content:narrow(
                     1, data.data.index[class][dataID][( data.data.index[class][dataID]:size(1) )], 1
                  )
               )
            ):lower()
         , 1014);
         
         formatedData[label][i] = class;
         i = i + 1;
      end
   end

   return formatedData

end


function Train:toTensor(data, length)
   
   local tensor = torch.Tensor(#self.data.alphabet, length);
   tensor:zero();
   for i = #data, math.max(#data - length + 1, 1), -1 do
      
      if self.data.dict[data:sub(i,i)] then
         tensor[self.data.dict[data:sub(i,i)]][#data - i + 1] = 1;
      end
   end
   
   return tensor;
end

function Train:run()
   
   
end
