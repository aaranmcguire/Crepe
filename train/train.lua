local ffi = require("ffi")
local tds = require ("tds")

local Train = torch.class("Train")

function Train:__init(data, network)
   -- Set vars
   self.data = data;
   self.module = network:model();
   self.criterion = network:loss();

   -- Load Network into GPU
   self.module = self.module:cuda();
   self.criterion = self.criterion:cuda();
   
   self:loadData(data, 1000)
   
   print("Ready to train...")
end

function Train:loadData(data, batchSize)
   local i = 1;
   local batch = 1;
   local formatedData = {}

   for class = 1, #data.data.index do
      print('Class #:'..class);
      print('# of data in class: '..data.data.index[class]:size(1));
      
      for dataID = 1, data.data.index[class]:size(1) do  
         
         table.insert(formatedData, {
            ["data"] = ffi.string(
               torch.data(
                  data.data.content:narrow(
                     1, data.data.index[class][dataID][( data.data.index[class][dataID]:size(1) )], 1
                  )
               )
            ):lower(),
               
            ["label"] = class
         });
         
         i = i + 1;
         
         --if (i % batchSize == 0) then
         --   batch = batch + 1
         --end

      end
      collectgarbage()
   end

   return formatedData

end

function Train:loadBatch(num)

end
   
function Train:stringToTensor(data, length)
   
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
