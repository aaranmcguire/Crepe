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
   
   self.data = self:loadData(data)
   
   self.batches = self:createBatches()
   
   print("Ready to train...")
end

function Train:loadData(data, batchSize)
   local formatedData = {}

   for class = 1, #data.data.index do
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
      end
      collectgarbage()
   end

   return formatedData
end

function Train:createBatches(batchSize)
   local batch = 1
   local batchSize = batchSize or 1000
   local batches = {}
   
   for i = 1, #self.data do
      
      if type(batches[batch]) ~= 'table' then
         batches[batch] = {}
      end
      
      table.insert(batches[batch], self.data[i])
               
      if (i % batchSize == 0) then
         batch = batch + 1
      end
   end
   
   return batches
end

function Train:loadBatch(num)
   local data = {}
   local label = {}
   
   for i = 1, #self.batches[num] do
      
      data[i] = self.batches[num][i]["data"];
      label[i] = self.batches[num][i]["label"];
      
   end
   
   return {["data"] = data, ["label"] = label}
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
   
   for batch = 1, #self.batches do
      print("Batch:"..batch)
      local dataset = self:loadBatch(batch)
      
      
      
   end
   
end
