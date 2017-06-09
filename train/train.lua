local ffi = require("ffi")
local tds = require ("tds")

local Train = torch.class("Train")

function Train:__init(data, network)
   -- Set vars
   self.data = data;
   self.alphabet = data.alphabet;
   self.dict = data.dict;
   self.batchSize = 500;
   
   self.module = network:model();
   self.criterion = network:loss();

   -- Load Network into GPU
   self.module = self.module:cuda();
   self.criterion = self.criterion:cuda();
   
   self.data = self:loadData(data)
   
   self.batches = self:createBatches()
   
   
   print("Ready to train...")
end

function Train:loadData(data)
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

function Train:createBatches()
   local batch = 1
   local batchSize = self.batchSize or 1000
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
   local data = torch.Tensor(self.batchSize, #self.alphabet, 1014);
   data:zero();
   local label = torch.Tensor(self.batchSize)
   
   for i = 1, #self.batches[num] do
      
      --self:stringToTensor(self.batches[num][i]["data"], 1014, data:select(1, i));
      label[i] = self.batches[num][i]["label"];
      
   end
   
   return {["data"] = data, ["label"] = label}
end
   
function Train:stringToTensor(data, length, tensor)
   for i = #data, math.max(#data - length + 1, 1), -1 do
      
      if self.dict[data:sub(i,i)] then
         tensor[self.dict[data:sub(i,i)]][#data - i + 1] = 1;
      end
   end
end

function Train:run()
   
   for batch = 1, #self.batches do
      print("Batch:"..batch)
      local trainset = self:loadBatch(batch)
      
      
   end
   
end
