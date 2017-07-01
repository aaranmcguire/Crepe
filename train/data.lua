require("image")
local ffi = require("ffi")

-- The class
local Data = torch.class("Data")

function Data:__init(config)
   -- Alphabet settings
   self.alphabet = config.alphabet or "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
   self.dict = {}
   for i = 1,#self.alphabet do
      self.dict[self.alphabet:sub(i,i)] = i
   end

   self.lenth = 1024;
   self.batchSize = 7000;
   self.file = config.file

   self.config = config
   self.data = torch.load(self.file)
	
   if config.limitDataSetSize then
      self.limitDataSetSize = config.limitDataSetSize
   end
	
   self.data = self:loadData()
   self.batches = self:createBatches()

end

function Data:loadData()
   local formatedData = {}
   local rand = math.random

   for class = 1, #self.data.index do
      for dataID = 1, (self.limitDataSetSize or self.data.index[class]:size(1)) do  --self.data.index[class]:size(1)
         
         table.insert(formatedData, {
            ["data"] = ffi.string(
               torch.data(
                  self.data.content:narrow(
                     1, self.data.index[class][dataID][( self.data.index[class][dataID]:size(1) )], 1
                  )
               )
            ):lower(),
               
            ["label"] = class
         });
      end
      collectgarbage()
   end

   for i = #formatedData, 2, -1 do
        j = rand(i)
        formatedData[i], formatedData[j] = formatedData[j], formatedData[i]
    end
	
   return formatedData
end

function Data:createBatches()
   local batch = 1
   local batches = {}
   
   for i = 1, #self.data do
      
      if type(batches[batch]) ~= 'table' then
         batches[batch] = {}
      end
      
      table.insert(batches[batch], self.data[i])
               
      if (i % self.batchSize == 0) then
         batch = batch + 1
      end
   end
   
   return batches
end


function Data:loadBatch(num)
   local data = torch.Tensor(#self.batches[num], self.lenth, #self.alphabet);
   local label = torch.Tensor(#self.batches[num])
   
   for i = 1, #self.batches[num] do
      
      data[i] = self:stringToTensor(self.batches[num][i]["data"], self.lenth, data:select(1, i))
      label[i] = self.batches[num][i]["label"];
      
   end
   
   local dataset = {["data"] = data, ["label"] = label}
   
   dataset.data:double()
   
   setmetatable(dataset, {
      __index = function(t, i) 
         return {
            t.data[i],
            t.label[i]
         } 
      end
   });

   function dataset:size() 
      return self.data:size(1) 
   end
   
   return dataset
end

function Data:stringToTensor(data, length, tensor)
   tensor:zero();
   for i = 1, length do
      if self.dict[data:sub(i,i)] then
         tensor[i][self.dict[data:sub(i,i)]] = 1;
      end
   end
   
   return tensor
end

return Data
