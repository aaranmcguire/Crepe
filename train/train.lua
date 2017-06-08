local ffi = require("ffi")

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
   
   local formatedData = {}
   local formatedData.data = {}
   local formatedData.label = {}
   
   for class = 1, #data.data.index do
      print('Class #:'..class);
      print('# of data in class: '..data.data.index[class]:size(1));
      
      for dataID = 1, data.data.index[class]:size(1) do
       
         table.insert(formatedData.data, self:toTensor(
            ffi.string(
               torch.data(
                  data.data.content:narrow(
                     1, data.data.index[class][dataID][( data.data.index[class][dataID]:size(1) )], 1
                  )
               )
            ):lower()
         , 1014));
         
         table.insert(formatedData.label, class:type('torch.CudaTensor'));
         
      end
   end
   
   setmetatable(formatedData, 
      {
         __index = function(t, i) 
            return { t.data[i], t.label[i] } 
         end
      }
   );
   
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
