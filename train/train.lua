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
   self.dataset = self.dataset:cuda()
   
   print("Ready to train...")
end

function Train:formatData(data)
   
   local formatedData = {}
   for class = 1, #data.data.index do
      print('Class #:'..class);
      print('# of data in class: '..data.data.index[class]:size(1));
      
      for dataID = 1, data.data.index[class]:size(1) do
       
         local dataTensor = self:toTensor(
            ffi.string(
               torch.data(
                  data.data.content:narrow(
                     1, data.data.index[class][dataID][( data.data.index[class][dataID]:size(1) )], 1
                  )
               )
            ):lower()
         , 1014);
         
         table.insert(formatedData, {dataString, class});
      end
   end
   
   return formatedData
   
   
   --print('---')
   --local tensor = torch.Tensor(#data.alphabet, 1014)
   --tensor:zero()
   --for i = #dataString, math.max(#dataString - 1014 + 1, 1), -1 do
    --  print('I:'..i)
    --  if data.dict[dataString:sub(i,i)] then
    --     tensor[data.dict[dataString:sub(i,i)]][#dataString - i + 1] = 1
     -- end
   --end
   --^^ Works backwards on string lenth resulting in backwards text, and padding at the end.
   --^^ I don't think this should matter as character placement in words is a human concept, not a computer one.
   
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
