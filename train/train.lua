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
   
   --local randomClass = torch.random(#data.data.index)
   --^^ Random select of one of the classifications.
   
   --local randomDataFromClass = torch.random(data.data.index[randomClass]:size(1))
   --^^ Random select or one of the data inputs from teh selected Class.
   
   --local dataString = ffi.string(
   --   torch.data(
   --      data.data.content:narrow(
   --         1, data.data.index[randomClass][randomDataFromClass][( data.data.index[randomClass][randomDataFromClass]:size(1) )], 1
   --      )
   --   )
   --):lower();
   --^^  ¯\_(ツ)_/¯ -- No clue what this is doing, but this is the string of the input
   
   local formatedData = {}
   for class = 1, #data.data.index do
      print('Class #:'..class);
      print('# of data in class: ') 
      print(data.data.index[class]:size(1));
      
      
      for dataID = 1, data.data.index[class]:size(1) do
       
         local dataString = ffi.string(
            torch.data(
               data.data.content:narrow(
                  1, data.data.index[class][dataID][( data.data.index[class][dataID]:size(1) )], 1
               )
            )
         ):lower();
         
         formatedData.insert({dataString, class});
      end
   end
   
   return formatedData
   
   
   --local dataString = 'abcdefghijklmnopqrstuvwxyz'
   
   --print(dataString)
   --print('Data Length: '..#dataString)
   --print('Alphabet Length:'..#data.alphabet)
   
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


function Train:run()
   
   
end
