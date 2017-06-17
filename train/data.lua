--[[
Data Program for Crepe
By Xiang Zhang @ New York University
--]]

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

   self.length = config.length or 1014
   self.batch_size = config.batch_size or 128
   self.file = config.file

   self.config = config
   self.data = torch.load(self.file)

end

function Data:nClasses()
   return #self.data.index
end

function Data:getBatch(inputs, labels, data)
   local data = data or self.data
   local inputs = inputs or torch.Tensor(self.batch_size, #self.alphabet, self.length)
   local labels = labels or torch.Tensor(inputs:size(1))

   for i = 1, inputs:size(1) do
      local label, s
      -- Choose data
      label = torch.random(#data.index)
      local input = torch.random(data.index[label]:size(1))
      s = ffi.string(torch.data(data.content:narrow(1, data.index[label][input][data.index[label][input]:size(1)], 1)))
      for l = data.index[label][input]:size(1) - 1, 1, -1 do
	 s = s.." "..ffi.string(torch.data(data.content:narrow(1, data.index[label][input][l], 1)))
      end
      labels[i] = label
      -- Quantize the string
      self:stringToTensor(s, self.length, inputs:select(1, i))
   end

   return inputs, labels
end

return Data
