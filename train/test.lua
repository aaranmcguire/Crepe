--[[ Tester for Crepe
By Xiang Zhang @ New York University
--]]

require("sys")

local Test = torch.class("Test")

-- Initialization of the testing script
-- data: Testing dataset
-- model: Testing model
-- loss: Loss used for testing
-- config: (optional) the configuration table
--    .confusion: (optional) whether to use confusion matrix
function Test:__init(data,model,loss,config)
   local config = config or {}

   -- Store the objects
   self.data = data
   self.model = model
   self.loss = loss

   -- Move the type
   self.loss:type(model:type())


   -- Set the confusion
   if config.confusion then
      self.confusion = torch.zeros(data:nClasses(),data:nClasses())
   end

   -- Store configurations
   self.normalize = config.normalize
end

-- Execute testing for a batch step
function Test:run(logfunc)
   -- Initializing the errors and losses
   self.e = 0
   self.l = 0
   self.n = 0
   if self.confusion then self.confusion:zero() end

   -- Start the loop
   for batch,labels,n in self.data:iterator() do
      self.batch = self.batch or batch:transpose(2,3):contiguous():type(self.model:type())
      self.labels = self.labels or labels:type(self.model:type())
      self.batch:copy(batch:transpose(2, 3):contiguous())
      self.labels:copy(labels)


      -- Forward propagation
      self.output = self.model:forward(self.batch)
      self.objective = self.loss:forward(self.output,self.labels)
      if type(self.objective) ~= "number" then self.objective = self.objective[1] end
      self.max, self.decision = self.output:double():max(2)
      self.max = self.max:squeeze():double()
      self.decision = self.decision:squeeze():double()
      self.err = torch.ne(self.decision,self.labels:double()):sum()/self.labels:size(1)
      


      -- Accumulate the errors and losses
      self.e = self.e*(self.n/(self.n+n)) +  self.err*(n/(self.n+n))
      self.l = self.l*(self.n/(self.n+n)) + self.objective*(n/(self.n+n))
      if self.confusion then
	 for i = 1,n do
	    self.confusion[labels[i]][self.decision[i]] = self.confusion[labels[i]][self.decision[i]]+1
	 end
      end
      self.n = self.n + n
      

      -- Call the log function
      if logfunc then logfunc(self) end

   end
   -- Average on the confusion matrix
   if self.confusion and self.n ~= 0 then self.confusion:div(self.n) end
end
