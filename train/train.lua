local Train = torch.class("Train")

function Train:__init(data, network)
   -- Set vars
   self.model = network:model();
   self.loss = network:loss();

   -- Load Network into GPU
   self.model = self.model:cuda();
   self.loss = self.loss:cuda();
   
   print("Ready to train...")
end

-- Run for a number of steps
-- epoches: number of epoches
-- logfunc: (optional) a function to execute after each step.
function Train:run(epoches,logfunc)
   -- Recapture the weights
   if self.recapture then
      self.params,self.grads = nil,nil
      collectgarbage()
      self.params,self.grads = self.model.sequential:getParameters()
      collectgarbage()
   end
   -- The loop
   for i = 1,epoches do
      self:batchStep()
      if logfunc then logfunc(self,i) end
   end
end

-- Run for one batch step
function Train:batchStep()

   -- Get a batch of data
   self.batch_untyped,self.labels_untyped = self.data:getBatch(self.batch_untyped,self.labels_untyped)
   -- Make the data to correct type
   self.batch = self.batch or self.batch_untyped:transpose(2, 3):contiguous():type(self.model.sequential:type())
   self.labels = self.labels or self.labels_untyped:type(self.model.sequential:type())
   self.batch:copy(self.batch_untyped:transpose(2, 3):contiguous())
   self.labels:copy(self.labels_untyped)


   -- Forward propagation
   self.output = self.model.sequential:forward(self.batch)
   self.objective = self.loss.sequential:forward(self.output,self.labels)
   if type(self.objective) ~= "number" then self.objective = self.objective[1] end
   self.max, self.decision = self.output:double():max(2)
   self.max = self.max:squeeze():double()
   self.mask = self.labels:double():gt(0):double()
   self.decision = self.decision:squeeze():double()
   if self.mask:sum() > 0 then
      self.error = torch.ne(self.decision,self.labels:double()):double():cmul(self.mask):sum()/self.mask:sum()
   else
      self.error = 1
   end
   
   
   -- Backward propagation   
   self.grads:zero()
   self.gradOutput = self.loss:backward(self.output,self.labels)
   self.gradBatch = self.model.sequential:backward(self.batch,self.gradOutput)

   -- Update the step
   self.old_grads:mul(self.momentum):add(self.grads:mul(-self.rate))
   self.params:mul(1-self.rate*self.decay):add(self.old_grads)

   -- Increment on the epoch
   self.epoch = self.epoch + 1
   -- Change the learning rate
   self.rate = self.rates[self.epoch] or self.rate
end
