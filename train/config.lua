-- The namespace
config = {}

local alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'"/\\|_@#$%^&*~`+-=<>()[]{}'

-- Training data
config.train_data = {}
config.train_data.file = "/data/realDB.t7b"
config.train_data.alphabet = alphabet
config.train_data.length = 6634
--config.train_data.limitDataSetSize = 200

-- Validation data
config.val_data = {}
config.val_data.file = "/data/realDB.t7b"
config.val_data.alphabet = alphabet
config.val_data.length = 6634

-- Main program
config.main = {}
config.main.device = 1
