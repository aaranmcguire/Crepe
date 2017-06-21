-- The namespace
config = {}

local alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'"/\\|_@#$%^&*~`+-=<>()[]{}'

-- Training data
config.train_data = {}
config.train_data.file = "/data/train.t7b"
config.train_data.alphabet = alphabet
config.train_data.length = 1024
config.train_data.limitDataSetSize = 1000

-- Validation data
config.val_data = {}
config.val_data.file = "/data/test.t7b"
config.val_data.alphabet = alphabet
config.val_data.length = 1024

-- Main program
config.main = {}
config.main.device = 1
