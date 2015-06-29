require 'torch'
require 'nn'
require 'cunn'
require 'sys'
--require 'cutorch'

local filePath  = '/home/ubuntu/EEG_data/s'
local labelPath = '/home/ubuntu/EEG_labels/labels.csv'
local utils     = require('utils')

---------------------------------
-- Parameters
---------------------------------
input_length   = 128
n_feature_maps = 10
n_channels     = 20 -- este puede cambiar
trials         = 40 -- este valor  no deberia cambiar
filter_size    = 5
step           = 2
hidden_units   = 732 --directamente del paper de Zheng
n_classes      = 3
n_users        = 32 -- numero de usuarios disponibles
n_epochs       = 4

gwidth         = trials * n_channels


--------------------------------
-- Model
--------------------------------

mlp        = nn.Sequential()
main_model = nn.Parallel(1,3)

for i = 1,n_channels do
    local model = nn.Sequential()

    model:add(nn.TemporalConvolution(1,n_feature_maps,filter_size))
    model:add(nn.ReLU())
    model:add(nn.SpatialSubSampling(1,1,1,1,2))


    model:add(nn.TemporalConvolution(n_feature_maps,n_feature_maps,filter_size))
    model:add(nn.ReLU())
    model:add(nn.SpatialSubSampling(1,1,1,1,2))

    main_model:add(model)
end

mlp:add(main_model)

x     = ((input_length + (1 - filter_size)*(1 +step))/ (step^2))* n_channels

--mlp:add(nn.Reshape(1,n_feature_maps*x))
mlp:add(nn.View(n_feature_maps*x))
mlp:add(nn.Linear(n_feature_maps*x,hidden_units))
mlp:add(nn.Sigmoid())
mlp:add(nn.Linear(hidden_units,n_classes))
mlp:cuda()

-----------------------------
-- Training
-----------------------------

labels = utils.getLabels(labelPath,40,1,3)

--errors
local te = io.open('train_errorReLU256-10','w')
local tv = io.open('val_errorReLU256-10', 'w')

for e = 1, n_users do
	print("user out "..e)
	--user_out = 1

	for u=1,n_users do
		if u ~= e then
  			print("	user in : "..u)

			inp = utils.loadUserData(utils.getRealPath(filePath,u),trials,n_channels, input_length)

			for i = 1,n_channels do

				y    = labels[i]

				inp2 = torch.Tensor(inp[i]):cuda()
				 y2   = y:cuda()

  				 pred  = mlp:forward(inp2)

  				 criterion = nn.MSECriterion()

  				 local err = criterion:forward(pred:float(),y:float())
				 --te:write(i..","..u..","..e..","..err)
  				 local gradCriterion = criterion:backward(pred:double(),y:double())

  				 mlp:zeroGradParameters()
  				 mlp:backward(inp[1]:cuda(), gradCriterion:cuda());
  				 mlp:updateParameters(0.01);
 			end

			user_out = utils.loadUserData(utils.getRealPath(filePath, e),trials,n_channels, input_length)


			sum_err_channels = 0
			for i = 1, n_channels do
				y = labels[i]
				inp2 = torch.Tensor(user_out[i]):cuda()
        			pred = mlp:forward(inp2)
				-- compara
				criterion = nn.MSECriterion()
                                local err = criterion:forward(pred:float(),y:float())
				sum_err_channels = sum_err_channels + err
			end
			valerr = sum_err_channels/n_channels
			tv:write(u..","..e..","..valerr.."\n")
		 end
        end
end

te:close()
tv:close()
