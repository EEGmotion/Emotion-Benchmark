local utils = {}

function string:split(sep)

        local sep, fields = sep, {}
        local pattern = string.format("([^%s]+)", sep)
        self:gsub(pattern, function(substr) fields[#fields + 1] = substr end)
        return fields
end


function utils.getLabels(pathToFile,nInstances, col, n_classes)

        local v = torch.Tensor(nInstances)
        local i = 1
        for line in io.lines(pathToFile) do
            local l = line:split(',')
            for key, val in ipairs(l) do
                if key == col then
                    v[i] = val
                   i=i+1
                end
            end
        end

        --v = torch.floor(v)
        --print(v)

        class_i = {}
        class_f = {}

        local min = torch.min(v)
        local max = torch.max(v)

        --print(min.." "..max)
        --print("-----------")

        local step = (max-min)/n_classes

        local inicio = min
        local final  = min + step

        class_i[1] = inicio
        class_f[1] = final

        for i =2, n_classes do
            inicio =  inicio + step
            final  = final + step
            class_i[i] = inicio
            class_f[i] = final
        end

        local cm = torch.Tensor(nInstances,n_classes):zero()

        --print(class_i)
        --print(class_f)
        --print("-----------")

        for j = 1,  nInstances do
            for k = 1, n_classes do
                if  class_i[k] < v[j] and v[j] <= class_f[k] then
                    cm[j][k] = 1
                end
            end
        end

        --print(cm)
        return cm

end

function utils.loadUserData(path,n_trials,n_channels,channel_length)

	local rows = channel_length
        local cols = n_trials * n_channels
        local max_channels = 40 -- hack lol

        local data = torch.Tensor(cols, 1,rows,1)

        local i = 1
        for line in io.lines(path) do
                if i <= rows then
                        local l = line:split(' ')

                        local inicio = 1
                        local fin    = n_channels

                        for c = 1, n_trials do
                            for key, val in ipairs(l) do
                                if key <= fin and key >= inicio then
                                    data[key - (max_channels - n_channels)*(c-1)][1][i][1] = val
                                end  -- if

                            end  -- key, val
                            inicio = inicio + max_channels
                            fin    = fin + max_channels
                        end -- c
                i = i + 1
                end -- i
        end -- line

        local d = torch.Tensor(data)
        local trials  = d:chunk(n_trials,1)

	return trials
end


function utils.getRealPath(generic_path,user_id)
 if user_id <10 then
   path = generic_path .."0"..user_id..".txt"
  else
   path = generic_path..user_id..".txt"
  end

 return path
end


function utils.save()

	i = 1
	j = 2
	e = 3.215

	te = io.open('train.asc', 'w')
	--ve = torch.DiskFile('val.asc', 'w')

	for k = 1, 10 do
		te:write(i..","..j..","..e.."\n")
		i = i +1
		j = j +1
		e = e  + 1
	end
	te:close()
end

return utils
