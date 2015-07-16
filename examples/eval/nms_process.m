model_name_lst = {'RESULTS_PATH'}

for m = 1:size(model_name_lst,2)
    fprintf('%d',m);
    model_name = model_name_lst{m};
    root = fullfile('./MAT_RESULTS_FOLDER/',model_name,'/');
    root_res = fullfile('./NMS_RESULTS_FOLDER/',model_name,'/');
    mkdir(root_res)

    files = dir(root);
    files = files(3:end,:);
    filenames = cell(1,size(files, 1));
    res_names = cell(1,size(files, 1));
    for i = 1:size(files, 1),
        filenames{i} = files(i).name;
        res_names{i} = [files(i).name(1:end-4), '.png'];
    end

    for i = 1:size(filenames,2),
        matObj = matfile([root,filenames{i}]);
        varlist = who(matObj);
        x = matObj.(char(varlist));
        E=convTri(single(x),1);
        [Ox,Oy]=gradient2(convTri(E,4));
        [Oxx,~]=gradient2(Ox); [Oxy,Oyy]=gradient2(Oy);
        O=mod(atan(Oyy.*sign(-Oxy)./(Oxx+1e-5)),pi);
        E=edgesNmsMex(E,O,1,5,1.01,4);
        imwrite(uint8(E*255),[root_res, res_names{i}])
    end

end
