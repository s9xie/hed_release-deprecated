model_name_lst = {'RESULTS_PATH'}

for i = 1:size(model_name_lst,2)
    tic;
    resDir = fullfile('./NMS_RESULTS_FOLDER/',model_name_lst{i});
    fprintf('%s\n',resDir);
    gtDir = './BSR_full/BSDS500/data/groundTruth/test';
    edgesEvalDir('resDir',resDir,'gtDir',gtDir, 'thin', 1, 'pDistr',{{'type','parfor'}},'maxDist',0.0075);

    figure; edgesEvalPlot(resDir,'HED');
    toc
end


