model_name = 'HED';
sub_classes = {'fuse_mat', 'dsn12345_mat'};
root = fullfile('./NMS_RESULTS_FOLDER/',model_name, sub_classes{1});

root_res = fullfile('./NMS_RESULTS_FOLDER/',model_name, 'merge_mat');
mkdir(root_res)
 
files = dir(root);
files = files(3:end,:);
filenames = cell(1,size(files, 1));
res_names = cell(1,size(files, 1));
for i = 1:size(files, 1),
    filenames{i} = files(i).name;
end

for i = 1:size(filenames,2),
  fuse = double(imread(fullfile('./NMS_RESULTS_FOLDER/',model_name, sub_classes{1},filenames{i})));
  dsn12345 = double(imread(fullfile('./NMS_RESULTS_FOLDER/',model_name, sub_classes{2},filenames{i})));

  fuse_12345 = (fuse+dsn12345)/2;
  imwrite(uint8(fuse_12345),fullfile(root_res, filenames{i}))
end
  
  
  
