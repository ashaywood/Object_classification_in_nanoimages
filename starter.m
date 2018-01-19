% addpath('C:\Program Files\SDC Morphology Toolbox for MATLAB 1.6');
% startup
% addpath('C:\Program Files\SDC Morphology Toolbox for MATLAB 1.6\data');

%Read the original image
%f = mmreadgray('binary_blobs_white_and_black_nosie.bmp');
f2=imread('goldnp4.png');
f3=rgb2gray(f2);
f= im2uint8(f3);

imshow(f);

