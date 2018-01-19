fmed = im2uint8(medfilt2(f, [5 5]));
% fgauss=im2uint8(imgaussfilt(fmed,10));

I=fmed;
% Step 2: Use the Gradient Magnitude as the Segmentation Function
hy = fspecial('sobel');
hx = hy';
Iy = imfilter(double(I), hy, 'replicate');
Ix = imfilter(double(I), hx, 'replicate');
gradmag = sqrt(Ix.^2 + Iy.^2);
figure
imshow(gradmag,[]), title('Gradient magnitude (gradmag)')

% Step 3: Mark the Foreground Objects

%Compute the opening
se = strel('disk', 5);
Io = imopen(I, se);
figure
imshow(Io), title('Opening (Io)')

% Compute the opening-by-reconstruction using imerode and imreconstruct.
Ie = imerode(I, se);
Iobr = imreconstruct(Ie, I);
figure
imshow(Iobr), title('Opening-by-reconstruction (Iobr)')

% Following the opening with a closing can remove the dark spots and stem marks. Compare a regular morphological closing with a closing-by-reconstruction. First try imclose:
Ioc = imclose(Io, se);
figure
imshow(Ioc), title('Opening-closing (Ioc)')


% Now use imdilate followed by imreconstruct. Notice you must complement the image inputs and output of imreconstruct.
Iobrd = imdilate(Iobr, se);
Iobrcbr = imreconstruct(imcomplement(Iobrd), imcomplement(Iobr));
Iobrcbr = (Iobrcbr);
figure
imshow(Iobrcbr), title('Opening-closing by reconstruction (Iobrcbr)')

level = graythresh(Iobrcbr);
fgm =  imbinarize(Iobrcbr, level);
figure
imshow(fgm), title('Regional maxima of opening-closing by reconstruction (fgm)')


% Calculate the regional maxima of Iobrcbr to obtain good foreground markers.
% fgm = imregionalmax(Iobrcbr);
% figure
% imshow(fgm), title('Regional maxima of opening-closing by reconstruction (fgm)')

% Superimpose the foreground marker image on the original image.
I2 = I;
I2(fgm) = 255;
figure
imshow(I2), title('Regional maxima superimposed on original image (I2)')

% Notice that some of the mostly-occluded and shadowed objects are not marked, 
% which means that these objects will not be segmented properly in the end result. 
% Also, the foreground markers in some objects go right up to the objects' edge. 
% That means you should clean the edges of the marker blobs and then shrink them a bit. 
% You can do this by a closing followed by an erosion.
se2 = strel(ones(3,3));
fgm2 = imclose(fgm, se2);
fgm3 = imerode(fgm2, se2);

fgm4 = bwareaopen(fgm3, 20);
I3 = I;
I3(fgm4) = 255;
figure
imshow(I3)
title('Modified regional maxima superimposed on original image (fgm4)')

% Step 4: Compute Background Markers
bw = imbinarize(Iobrcbr);
figure
imshow(bw), title('Thresholded opening-closing by reconstruction (bw)')

D = bwdist(bw);
DL = watershed(D);
bgm = DL == 0;
figure
imshow(bgm), title('Watershed ridge lines (bgm)')


% Step 5: Compute the Watershed Transform of the Segmentation Function.
% The function imimposemin can be used to modify an image so that it has regional minima only in certain desired locations. Here you can use imimposemin to modify the gradient magnitude image so that its only regional minima occur at foreground and background marker pixels.
gradmag2 = imimposemin(gradmag, bgm | fgm4);

% Finally we are ready to compute the watershed-based segmentation.
L = watershed(gradmag2);


% Step 6: Visualize the Result
% One visualization technique is to superimpose the foreground markers, background markers, and segmented object boundaries on the original image. You can use dilation as needed to make certain aspects, such as the object boundaries, more visible. Object boundaries are located where L == 0.
I4 = I;
I4(imdilate(L == 0, ones(3, 3)) | bgm | fgm4) = 255;
figure
imshow(I4)
title('Markers and object boundaries superimposed on original image (I4)')

Lrgb = label2rgb(L, 'jet', 'w', 'shuffle');
figure
imshow(Lrgb)
title('Colored watershed label matrix (Lrgb)')


figure
imshow(I)
hold on
himage = imshow(Lrgb);
himage.AlphaData = 0.3;
title('Lrgb superimposed transparently on original image')

flabel=L;

%Find the 20th percentile and remove objects smaller that this
s=regionprops(gradmag2,'basic');
fscale=regionprops(logical(gradmag2),'area');
fscalearray=transpose(double([fscale.Area]));
fquart=quantile(fscalearray,0.05);
idx=find([fscale.Area]>fquart);
f2=((ismember(flabel,idx)));
fcentroid=regionprops(flabel,'centroid');
fcentroid2=cat(1,fcentroid.Centroid);

% Calculate the starting values for the paramters:
    %Scale, shift, rotation and mean intensity
fscale=regionprops(flabel,'area');
feccentric=regionprops(flabel,'eccentricity');
%fextrema=regionprops(flabel,'extrema');
fmajoraxis=regionprops(flabel,'majoraxislength');
fminoraxis=regionprops(flabel,'minoraxislength');
frotation=regionprops(flabel,'orientation');
fmean=regionprops(flabel,f,'MeanIntensity');
fscale2=cat(1,fscale.Area);
frotation2=cat(1,frotation.Orientation);
feccentric2=cat(1,feccentric.Eccentricity);
fmean2=cat(1,fmean.MeanIntensity);
measurements = regionprops(flabel, f, 'PixelValues');

size0=size(fscale,1); %Number of initial objects
% Assign a template to each object based on the eccentricity 
%(E1=1.2 => e=0.6) e=sqrt(1-E2^2/E1^2) ==> E1=1/(1-e^2)^1/4
ftemplate=ones(size0,1);
fvar=ones(size0,1);

for i=1:size0
    fvar(i) = sqrt(var(double(measurements(i).PixelValues)));
    if feccentric2(i)<0.6 
        ftemplate(i)=2;
     % 1's are ellipses and 2's are circles
    end
end
% Obtain initial parameter matrix, columns 2&3 are the x,y coordinates 

%           s       c=(x,y)    theta      T         mu
parameters=[fscale2,fcentroid2,frotation2,ftemplate,fmean2, fvar,feccentric2];









