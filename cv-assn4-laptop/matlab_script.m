clc, clear, close all;
% Processing between image 2 and 3

% Instantiate Img object
Img img(3);

% Read images
img(2).pixels = imread('keble_b.jpg');
img(3).pixels = imread('keble_c.jpg');

% Grayscale
img(2).pixels_gray = rgb2gray(img(2).pixels);
img(3).pixels_gray = rgb2gray(img(3).pixels);

% STEP 1: Find features
load('corners.mat'); % Provided corners
img(2).corners_provided = corners_center
img(3).corners_provided = corners_right

% Look at corners:
figure(1)
for i = 2:3
    subplot(2,1,i-1), imshow(img(i).pixels), hold on;
    draw_corners(img(i).pixels, img(i).corners_provided);
end

% STEP 2: Extract descriptors:
[img(2).desc, img(2).valid_points] = extractFeatures(img(2).pixels_gray, img(2).corners_provided);
[img(3).desc, img(3).valid_points] = extractFeatures(img(3).pixels_gray, img(3).corners_provided);

% Grab provided descriptors:
load('sift_features.mat'); % Provided corners
img(2).desc_provided = sift_center;
img(3).desc_provided = sift_right;

% Pass to C++ - THIS SHOULD BE THE SAME FOR BOTH SETS OF IMAGES
desc_dim_2__to_cpp = size(img(2).desc,2);
desc_dim_3__to_cpp = size(img(3).desc,2);


desc_2 = double(img(2).desc);
desc_3 = double(img(3).desc);

desc_2__to_cpp = desc_2;
desc_3__to_cpp = desc_3;





% SHOULD JUST BE THE PROVIDED CORNERS!!!
% num_valid_points_2__to_cpp = size(img(2).desc,1);
% num_valid_points_3__to_cpp = size(img(3).desc,1);
% valid_points_2 = double(img(2).valid_points);
% valid_points_3 = double(img(3).valid_points);
% valid_points_2__to_cpp = valid_points_2;
% valid_points_3__to_cpp = valid_points_3;
num_valid_points_2__to_cpp = size(corners_center,1); % # of rows are number of corners
num_valid_points_3__to_cpp = size(corners_right,1);
valid_points_2__to_cpp = corners_center;
valid_points_3__to_cpp = corners_right;



% STEP 3: Match features:
i=3
indexPairs     = matchFeatures( img(i).desc, img(i-1).desc, 'MaxRatio', 0.3 );
img(i).matchedPoints_current = img(i).valid_points(  indexPairs(:,1),:);
img(i).matchedPoints_prev    = img(i-1).valid_points(indexPairs(:,2),:);

% Pass number of matches to C++
num_matches__to_cpp = size(indexPairs,1)

% Pass ALL matches to C++
X_full = zeros(num_matches__to_cpp,2);
x_full = zeros(num_matches__to_cpp,2);
for i = 1:num_matches__to_cpp   
    
    xxxx= indexPairs(i,1)
    
    % X->x or x->X ???
    X_full(i,:) = img(3).valid_points( indexPairs(i,1), :);
    x_full(i,:) = img(2).valid_points( indexPairs(i,2), :);
end
% Pass to C++
X2_full__to_cpp = X_full;
x2_full__to_cpp = x_full;



% Look at matched features
figure,
title('matlab features')
showMatchedFeatures(...
    img(2).pixels_gray, img(3).pixels_gray,...
    img(3).matchedPoints_prev,...
    img(3).matchedPoints_current,'montage');

% Randomly select 4 matches and draw them:
num_valid_points = 4;
X = zeros(num_valid_points,2);
x = zeros(num_valid_points,2);

% PROBLEM HERE IS THERE ARE REPEATED POINTS CHOSEN
for i = 1:num_valid_points
    rand_index = ceil(rand * (size(indexPairs,1)-i)); % Decrement possible index to choose from each time you remove one row
    
    img(3).matchedPoints_current = img(3).valid_points( indexPairs(rand_index,1), :);
    img(3).matchedPoints_prev    = img(2).valid_points( indexPairs(rand_index,2), :);
    temp = img(3).matchedPoints_current
    
    % X->x or x->X ???
    X(i,:) = img(3).matchedPoints_current
    x(i,:) = img(3).matchedPoints_prev
    
    % Need to remove the chosen points here to have unique point correspondences
    % Need to remove the chosen points here to have unique point correspondences
    % Need to remove the chosen points here to have unique point correspondences
    % Need to remove the chosen points here to have unique point correspondences
    indexPairs(rand_index,:) = [];
end
figure,
showMatchedFeatures(...
    img(2).pixels_gray, img(3).pixels_gray,...
    x,...       % Previous
    X,'montage'); % Current
title('random matches')



%% Compute the homography from the four random points here

[H] = est_homog_(X, x)
[H2] = est_homog_(x, X)


%% Back-project the points here
i = 1;

X1_ = [X(i,1); X(i,2); 1];
x1_ = H * X1_

% De-homography
x1 = [x1_(1)/x1_(3); x1_(2)/x1_(3)]
x1_gold = x(i,:)


%% Compute re-projection error at this point to determine number of inliers for RANSAC

%% Warp image
img(3).T  = projective2d(H)
img(3).T2 = projective2d(H2)
T = img(3).T
T2 = img(3).T2
T.T
T2.T

img(3).pixels_warped = imwarp(img(3).pixels,img(3).T2);

debug = img(3).pixels_warped;
size_debug = size(debug)
figure,
imshow(img(3).pixels_warped), title('warped image')

% Pass to C++
% Pass to C++
X_to_cpp = X;
x_to_cpp = x;
H_to_cpp = H;


%==========================================================================
% Look at image with corners superimposed on top:
function draw_corners(img, corners)
    % img is a 3D array where the 4th dimension is the exact image
    imshow(img); hold on; title('center figure');
    x_coordinates = corners(:,1);
    y_coordinates = corners(:,2);
    plot(x_coordinates, y_coordinates, 'r*', 'LineWidth', 0.5, 'MarkerSize', 5);
end