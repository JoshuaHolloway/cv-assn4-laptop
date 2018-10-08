% clc, clear, close all;
% Processing between image 2 and 3

% Instantiate Img object
Img img(3);

% Read images
img(1).pixels = imread('keble_a.jpg');
img(2).pixels = imread('keble_b.jpg');

% Grayscale
img(1).pixels_gray = rgb2gray(img(1).pixels);
img(2).pixels_gray = rgb2gray(img(2).pixels);

% STEP 1: Find features
load('corners.mat'); % Provided corners
img(1).corners_provided = corners_left
img(2).corners_provided = corners_center

% Look at corners:
figure(1)
for i = 1:2
    subplot(2,1,i), imshow(img(i).pixels), hold on;
    draw_corners(img(i).pixels, img(i).corners_provided);
end

% STEP 2: Extract descriptors:
[img(1).desc, img(1).valid_points] = extractFeatures(img(1).pixels_gray, img(1).corners_provided);
[img(2).desc, img(2).valid_points] = extractFeatures(img(2).pixels_gray, img(2).corners_provided);

% STEP 3: Match features:
i=2
indexPairs     = matchFeatures( img(i-1).desc, img(i).desc, 'MaxRatio', 0.23);
img(i).matchedPoints_current = img(i).valid_points(  indexPairs(:,2),:);
img(i).matchedPoints_prev    = img(i-1).valid_points(indexPairs(:,1),:);

% Look at matched features
figure,
title('matlab features')
showMatchedFeatures(...
    img(1).pixels_gray, img(2).pixels_gray,...
    img(2).matchedPoints_prev,...
    img(2).matchedPoints_current,'montage');

% Randomly select 4 matches and draw them:
num_random_points = 4;
X = zeros(num_random_points,1);
x = zeros(num_random_points,1);

num_valid_points = 4;
X = zeros(num_valid_points,2);
x = zeros(num_valid_points,2);

% PROBLEM HERE IS THERE ARE REPEATED POINTS CHOSEN
for i = 1:num_valid_points
    rand_index = ceil(rand * (size(indexPairs,1)-i)); % Decrement possible index to choose from each time you remove one row
    
    img(2).matchedPoints_current = img(2).valid_points( indexPairs(rand_index,2), :);
    img(2).matchedPoints_prev    = img(1).valid_points( indexPairs(rand_index,1), :);
    
    % X->x or x->X ??? -- PRETTY SURE SWITCHED HERE FROM version for 2<->3
    x(i,:) = img(2).matchedPoints_current
    X(i,:) = img(2).matchedPoints_prev
    
    % Need to remove the chosen points here to have unique point correspondences
    % Need to remove the chosen points here to have unique point correspondences
    % Need to remove the chosen points here to have unique point correspondences
    % Need to remove the chosen points here to have unique point correspondences
    indexPairs(rand_index,:) = [];
end
figure,
showMatchedFeatures(...
    img(1).pixels_gray, img(2).pixels_gray,...
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