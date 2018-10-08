clc, clear, close all;
%% Overview
% In this example, feature based techniques are used to
% automatically stitch together a set of images. The procedure for image
% stitching is an extension of feature based image registration. Instead of
% registering a single pair of images, multiple image pairs are
% successively registered relative to each other to form a panorama.

%% Step 1 - Load Images
image_set = imageDatastore({'keble_b.jpg','keble_a.jpg','keble_c.jpg'});

% Display images to be stitched
montage(image_set.Files)

%% Step 2 - Register Image Pairs 
% To create the panorama, start by registering successive image pairs using
% the following procedure:
%
% # Detect and match features between $I(n)$ and $I(n-1)$.
% # Estimate the geometric transformation, $T(n)$, that maps $I(n)$ to $I(n-1)$.
% # Compute the transformation that maps $I(n)$ into the panorama image as $T(n) * T(n-1) * ... * T(1)$.
% I = readimage(image_set, 1);
I  = readimage(image_set,1);

% Initialize features for I(1)
grayImage = rgb2gray(I);
points = detectSURFFeatures(grayImage);
[features, points] = extractFeatures(grayImage, points);
 
    % DEBUG - look at strongest features
    img_josh = imread('keble_a.jpg');
    
    figure(2), 
    subplot(1,3,1),
    current_img = readimage(image_set, 1);
    imshow(current_img), hold on;
    plot(selectStrongest(points, 50));
    
    % QUESTION: After you have the strongest in both images
    % how do you actually know which ones are the strongest matches
    
    % QUESTION: Is it actually the 1st image that corresponds to these
    % features?





% Initialize all the transforms to the identity matrix. Note that the
% projective transform is used here because the building images are fairly
% close to the camera. Had the scene been captured from a further distance,
% an affine transform would suffice.
numImages = numel(image_set.Files);
tforms(numImages) = projective2d(eye(3));

% Initialize variable to hold image sizes.
imageSize = zeros(numImages,2);

% Iterate over remaining image pairs
for n = 2:numImages
    
    % Store points and features for I(n-1).
    pointsPrevious = points;
    featuresPrevious = features;
        
    % Read I(n).
    current_img = readimage(image_set, n);
    grayImage = rgb2gray(current_img);    
    imageSize(n,:) = size(grayImage); % Save image size  
    
    %% JOSH STEP 1: FIND FEATURES AND COMPUTE CORRESPONDING DESCRIPTORS
    % Detect and extract SURF features for I(n).
    points = detectSURFFeatures(grayImage);    
    points_size = size(points)
    features = extractFeatures(grayImage, points);
    vaslid_points_size = size(points)
    % JOSH:
    % I think 'features' is actually descriptors
    % And 'points' is the coordinates of the descriptor
  
    % DEBUG - look at strongest features
    figure(2),
    subplot(1,3,n),
    imshow(current_img), hold on;
    plot(selectStrongest(points, 50));
       
    %% JOSH STEP 2: MATCH FEATURES
    % Find correspondences between I(n) and I(n-1).
    indexPairs = matchFeatures(features, featuresPrevious, 'Unique', true);
    matchedPoints = points(indexPairs(:,1));
    matchedPointsPrev = pointsPrevious(indexPairs(:,2));
    size_points = points
    debug = indexPairs(:,2)

    % DEBUG: Visualize the point correspondances
    figure(3),
    subplot(1,2,n-1),
    showMatchedFeatures(...
        readimage(image_set, n-1), readimage(image_set, n),...
        matchedPoints, matchedPointsPrev, 'montage');
    title('matched features');
    
    % DEBUG: Randomly select one match and display it
    size(indexPairs)
    ind = ceil(rand * size(indexPairs,1)) % Randomly choose one index from indexPairs
    mrow = indexPairs(ind,:) % Index into indexPairs from index chosen in prev line
    idx_1 = mrow(1)
    idx_2 = mrow(2)
    % Index into points with the index [idx_1] for one match 
    %   and [idx_2] for second match
    matchedPoint_rand_index = points(idx_1);
    matchedPoint_prev_rand_index = pointsPrevious(idx_2);
    showMatchedFeatures(...
        readimage(image_set, n-1), readimage(image_set, n),...
        matchedPoint_rand_index, matchedPoint_prev_rand_index, 'montage');

    % What is the difference between points in this program and provided
    % points in mine?
    %   -points is a 921x1 set of SURF-objects
    %       -location is a member of each point object with 2D-coordinates 
    %   -corners_left is a 200x2 matrix containing 2D-coordinates of features
    
    % Why does the match function provide indices outside of range of [0,200]?
    
    
    %% JOSH STEP 3: GEOMETRIC TRANSFORMATION
    % Estimate the transformation between I(n) and I(n-1).
    tform = estimateGeometricTransform(matchedPoints, matchedPointsPrev,...
        'projective', 'Confidence', 99.9, 'MaxNumTrials', 2000);
    tforms(n) = tform;
    
    % Compute T(n) * T(n-1) * ... * T(1)
    tforms(n).T = tforms(n).T * tforms(n-1).T; 
end

%% Step 3 - Initialize the Panorama
% Now, create an initial, empty, panorama into which all the images are
% mapped. 
% 
% Use the |outputLimits| method to compute the minimum and maximum output
% limits over all transformations. These values are used to automatically
% compute the size of the panorama.
for i = 1:numel(tforms)           
    [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1 imageSize(i,2)], [1 imageSize(i,1)]);
end

maxImageSize = max(imageSize);

% Find the minimum and maximum output limits 
xMin = min([1; xlim(:)]);
xMax = max([maxImageSize(2); xlim(:)]);

yMin = min([1; ylim(:)]);
yMax = max([maxImageSize(1); ylim(:)]);

% Width and height of panorama.
width  = round(xMax - xMin);
height = round(yMax - yMin);

% Initialize the "empty" panorama.
panorama = zeros([height width 3], 'like', I);

%% Step 4 - Create the Panorama
% Use |imwarp| to map images into the panorama and use
% |vision.AlphaBlender| to overlay the images together.
blender = vision.AlphaBlender('Operation', 'Binary mask', ...
    'MaskSource', 'Input port');  

% Create a 2-D spatial reference object defining the size of the panorama.
xLimits = [xMin xMax];
yLimits = [yMin yMax];
panoramaView = imref2d([height width], xLimits, yLimits);

% Create the panorama.
for i = 1:numImages
    
    I = readimage(image_set, i);   
   
    % Transform I into the panorama.
    warpedImage = imwarp(I, tforms(i), 'OutputView', panoramaView);
                  
    % Generate a binary mask.    
    mask = imwarp(true(size(I,1),size(I,2)), tforms(i), 'OutputView', panoramaView);
    
    % Overlay the warpedImage onto the panorama.
    panorama = step(blender, panorama, warpedImage, mask);
end

figure
imshow(panorama)