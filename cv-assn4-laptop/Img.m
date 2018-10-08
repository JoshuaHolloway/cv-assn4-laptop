classdef Img
    %UNTITLED4 img class to be used to store feature information pertaining
    % to each image for cv-assn-4
    
    properties
        Property1
        corners_provided
        corners_matlab
        corners_custom_x
        corners_custom_y
        
        
        desc % Uses provided feature and matlab computed descriptor
        desc_provided
        desc_matlab
        
        valid_points_matlab
        
        % Match points with the current and previous Img objects
        % e.g. img(2) is matched with img(1), etc.
        matchedPoints_matlab
        
        Value % Number of objects in array of Img objects
        pixels % RGB-image data
        pixels_gray
        pixels_warped
        
        T % data-structure that stores the homography - projective2d object
        T2 % From x->X (I think... maybe the other way around)
        
        matchedPoints_matlab_current_random
        matchedPoints_matlab_prev_random
        
        matched_indices % Between current Img object and previous one: img(i-1) and img(i)
    end
    
    methods
%         function obj = img(inputArg1,inputArg2)
%             %UNTITLED4 Construct an instance of this class
%             %   Detailed explanation goes here
%             obj.Property1 = inputArg1 + inputArg2;
%         end
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
%         function obj = Img(corners)
%             % Input arguement corresponds to the provided harris corners
%             obj.corners_provided = corners;
%         end
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
%         function obj = Img(number, corners)
% 
%                 %obj(number) = obj;
%                 obj(number).corners_provided = corners;
%         end
          function obj = Img(v)
             if nargin > 0
                obj.Value = v;
             end
          end
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        function outputArg = method1(obj,inputArg)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            outputArg = obj.Property1 + inputArg;
        end
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        function detect_harris(obj,inputArg)
            debug = inputArg;
            obj.pixels_gray = rgb2gray(obj.pixels);
            obj.corners_matlab = detectHarrisFeatures(obj.pixels_gray);
            figure, imshow(obj.pixels), hold on;
            plot(obj.corners_matlab.selectStrongest(50)); 
        end
    end
end

