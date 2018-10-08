function [ H ]   = est_homog_(X, x) % Is it X->x or x->X???
% est_homography estimates the homography to transform each of the
% X into the x
% Inputs:
%     X: a 4x2 matrix of corner points in the video
%     x: a 4x2 matrix of logo points that correspond to X
% Outputs:
%     H: a 3x3 homography matrix such that x ~ H*X


    h = zeros(9,1);
    A = zeros(8,9);  
    
    X1 = X(1,1);    Y1 = X(1,2);
    X2 = X(2,1);    Y2 = X(2,2);
    X3 = X(3,1);    Y3 = X(3,2);
    X4 = X(4,1);    Y4 = X(4,2);

    x1 = x(1,1);   y1 = x(1,2);
    x2 = x(2,1);   y2 = x(2,2);
    
    x3 = x(3,1);   y3 = x(3,2);
    x4 = x(4,1);   y4 = x(4,2);
    
    A = [
        -X1, -Y1, -1,   0,   0,  0, X1*x1, Y1*x1, x1;
          0,   0,  0, -X1, -Y1, -1, X1*y1, Y1*y1, y1;
        -X2, -Y2, -1,   0,   0,  0, X2*x2, Y2*x2, x2;
          0,   0,  0, -X2, -Y2, -1, X2*y2, Y2*y2, y2;
        -X3, -Y3, -1,   0,   0,  0, X3*x3, Y3*x3, x3;
          0,   0,  0, -X3, -Y3, -1, X3*y3, Y3*y3, y3;
        -X4, -Y4, -1,   0,   0,  0, X4*x4, Y4*x4, x4;
          0,   0,  0, -X4, -Y4, -1, X4*y4, Y4*y4, y4
          ];   

    [U, S, V] = svd(A);

    h = V(:,9);
    H = reshape(h,3,3)'; % MATLAB is col-major - so annoying!
    
    H = H ./ H(3,3);
end