close all; clear all; clc;
%% Show the original image
img = imread('cameraman.tif');
img = mat2gray(img);         % rescale the image values
figure(); imshow(img,[]);
[row, col] = size(img);
%% Create the blurred image
angle_r = 10/10000;             % change it here
img_b = img;
for i = 1:(10/angle_r)
    img_r = imrotate(img, angle_r*i,'bicubic','crop');   % choose the highest possible quality interpolation; choose the same size
    img_b = img_r + img_b;
end
img_b = mat2gray(img_b);     % Note here to rescale the blurred image for furthur steps
%figure(); imshow(img_b,[]);
%% Conversion from cartesian coordinates to polar coordinates
[theta,r] = meshgrid(-179:180,0:(row/2-1));
[X,Y] = pol2cart(theta/180*pi,r);
X = round(X);
Y = round(Y);
[x,y] = meshgrid((1-row/2):(row/2));
g = interp2(x,y,img_b,X,Y,'cubic');
%figure();imshow(g,[]);
%% In frequency domain
% Compute PSF h and H
h = zeros(1,360);
h(:,1:10) = 1/10;
H = fftshift(fft(h));   
H2 = (abs(H)).^2;
%figure(); plot(abs(H));
% Compute the blurred image G
G = fftshift(fft2(g));
%figure();imshow(log(abs(G)+1));
% Compute wiener filter
Q = (conj(H))./(H2+0.01*ones(1,360));
Q = diag(Q);
% Restore the image (signal F)
F = G*Q;
%figure();imshow(log(abs(F)+1),[]);
%% In imgae domain
f = (abs(ifft2(F)));
%figure();imshow(f,[])
%% Conversion from polar coordinates to Cartesian coordinates
[THETA,R] = cart2pol(x,y);
THETA = round(THETA/pi*180);
R = round(R);
img_prime = interp2(theta,r,f,THETA,R,'cubic');
figure(); imshow(img_prime);
% %% Using Matlab in-built function
% PSF = fspecial('motion',10, 0);
% wnr2 = deconvwnr(g, PSF, 0.01);
% figure, imshow(wnr2)
% img_prime = interp2(theta,r,wnr2,THETA,R,'cubic');
% figure(); imshow(img_prime);
