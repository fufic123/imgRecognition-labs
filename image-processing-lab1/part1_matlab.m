% Part 1 — Image Processing in Octave/MATLAB

pkg load image;

output_dir = fullfile(fileparts(mfilename('fullpath')), 'output');

% 1. Load image
img_path = fullfile(fileparts(mfilename('fullpath')), 'image.jpg');
img = imread(img_path);

% 2. Convert RGB to grayscale
gray = rgb2gray(img);

% 3. Histogram equalization
equalized = histeq(gray);

% 4. Edge detection
edges = edge(gray, 'canny');

% 5. Gaussian filter
h = fspecial('gaussian', [5 5], 1);
blurred = imfilter(img, h);

% 6. Display all in one window
figure('visible', 'off');

subplot(1, 5, 1); imshow(img);       title('Original');
subplot(1, 5, 2); imshow(gray);      title('Grayscale');
subplot(1, 5, 3); imshow(equalized); title('Equalized');
subplot(1, 5, 4); imshow(edges);     title('Edges');
subplot(1, 5, 5); imshow(blurred);   title('Blurred');

out_path = fullfile(output_dir, 'result_matlab.png');
saveas(gcf, out_path);
printf('Saved -> %s\n', out_path);
