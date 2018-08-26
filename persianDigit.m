clc;
clear;
cd 'D:\University Files\Projects\Persian Handwritten Digit Recognition';

%% Reading .cdb files into MATLAB

SaveImages = false;
filename = 'data set\Test 20000.cdb';

fid = fopen(filename, 'rb');
yy = fread(fid, 1, 'uint16');
m = fread(fid, 1, 'uint8');
d = fread(fid, 1, 'uint8');
W = fread(fid, 1, 'uint8');
H = fread(fid, 1, 'uint8');
TotalRec = fread(fid, 1, 'uint32');
LetterCount = fread(fid, 128, 'uint32');
imgType = fread(fid, 1, 'uint8');
Comments = fread(fid, 256, 'int8');
Reserved = fread(fid, 245, 'uint8');

if( (W > 0) && (H > 0))
    normal = true;
else
    normal = false;
end

Data = cell(TotalRec,1);
labels = zeros(TotalRec,1);

for i = 1:TotalRec
    StartByte = fread(fid, 1);
    labels(i) = fread(fid, 1);
    if (~normal)
        W = fread(fid, 1);
        H = fread(fid, 1);
    end
    ByteCount = fread(fid, 1, 'uint16');
    Data{i} = uint8(zeros(H, W));

    if(imgType == 0)
        for y = 1:H
            bWhite = true;
            counter = 0;
            while (counter < W)
                WBcount = fread(fid, 1);
                x = 1;
                while(x <= WBcount)
                    if(bWhite)
                        Data{i}(y, x+counter) = 0; 
                    else
                        Data{i}(y, x+counter) = 255; 
                    end
                    x = x+1;
                end
                bWhite = ~bWhite;
                counter = counter + WBcount;
            end
        end
    else
        Data{i} = transpose(reshape(uint8(fread(fid, W*H)), W, H));
    end
  
end
fclose(fid);
disp ('Done');

%% Normalizing the size of images
X = {};
Xtest = {};
for i = 1:size(TrainData, 1)    
    X{i, 1} = imresize(TrainData{i}, [32 32]);
end

for i = 1:size(TestData, 1)    
    Xtest{i, 1} = imresize(TestData{i}, [32 32]);
end

disp ('Done');

%% Extracting zoning features for training set
zX = [];

for i = 1:size(X, 1)
    cnt = 0;
    im = X{i};
    for j = 1:4:29
        for k = 1:4:29
            imSeg = im(j:j+3, k:k+3);
            zX(i, cnt+1) = sum(sum(imSeg>1));
            cnt = cnt + 1;
        end
    end
end

disp ('Done');

%% Extracting zoning features for test set

zXtest = [];

for i = 1:size(Xtest, 1)
    cnt = 0;
    im = Xtest{i};
    for j = 1:4:29
        for k = 1:4:29
            imSeg = im(j:j+3, k:k+3);
            zXtest(i, cnt+1) = sum(sum(imSeg>1));
            cnt = cnt + 1;
        end
    end
end

disp ('Done');

%% Sorting the zoning features of training set
zsX = [];
sY = [];

for i = 0:9
    for j = 1:size(Y, 1)
        if Y(j) == i
            zsX = [zsX; zX(j, :)];
            sY = [sY; i];
        end
    end
end

disp ('Done');

%% K-Means clustering for training set

centers = [];
K = 60;
max_iters = 10;

for i = 0:9
    initial_centroids = kMeansInitCentroids(zsX(i*6000 + 1:i*6000 + 6000, :), K);
    [centroids, ~] = runkMeans(zsX(i*6000 + 1:i*6000 + 6000, :), ...
                                 initial_centroids, max_iters, false);    
    centers = [centers; centroids];
end

kcY = [];
for i = 1:10
    for j = 1:K
    kcY = [kcY, i];
    end
end

disp ('Done');

%% Training probabilistic neural network (PNN)

centersTest = [];
for i = 1:size(zXtest, 1)
    % idx = findClosestCentroids(zXtest(i, :), centers);
    centersTest = [centersTest; centers(findClosestCentroids(zXtest(i, :), centers), :)];
end

T = ind2vec(kcY);
net = newpnn(centers', T, 6);
Ysim = sim(net, centersTest');
YsimRes = vec2ind(Ysim);
(size(YsimRes, 2) - sum(YsimRes' ~= (Ytest+1)))/size(YsimRes, 2)
disp ('Done');

%% Testing a single image

img = imresize(rgb2gray(imread('data set\test-3.jpg')), [32 32]);
imshow(img, []);

zimg = [];
for j = 1:4:29
    for k = 1:4:29
        imSeg = img(j:j+3, k:k+3);
        zimg = [zimg, sum(sum(imSeg>1))];
    end
end

idx = findClosestCentroids(zimg, centers);
Yres = vec2ind(sim(net, centers(idx, :)')) - 1

%% Using particle swarm optimization (PSO) to determine optimum number of clusters in each class

CostFunction = @(K, zsX, zXtest, Ytest) PSOfitFunction(K, zsX, zXtest, Ytest);  % Cost function
nVar = 10;            % Number of (unknown) decision variables
VarSize = [1 nVar];   % Size of decision variables matrix

VarMin = 130;         % Lower bound of variables
VarMax = 205;         % Upper bound of variables

% PSO parameters
MaxIt = 50;      % Maximum number of iterations
nPop = 100;        % Number of particles (Swarm Size)

% PSO Parameters
w = 0.99;         % Inertia weight
wdamp = 0.99;     % Inertia weight damping ratio
c1 = 1.9;         % Personal learning coefficient
c2 = 2.1;         % Global learning coefficient

% Velocity limits
VelMax = 0.1*(VarMax - VarMin);
VelMin = -VelMax;

% Initialization
empty_particle.Position = [];
empty_particle.Cost = [];
empty_particle.Velocity = [];
empty_particle.Best.Position = [];
empty_particle.Best.Cost = [];

particle = repmat(empty_particle, nPop, 1);

GlobalBest.Cost = inf;

for i = 1:nPop
    
    particle(i).Position = round(unifrnd(VarMin, VarMax, VarSize));  % Initialize position
    particle(i).Velocity = zeros(VarSize);  % Initialize velocity
    particle(i).Cost = CostFunction(particle(i).Position, zsX, zXtest, Ytest);  % Evaluation
    
    % Update personal best
    particle(i).Best.Position = particle(i).Position;
    particle(i).Best.Cost = particle(i).Cost;
    
    % Update global best
    if particle(i).Best.Cost < GlobalBest.Cost
        GlobalBest = particle(i).Best;
    end
    
end
BestCost = zeros(MaxIt, 1);

% PSO Main Loop
for it = 1:MaxIt
    for i=1:nPop
  
        % Updating velocity
        particle(i).Velocity = w*particle(i).Velocity ...
            +c1*rand(VarSize).*(particle(i).Best.Position-particle(i).Position) ...
            +c2*rand(VarSize).*(GlobalBest.Position-particle(i).Position);
        
        % Applying velocity limits
        particle(i).Velocity = max(particle(i).Velocity, VelMin);
        particle(i).Velocity = min(particle(i).Velocity, VelMax);
        
        % Updating position
        particle(i).Position = particle(i).Position + particle(i).Velocity;
        
        % Velocity mirror effect
        IsOutside = (particle(i).Position<VarMin | particle(i).Position>VarMax);
        particle(i).Velocity(IsOutside) = -particle(i).Velocity(IsOutside);
        
        % Applying position limits
        particle(i).Position = max(particle(i).Position, VarMin);
        particle(i).Position = min(particle(i).Position, VarMax);
        
        % Evaluation
        particle(i).Cost = CostFunction(particle(i).Position, zsX, zXtest, Ytest);
        
        % Updating personal best
        if particle(i).Cost < particle(i).Best.Cost
            particle(i).Best.Position = particle(i).Position;
            particle(i).Best.Cost = particle(i).Cost;
            
            % Updating global best
            if particle(i).Best.Cost < GlobalBest.Cost
                GlobalBest = particle(i).Best;
            end
        end
        
    end
    
    BestCost(it) = GlobalBest.Cost;
    
    disp(['Iteration ' num2str(it) ': Best Cost = ' num2str(BestCost(it))]);
    
    w = w*wdamp;
    
end

BestSol = GlobalBest;
