
function z = PSOfitFunction(K, zsX, zXtest, Ytest)

    centers = [];
    max_iters = 10;

    for i = 0:9
        initial_centroids = kMeansInitCentroids(zsX(i*6000 + 1:i*6000 + 6000, :), K(i+1));
        [centroids, ~] = runkMeans(zsX(i*6000 + 1:i*6000 + 6000, :), ...
                                     initial_centroids, max_iters, false);    
        centers = [centers; centroids];
    end

    kcY = [];
    for i = 1:10
        for j = 1:K(i)
        kcY = [kcY, i];
        end
    end

    centersTest = [];
    for i = 1:size(zXtest, 1)
        % idx = findClosestCentroids(zXtest(i, :), centers);
        centersTest = [centersTest; centers(findClosestCentroids(zXtest(i, :), centers), :)];
        clc;
        disp ([num2str(i), ' done']);
    end

    z = sum(vec2ind(sim(newpnn(centers', ind2vec(kcY), 6), centersTest'))' ~= (Ytest+1));

end
