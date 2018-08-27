function idx = findClosestCentroids(X, centroids)

   % Set K
   K = size(centroids, 1);

   idx = zeros(size(X,1), 1);

   for i = 1:size(X, 1)
      temp = [];
      for k = 1:K
          temp = [temp; norm(X(i, :) - centroids(k, :))];
      end
      [~, idx(i)] = min(temp);
   end

end

