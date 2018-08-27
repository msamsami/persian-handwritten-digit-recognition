function centroids = computeCentroids(X, idx, K)

   [m n] = size(X);

   centroids = zeros(K, n);

   for k = 1:K
      centroids(k, :) = mean(X(idx == k, :));
   end

end

