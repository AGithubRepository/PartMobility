function [R, t] = abspose(P, Q)
% ABSPOSE - Absolute orientation 
%   INPUTS:
%     P - the reference point set arranged as a 3xn matrix
%     Q - the point set obtained by transforming P with some pose estimate (typically the last estimate)
%   OUTPUTS:
%     R - estimated rotation matrix
%     t - estimated translation vector

n = size(P,2);

% compute means of P and Q
pbar = sum(P,2)/n;
qbar = sum(Q,2)/n;

for i = 1:n
  PP(:,i) = P(:,i)-pbar;
  QQ(:,i) = Q(:,i)-qbar;
end

% compute M matrix
M(1:3,1:3) = 0;
for i = 1:n
  M = M+PP(:,i)*QQ(:,i).';
end

% calculate SVD of M
[U,S,V] = svd(M);
C = zeros(3,3);
C(1,1) = 1;
C(2,2) = 1;
C(3,3) = det(V*U.');
% compose the optimal estimate of R
R = V*C*(U.');
t = qbar - R*pbar;
end