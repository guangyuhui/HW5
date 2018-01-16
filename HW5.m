%---------------------------Importance Sampling----------------------------
clc
clear all
close all

Ne = 100000; % sample size


%---------------------------1,2 -------------------------------------------
% for k = 1:1000
% 
% x = normrnd(2,1,[Ne,1]); % draw samples from proposed distribution N(0,1)
% p = 1/sqrt(2*pi*4)*exp(-x.^2./(2*4)); % target distribution N(0,4)
% q = 1/sqrt(2*pi*1)*exp(-(x-2).^2./(2*1)); % proposal distribution N(0,1)
% 
% E = mean (fun(x)'.*p./q);  % Estimated expectation
% 
% error(k) = abs(0.0227501-E)/ abs(0.0227501); % Error of estimation
% 
% end
% 
% average_error=mean(error)


%----------------------------3---------------------------------------------


eps = 0.1;
for n = 10:10  % vary n from 10 to 1000

R = eye(n);

for k = 1:1    % 100 experiments

x = mvnrnd(zeros(Ne,n),(1+eps)*R); % draw samples from N(0,(1+eps)*I)

for i = 1:Ne
p(i) = 1/sqrt((2*pi)^n*det(R))*exp(-0.5*x(i,:)*inv(R)*x(i,:)'); % target distribution N(0,I)
q(i) = 1/sqrt((2*pi)^n*det((1+eps)*R))*exp(-0.5*x(i,:)*inv((1+eps)*R)*x(i,:)');% proposal distribution N(0,(1+eps)*I)
end

w_tilde = log(q)-log(p);
w_tilde = w_tilde-max(w_tilde);
w = exp(-w_tilde);

w_hat = w/sum(w); % normalized weights

rho(k) = mean(w_hat.^2)/(mean(w_hat))^2;

end

rho_average(n)=mean(rho);

end

plot([10:1000],rho_average(10:1000))





