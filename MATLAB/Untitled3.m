num =100000;
x2 = unifrnd(0,1,[num,1]);
x1 = unifrnd(0,1,[num,1]);
X2 = max(x2,x1);
X1 = unifrnd(0,X2);
data1 = [X2,X1];

hist3(data1,[50 50]);
xlabel('X2')
ylabel('X1')


mu = [0,0]; %// data
sigma = [.5 0; 0 .5]; %// data
x = -5:.1:5; %// x axis
y = -4:.1:4; %// y axis

[X Y] = meshgrid(x,y); %// all combinations of x, y
Z = mvnpdf([X(:) Y(:)],mu,sigma); %// compute Gaussian pdf
Z = reshape(Z,size(X)); %// put into same size as X, Y
%// contour(X,Y,Z), axis equal  %// contour plot; set same scale for x and y...
surf(X,Y,Z) %// ... or 3D plot


str='oil.fig';
imread(str)
num =5;
x2 = unifrnd(0,1,[num,num]);
x1 = unifrnd(0,1,[num,1]);
X2 = max(x2,x1);
X1 = unifrnd(0,X2);
[X, Y] = meshgrid(X2,X1); %// all combinations of x, y
Z = repmat(ones(1)*2,[size(X,1),size(Y,1)]);

surf(X,Y,Z) %// ... or 3D plot



num =500;
x2 = unifrnd(0,1,[num,num]);
x1 = unifrnd(0,x2);
Z = repmat(ones(1)*2,[num,num]);
mesh(x2,x1,Z)

xlabel('X2')
ylabel('X1')
zlabel('f(x1,x2)')


for j=1:5000
    i = randi([0,num-1])+1;
    j = randi([0,num-1])+1;
    x1(i,j) = 0;
    Z(i,j) = 0;
end
for j=1:5000
    i = randi([0,num-1])+1;
    j = randi([0,num-1])+1;
    x1(i,j) = 0;
    Z(i,j) = 2;
end

for j=1:5000
    i = randi([0,num-1])+1;
    j = randi([0,num-1])+1;
    x2(i,j) = 1;
    Z(i,j) = 2;
end



num =100;
x2 = round(unifrnd(0,1,[num,num]),1);
x1 = round(unifrnd(0,1,[num,num]),1);
z = x2-x1>0;
Z = z*2;
mesh(x2,x1,Z)




num =500;
x2 = unifrnd(0,1,[num,num]);
x1 = unifrnd(0,x2);
Z = repmat(ones(1)*2,[num,num]);
stem3(x2,x1,Z)
xlabel('X2')
ylabel('X1')
zlabel('f(x1,x2)')


for j=1:10
   i = randi([0,num-1])+1;
   x2(i) = 1;
   x1(i) = 0;
end
for j=1:10
   i = randi([0,num-1])+1;
   x2(i) = 1;
end
for j=1:10
   i = randi([0,num-1])+1;
   x1(i) = 0;
end

