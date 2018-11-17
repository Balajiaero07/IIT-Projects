
n1 = 2;
n2 = 2;
p1 = 0.1;
p2 = 0.87;

t  = 0;

for d=1:n1+1
    z = d-1;
    for x=1:n1+1
        k = binopdf(x+z-1,n1,p1)*binopdf(x-1,n2,p2);
        t = t+k;
    end
end

for z=1:n2
    for x=1:n2+1
        k = binopdf(x-1,n1,p1)*binopdf(z+x-1,n2,p2);
        t = t+k;
    end
end



x2 = unifrnd(0,1,[100,1]);
x1 = unifrnd(0,x2);
scatter(x2,x1,'+')

n =5;

b =0;
for l=1:length(x1)
    t=0;
   for d=1:n+1
       z = d-1;
        for x=1:n+1
            k = binopdf(x+z-1,n,x2(l))*binopdf(x-1,n,x1(l));
            t = t+k;
        end
   end
   for z=1:n
    for x=1:n+1
        k = binopdf(x-1,n,x2(l))*binopdf(z+x-1,n,x1(l));
        t = t+k;
    end
   end
   b = b+t;
end


A = zeros(n1+1,2)
  for x=1:min(n1,n2)+1
     A(x,:) = [x+z-1,x-1]
  end



n1 = 5;
n2 = 5;
p1 = 0.25;
p2 = 0.26;

t  = 0;





for d=1:n1+1
    z = d-1;
  for x=1:n1+1
     k = binopdf(x+z-1,n1,p1)*binopdf(x-1,n2,p2);
     t = t+k;
  end
end

t

for z=1:n2
    for x=1:n2+1
        k = binopdf(x-1,n1,p1)*binopdf(z+x-1,n2,p2);
        t = t+k;
    end
end




n1 = 5;
n2 = 5;
p1 = 0.99;
p2 = 0.01;

t  = 0;
z = 5;

for x=1:n1+1
   k = binopdf(x+z-1,n1,p1)*binopdf(x-1,n2,p2);
   t = t+k;
end
t

t  = 0;
z = 5;

for x=1:n1+1
   k = binopdf(x+z-1,n1,p1)*binopdf(x-1,n2,p2);
   t = t+k;
end
t

clearvars
p1= 0.8;
p2 =0.5;
n = 3;
g=zeros(1,1);
c=zeros(1,2);
for d=1:n+1
     z = d-1;
     for x=1:n+1
         A =[x+z-1,x-1];
         c = [c;A];
         k = binopdf(x+z-1,n,p1)*binopdf(x-1,n,p2);
         g = vertcat(g,k);
     end
end
 

n = 4;
c=zeros(1,2);
g=zeros(1,1);
for d=1:n+1
     z = d-1;
     for x=1:z+1
         A = [z,x-1];
         c = [c;A];
         k = binopdf(z,n,p1)*binopdf(x-1,n,p2);
         g = vertcat(g,k);
     end
end


clearvars X
x2 = unifrnd(0,1);
x1 = unifrnd(0,x2,[floor(x2*10),1]);
dum = repmat(x2,[floor(x2*10),1]);
X = [dum,x1];

for i=1:1000
    x2 = unifrnd(0,1);
    x1 = unifrnd(0,x2,[floor(x2*10),1]);
    dum = repmat(x2,[floor(x2*10),1]);
    X = [X;[dum,x1]];
end
scatter(X(:,1),X(:,2))

clearvars

X_2 = [0.8,0.4];
X_1 = [0.7,0.3];
for l=1:length(X_2)
    for n=3:100
        t=0;
        for d=1:n+1
           z = d-1;
           for x=1:z+1
               k = binopdf(z,n,X_2(l))*binopdf(x-1,n,X_1(l));
               t = t+k;
           end
        end
      cum_prob(l,n-2) = t;  
    end
end
plot(cum_prob')
legend('0.8-0.7','0.4-0.3')

legend('0.1','0.05','0.1','0.3','0.4','0.5','0.6')

 
clearvars
t=0;n=5;
P_2 =0.687;
P_1=0.686;
for d=1:n+1
  z = d-1;
  for x=1:z+1
    k = binopdf(z,n,P_2)*binopdf(x-1,n,P_1);
    t = t+k;
  end
end



clearvars
t=0;
n=50;
P_2 =0.9;
P_1=0.88;
A = zeros(1,2);
for z=1:n
  for x=1:n+1
    f = [x+z-1,x-1];
    A = [A;f];
    k = binopdf(x+z-1,n,P_2)*binopdf(x-1,n,P_1);
    t = t+k;
  end
end



clearvars
t=0;
n=50;
P_2 =0.9;
P_1=0.88;
A = zeros(1,2);
for y=1:n
     for x=1:y
      f = [y,x-1];
      A = [A;f];
      k = binopdf(y,n,P_2)*binopdf(x-1,n,P_1);
      t = t+k;
    end
end
