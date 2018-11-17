% Generate some data1.
num =100000;
x2 = unifrnd(0,1,[num,1]);
x1 = unifrnd(0,1,[num,1]);
X2 = max(x2,x1);
X1 = unifrnd(0,X2);

data1 = [X2,X1];

% Get some info.
m = mean(data1);
s = std(data1);
axisMin = [0,0];
axisMax = [1,1];
plot3(data1(:,1), data1(:,2), zeros(size(data1,1),1), 'k.', 'MarkerSize', 1);
hold on
maxP = 0;
for side = 1:2
    % Calculate the histogram.
    p = [0 hist(data1(:,side), 20) 0];
    p = p / sum(p);
    maxP = max([maxP p]);
    dx = (axisMax(side) - axisMin(side)) / numel(p) / 2.3;
    p2 = [zeros(1,numel(p)); p; p; zeros(1,numel(p))]; p2 = p2(:);
    x = linspace(axisMin(side), axisMax(side), numel(p));
    x2 = [x-dx; x-dx; x+dx; x+dx]; x2 = max(min(x2(:), axisMax(side)), axisMin(side));

    % Calculate the curve.
    nPtsCurve = numel(p) * 10;
    xx = linspace(axisMin(side), axisMax(side), nPtsCurve);

    % Plot the curve and the histogram.
    if side == 1
        plot3(xx, ones(1, nPtsCurve) * axisMax(3 - side), spline(x,p,xx), 'r-', 'LineWidth', 2);
        plot3(x2, ones(numel(p2), 1) * axisMax(3 - side), p2, 'k-', 'LineWidth', 1);
    else
        plot3(ones(1, nPtsCurve) * axisMax(3 - side), xx, spline(x,p,xx), 'b-', 'LineWidth', 2);
        plot3(ones(numel(p2), 1) * axisMax(3 - side), x2, p2, 'k-', 'LineWidth', 1);
    end

end
% Turn off hold.
hold off

% Axis labels.
xlabel('X2');
ylabel('X1');
zlabel('');

axis([axisMin(1) axisMax(1) axisMin(2) axisMax(2) 0 maxP * 1.05]);
grid on;