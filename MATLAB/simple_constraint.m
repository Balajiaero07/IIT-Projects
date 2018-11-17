function [c, ceq] = simple_constraint(x)
   c = [];
   ceq = x(4)*(1 + exp((-.71*x(2))/x(1)) + (1-x(4))^2) - x(3);
end