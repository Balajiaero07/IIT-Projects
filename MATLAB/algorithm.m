Aeq = [.715 .91 0 -1];
beq = 0;
lb = [0 0 0 0.7];
ub = [3.78 2.97 Inf 1.6];
ObjectiveFunction = @simple_fitness;
nvars = 4;    % Number of variables
ConstraintFunction = @simple_constraint;
[x,fval] = ga(ObjectiveFunction,nvars,[],[],Aeq,beq,lb,ub)

[x,fval] = ga(ObjectiveFunction,nvars,[],[],Aeq,beq,lb,ub,ConstraintFunction)


options = optimoptions(@ga,'MutationFcn',@mutationadaptfeasible);
[x,fval] = ga(ObjectiveFunction,nvars,[],[],Aeq,beq,lb,ub,ConstraintFunction,options)


options = optimoptions(options,'PlotFcn',{@gaplotbestf,@gaplotmaxconstr}, 'Display','iter');
[x,fval] = ga(ObjectiveFunction,nvars,[],[],Aeq,beq,lb,ub,ConstraintFunction,options)


X0 = [0 0 0 0]; % Start point (row vector)
options.InitialPopulationMatrix = X0;
% Next we run the GA solver.
[x,fval] = ga(ObjectiveFunction,nvars,[],[],Aeq,beq,lb,ub,ConstraintFunction,options)
