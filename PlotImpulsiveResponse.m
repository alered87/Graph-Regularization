function  G = PlotImpulsiveResponse(solutions,start_point,end_point)
% PlotImpulsiveResponse : calculating and plot the Impulsive Response 
%                         (Green's Function) of a linear differential
%                         equations with constant coefficients
%
%     G = PlotImpulsiveResponse(solutions,start_point,end_point)
%
%     solutions: solutions of the characteristic poly of a linear 
%                differential equations with constant coefficients
%     start_point: starting point for the plot
%     end_point: ending point for the plot
%
% Author: Alessandro Rossi (2016)
%         rossi111@unisi.it

syms s t ;

F = 1/prod(s-solutions);
G = ilaplace(F,s,t);

ezplot(G,[start_point,end_point]);
title('Impulsive Response','FontSize',14);
xlabel('t')
