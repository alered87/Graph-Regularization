function [A, B, theta, order, solutions] = makeSystemMatrix(parameters) 
% makeSystemMatrix :produces the matrices to implement the dynamic updating 
%  of solution as a system of first order linear differential equations 
%
%     [A,B,theta,order,solutions]=makeSystemMatrix([solution1;solution2])
%     [A,B,theta,order,solutions]=makeSystemMatrix([theta;order;memory])
%     [A,B,theta,order,solutions]=makeSystemMatrix([solution1;...;solution4])
%     [A,B,theta,order,solutions]=makeSystemMatrix([theta,alpha0,alpha1])
%     [A,B,theta,order,solutions]=makeSystemMatrix([theta,alpha0,alpha1,alpha2])
%
%     order: order of the system depending on the parameters
%     solutions: 1-by-order vector containing the  differential equation's 
%                solutions (given or calculated)
%     A: order-by-order matrix of the equation's system
%     B: order-by-1 system's vector constant terms
%     theta: dissipation term of the model (given or calculated)
%
% Author: Alessandro Rossi (2016)
%         rossi111@unisi.it

l = length(parameters);

switch l
    case 2 % [solution1;solution2] given
        order = 2;
        solutions = parameters';
        theta = -sum(parameters);
        beta = prod(parameters);
        reversed_poly_coefficients = [-beta,-theta];
    case 3
        type = size(parameters);
        if type(1) == 1 % [theta, alpha0, alpha1] given
            order = 2;
            theta = parameters(1);
            beta = CharacteristicPolyCoefficients(theta, parameters(2:3));
            solutions = roots([1,beta])';
            reversed_poly_coefficients = -fliplr(beta);
        else % [theta;order;memory]
            theta = parameters(1);
            order = parameters(2);
            solutions = SpreadTheta(parameters');
            p = poly(solutions);      
            reversed_poly_coefficients = -fliplr(p(2:5));
        end
    case 4
        type = size(parameters);
        if type(1) == 1 % [theta, alpha0, alpha1, alpha2] given
            order = 4;
            theta = parameters(1);
            beta = CharacteristicPolyCoefficients(theta, parameters(2:4));
            solutions = roots([1,beta])';
            reversed_poly_coefficients = -fliplr(beta);
        else % [solution1;...;solution4] given
            order = 4;
            solutions = parameters(1:4)';
            p = poly(parameters);
            theta = p(2)/2;       
            reversed_poly_coefficients = -fliplr(p(2:5));
        end
    otherwise
        error('Invalid setting of parameters: see makeSystemMatrix Help');
end

B = [zeros(order-1, 1); (-1)^(order/2+1)];
A = [zeros(order-1,1), eye(order-1) ; reversed_poly_coefficients ];
end


function beta = CharacteristicPolyCoefficients(theta, a)
% CharacteristicPolyCoefficients : calculates the coefficients of the
%                                  characteristic polynomial of the linear
%                                  differential equation
%
%     beta = CharacteristicPolyCoefficients(theta, a)
%
%     theta: dissipation term
%     a: vector containing the differential operator coefficients 
% 
%     beta: vector containing the coefficients of the characteristic
%           polynomial of the linear differential equation, from the
%           second-high order one to the constant term

order = length(a);

switch order
    case 2
        beta(1) = theta;
        beta(2) = ( (a(1)*a(2)*theta) - a(1)^2 ) / a(2)^2 ;
        % checking Routh-Hurwitz conditions
        if  theta <= a(1)/a(2) 
            warning('Routh-Hurwitz conditions not satisfied.');             
        end
    case 3
        beta(1) = 2*theta;
        beta(2) = (   a(3)^2*theta^2 + ...
                      a(2)*a(3)*theta + ...
                      2*a(1)*a(3) - a(2)^2   ) / a(3)^2 ;
        beta(3) = (   a(2)*a(3)*theta^2 + ...
                      ( 2*a(1)*a(3) - (a(2)^2) ) * theta    ) / (a(3)^2);
        beta(4) = (   a(1)*a(3)*theta^2 - ...
                      a(1)*a(2)*theta + a(1)^2   ) / (a(3)^2);
        % checking Routh-Hurwitz conditions
        R(1) = beta(2) > 0;
        R(2) = beta(3) > 0;
        R(3) = beta(4) > 0;
        R(4) = beta(1)*beta(2) > beta(3) ;
        R(5) = beta(1)*beta(3)*beta(2) > beta(3)^2+beta(4)*beta(1)^2;
        if  ~all(R) 
            warning('Routh-Hurwitz conditions not satisfied.');             
        end
end
end


function solutions = SpreadTheta(parameters)
%SpreadTheta : calculating the solutions of the model by spreading 'theta'
%              among the solutions, providing the resulting Impulsive 
%              Response with the given 'memory' and promptness
%
%     solutions = SpreadTheta([theta; order; memory])
% 
%     theta: dissipation parameter
%     order: order of the differential equation of the model
%     memory: roughly the desired saturation time of the Impulsive Response 

theta = parameters(1);
order = parameters(2);
memory = parameters(3);

solutions(1) = -1/memory; % memory solution

switch order
    case 2 % theta=sum(solutions)
        solutions(2) = theta+solutions(1);
    case 4 % theta=2*sum(solutions) spreated out according to spread_coeffs
        spread_coeffs = [0.6, 0.65, 0.75]; 
        solutions(2)=-spread_coeffs(1)*theta;
        solutions(3)=-spread_coeffs(2)*theta;
        solutions(4)=-spread_coeffs(3)*theta-solutions(1);
end
end
