function s_dot = sys_dim(params, pr, eta_r, q_conf, u, gain,t)

tr = T_fk(q_conf);

J = params.J(q_conf);

pa = tr(1:3,4);
eta_a = tr(1:3,3);

fpar = params;
fpar.ma_hat = eta_a;
fpar.mc_hat = eta_r;

p = pr-pa;

fpar.pnorm = norm(p);
fpar.p_hat = p/fpar.pnorm;
      
fpar.Z = eye(3)-(5*fpar.p_hat*fpar.p_hat');
fpar.G = eye(3)-(fpar.p_hat*fpar.p_hat');
fpar.D = 3*(fpar.p_hat*fpar.p_hat') - eye(3);
fpar.Y = eye(3);

JFp = jacobian_Fp(fpar);
JFma = jacobian_Fma(fpar);
JFmc = jacobian_Fmc(fpar);

Jtp = jacobian_Tp(fpar);
Jtma = jacobian_Tma(fpar);
Jtmc = jacobian_Tmc(fpar);

skew_mc_hat = [0 -fpar.mc_hat(3) fpar.mc_hat(2);
               fpar.mc_hat(3) 0 -fpar.mc_hat(1); 
              -fpar.mc_hat(2) fpar.mc_hat(1) 0];
                  

M_c = [eye(3,3) zeros(3,3); zeros(3,3) skew_mc_hat'];
          
skew_ma_hat = [0 -fpar.ma_hat(3) fpar.ma_hat(2);
               fpar.ma_hat(3) 0 -fpar.ma_hat(1); 
              -fpar.ma_hat(2) fpar.ma_hat(1) 0];
                  

M_a = -[eye(3,3) zeros(3,3); zeros(3,3) skew_ma_hat']*J;

Jq = [JFp JFma; Jtp Jtma]*M_a;
Jx = [JFp JFmc; Jtp Jtmc]*M_c;

force = mag_force(fpar);
torque = mag_torque(fpar);

error = [t(1:3)-force;-torque];
u_norm = u(4:6)/norm(u(4:6));

% regularization for rank 5
%Px = (Jx'*Jx+1e-12*eye(6))^(-1)*Jx'*Jx;

%P = [eye(3), zeros(3); zeros(3), Px(4:6,4:6)];

P = eye(6);

er_tau = error + gain(7:12).*([u(1:3) - pr;u_norm - eta_r]);

J_tot = [Jx+gain(1:6).*gain(7:12).*P+...
         gain(7:12).*P+P-er_tau/(er_tau'*er_tau)*[t(1:3);zeros(3,1)]', Jq];

s_dot = J_tot'*(J_tot*J_tot')^(-1)*(gain(1:6).*er_tau);

% x_dot = -(Jx'*Jx+1e-5*eye(6))^(-1)*Jx'*Jq*u(1:7);
% q_dot = u(1:7);

end

function f_magnet = mag_force(params)
    
    K = (3*params.mu*params.ma_mag*params.mc_mag/(4*pi*params.pnorm^4));

    term1 = params.ma_hat * (params.mc_hat'*params.p_hat);
    term2 = params.mc_hat * (params.ma_hat'*params.p_hat);
    term3 = (params.mc_hat'*(params.Z*params.ma_hat))*params.p_hat;
    f_magnet = K * (term1 + term2 + term3);

end

function t_magnet = mag_torque(params) 

    K = params.mu*params.ma_mag*params.mc_mag/(4*pi*params.pnorm^3);
    skew_mch=[0 -params.mc_hat(3) params.mc_hat(2) ; params.mc_hat(3) 0 -params.mc_hat(1); ...
        -params.mc_hat(2) params.mc_hat(1) 0 ];
    
    t_magnet = K * skew_mch*(params.D * params.ma_hat);

end

function disp = ReducedVersorLemma(a,b)

    skew_a = [0 -a(3) a(2) ; a(3) 0 -a(1); -a(2) a(1) 0 ];
    vsinth = skew_a*b/(norm(a)*norm(b));
    
    disp = asin(vsinth);
        
end

function Fp = jacobian_Fp(params) 

K = 3*params.mu*params.ma_mag*params.mc_mag/(4*pi*params.pnorm^5);

term1 = params.ma_hat*params.mc_hat'*params.Z;
term2 = params.mc_hat*params.ma_hat'*params.Z;
term3 = params.mc_hat'*params.ma_hat*params.Z;
term4 = -5*(params.p_hat*params.p_hat')*(params.mc_hat*params.ma_hat')*params.G;
term5 = -5*(params.p_hat*params.p_hat')*(params.ma_hat*params.mc_hat')*params.G;
term6 = -5*params.mc_hat'*params.p_hat*params.p_hat'*params.ma_hat*params.Z;

Fp = K * (term1 + term2 + term3 + term4 + term5 + term6);
end

function Fma = jacobian_Fma(params)

K = 3*params.mu*params.ma_mag*params.mc_mag/(4*pi*params.pnorm^4);

term1 = params.mc_hat'*params.p_hat*eye(3);
term2 = params.mc_hat*params.p_hat';
term3 = params.p_hat*params.mc_hat'*(params.Z);

Fma = K * (term1 + term2 + term3);
end

function Fmc = jacobian_Fmc(params)

K = 3*params.mu*params.ma_mag*params.mc_mag/(4*pi*params.pnorm^4);

term1 = params.ma_hat * params.p_hat';
term2 = params.ma_hat'*params.p_hat*eye(3);
term3 = params.p_hat*params.ma_hat'*(params.Z);

Fmc = K * (term1 + term2 + term3);

end

function Tp = jacobian_Tp(params)
pmc = (1/(params.pnorm^(3)))*params.mc_hat;
skew_pmc=[0 -pmc(3) pmc(2) ; pmc(3) 0 -pmc(1) ; -pmc(2) pmc(1) 0 ];
pma = params.D*(params.ma_hat);
skew_pma=[0 -pma(3) pma(2) ; pma(3) 0 -pma(1) ; -pma(2) pma(1) 0 ];
K = 3*params.mu*params.ma_mag*params.mc_mag/(4*pi);

term1 = (1/params.pnorm)*(params.p_hat*params.ma_hat')*(params.G);
term2 = ((params.p_hat'*params.ma_hat)/params.pnorm) * params.G;
term3 = skew_pmc*(term1 + term2);
term4 = (1/(params.pnorm^4))*skew_pma*(params.mc_hat*params.p_hat');

Tp = params.Y*(K *(term3 + term4));

end

function Tma = jacobian_Tma(params)

K = params.mu*params.ma_mag*params.mc_mag/(4*pi*params.pnorm^3);
pmc = params.mc_hat;
skew_pmc=[0 -pmc(3) pmc(2) ; pmc(3) 0 -pmc(1) ; -pmc(2) pmc(1) 0 ];

Tma = params.Y*(K*skew_pmc*params.D);

end

function Tmc = jacobian_Tmc(params)

K = params.mu*params.ma_mag*params.mc_mag/(4*pi*params.pnorm^3);

pma = params.D*(params.ma_hat);
skew_pma=[0 -pma(3) pma(2) ; pma(3) 0 -pma(1) ; -pma(2) pma(1) 0 ];

Tmc = params.Y*(-K*skew_pma);

end

 


