clear all
close all
clc

ros_init = false;

%% Init ROS
if(ros_init)
rosshutdown
    
% rosinit('160.69.69.100',11311);
% node = ros.Node('/matlab_mpc','160.69.69.100',11311);

rosinit
node = ros.Node('/matlab_mpc');

joint_sub = ros.Subscriber(node,'/iiwa/joint_states');
pose_sub = ros.Subscriber(node,'/MAC/filt_pose');
force_sub = ros.Subscriber(node,'/MAC/Reinforcement/ForceD');

gain_sub = ros.Subscriber(node,'/MAC/Reinforcement/Gain');

MPCpose_pub = ros.Publisher(node,'MAC/Reinforcement/goalMPC');
MPCqdot_pub = ros.Publisher(node,'MAC/Reinforcement/qdotMPC');

MPCpose = rosmessage(MPCpose_pub);
MPCqdot = rosmessage(MPCqdot_pub);

r = ros.Rate(node,2);
end
%% init
import casadi.*

params = {};
params.mu = 4*pi*1e-7;

res_mag_EPM = [0;0;1.48];
res_mag_IPM = [0;0;1.48];

mag_height_EPM = 0.1016;
mag_height_IPM = 0.011100087;
params.ma_mag = 2*pi*((mag_height_EPM/2)^(3))*norm(res_mag_EPM)/params.mu;
params.mc_mag = pi*(mag_height_IPM/2)^(2)*mag_height_IPM*norm(res_mag_IPM)/params.mu;

params.robot = importrobot('/home/antonio/mac_cpp/src/iiwa_tool/iiwa_tool_description/urdf/iiwa_tool.urdf');
params.toolname = "magnet_center";

params.robot.DataFormat = "column";
params.robot.Gravity = [0, 0, -9.81];

T = 0.5; % sampling time [s]
N = 4;

% joint states
q1 = SX.sym('q1'); q2 = SX.sym('q2'); q3 = SX.sym('q3'); q4 = SX.sym('q4');
q5 = SX.sym('q5'); q6 = SX.sym('q6'); q7 = SX.sym('q7');

q = [q1;q2;q3;q4;q5;q6;q7];

% capsule pose
x1 = SX.sym('x1'); x2 = SX.sym('x2'); x3 = SX.sym('x3'); eta1 = SX.sym('eta1');
eta2 = SX.sym('eta2'); eta3 = SX.sym('eta3');

% capsule desired pose
xd1 = SX.sym('xd1'); xd2 = SX.sym('xd2'); xd3 = SX.sym('xd3'); xd4 = SX.sym('xd4');
xd5 = SX.sym('xd5'); xd6 = SX.sym('xd6');

% desired force
target = SX.sym('target',3,1);

% control gains
gain = SX.sym('gain',12,1);

control = [xd1;xd2;xd3;xd4;xd5;xd6] ; n_controls = length(control);
state = [x1;x2;x3;eta1;eta2;eta3;q1;q2;q3;q4;q5;q6;q7]; n_states = length(state);

% initial joints position
q_init = [0.0; 0.0; 0.0; -1.6;  0.0;  1.5; 1.57];

% using predefine Casadi function to compute the endefector transformation
% and arm jacobian
tr = T_fk(state(7:end));

Jp = jacobian(tr(1:3,4),state(7:end));
Jr1 = jacobian(tr(1:3,1),state(7:end));
Jr2 = jacobian(tr(1:3,2),state(7:end));
Jr3 = jacobian(tr(1:3,3),state(7:end));
p1 = tr(1:3,1:3)^(-1)*Jr1;
p2 = tr(1:3,1:3)^(-1)*Jr2;
p3 = tr(1:3,1:3)^(-1)*Jr3;
Jr = [p2(3,:);p3(1,:);p1(2,:)];
J = [Jp;Jr];

params.J = Function('J',{q},{J});
 

% dynamic system Casadi function 
s_dot = sys_dim(params,state(1:3),state(4:6), state(7:end), control,gain,target);

f = Function('f',{state,control,gain,target},{s_dot}); 
U = SX.sym('U',n_controls,N);
P = SX.sym('P', n_states+5+12+3); % defing initial and final state
X = SX.sym('X',n_states,N+1);


%% MPC setup 
obj = 0; % Objective function
g = []; % constraints vector

Q = 1e6*eye(2,2);
Qr = 1e6*eye(3,3);
% Qp = 1e4*eye(3,3);
% Qpe = 1e4*eye(3,3);
Qp = 1e6*eye(3,3);
Qpe = 1e6*eye(3,3);


st  = X(:,1); % initial state
g = [g;st-P(1:n_states)]; % initial condition constraints
for k = 1:N
    st = X(:,k);  con = U(:,k);
    if k<N
    con_pk1 = U(:,k+1);
    else
    con_pk1 = [P(n_states+1:n_states+2);P(3);P(n_states+3:n_states+5)];    
    end
    Jq = params.J(st(end-6:end)');
    mu = sqrt(det(Jq*Jq'));
    
    obj = obj + (st(1:2)-P(n_states+1:n_states+2))'*Q*(st(1:2)-P(n_states+1:n_states+2)) + ...
        (st(4:6)-P(n_states+3:n_states+5))'*Qr*(st(4:6)-P(n_states+3:n_states+5)) + ...
        (con_pk1-con)'*[Qp, zeros(3,3);zeros(3,3), Qpe]*(con_pk1-con) +  ... 
        (P(1:6)-con)'*[Qp, zeros(3,3);zeros(3,3), Qpe]*(P(1:6)-con) -mu; % compute objective
    
    st_next = X(:,k+1);
    
    % integration runge-kutta 4th
    k1 = f(st, con, P(n_states+6:n_states+5+12),P(n_states+5+13:end));   
    k2 = f(st + T/2*k1, con, P(n_states+6:n_states+5+12),P(n_states+5+13:end)); 
    k3 = f(st + T/2*k2, con, P(n_states+6:n_states+5+12),P(n_states+5+13:end)); 
    k4 = f(st + T*k3, con, P(n_states+6:n_states+5+12),P(n_states+5+13:end)); 
    st_next_RK4 = T/6*(k1 +2*k2 +2*k3 +k4); 
    
    st_next_RK4(4:6) = [0 -st(6) st(5);
                        st(6) 0 -st(4); 
                       -st(5) st(4) 0]'*st_next_RK4(4:6);
    
    st_next_RK4 = st + st_next_RK4;               
    
    n_head = (con(4:6)'*con(4:6)-1);
    g = [g;st_next-st_next_RK4;k1(end-6:end);n_head]; % compute constraints
end


%% NLP definition

OPT_variables = [reshape(X,n_states*(N+1),1);reshape(U,n_controls*N,1)];
nlp_prob = struct('f', obj, 'x', OPT_variables, 'g', g, 'p', P);

opts = struct;

% ipopt set-up 
opts.ipopt.max_iter = 5;
opts.ipopt.print_level = 1;
opts.ipopt.tol = 1e-6;
opts.print_time = 0;
opts.ipopt.warm_start_init_point = 'yes';
opts.ipopt.dual_inf_tol =1e-6;
opts.ipopt.acceptable_compl_inf_tol = 1e-6;
opts.ipopt.constr_viol_tol = 1e-6;
solver = nlpsol('solver', 'ipopt', nlp_prob, opts);

% qrqp set-up
% opts.qpsol = 'qrqp';
% opts.beta = 4;
% opts.tol_du = 1e-4;
% opts.tol_pr = 1e-5;
% % opts.hessian_approximation = 'gauss-newton';
% opts.max_iter_ls = 5;
% opts.max_iter = 1;
% opts.qpsol_options.enable_fd = true;
% % opts.qpsol_options.error_on_fail = false;
%solver = nlpsol('solver', 'sqpmethod', nlp_prob, opts);


args = struct;

% Equality constraints
% dynamic system constraints
args.lbg(1:21*(N+1)-8) = 0;  
args.ubg(1:21*(N+1)-8) = 0; 

% Inequality constraints
% control action constraints
for i=1:7
args.lbg(n_states*2+i:n_states+8:end) = -0.001; 
args.ubg(n_states*2+i:n_states+8:end) =  0.001;
end

% state constraints
args.lbx(1:13*(N+1),1) = -inf;
args.lbx(1:13:end) = -inf;
args.lbx(7:13:end,1) = -2.83215314; args.lbx(8:13:end,1) = -1.95948;
args.lbx(9:13:end,1) = -2.83215; args.lbx(10:13:end,1) = -1.95948;
args.lbx(11:13:end,1) = -2.83215; args.lbx(12:13:end,1) = -1.95948;
args.lbx(13:13:end,1) = -2.919419;

args.ubx(1:13*(N+1),1) =  inf;
args.ubx(1:13:end) =  inf;
args.ubx(7:13:end,1) = 2.83215314; args.ubx(8:13:end,1) = 1.959488;
args.ubx(9:13:end,1) = 2.83215; args.ubx(10:13:end,1) = 1.959488;
args.ubx(11:13:end,1) = 2.832153; args.ubx(12:13:end,1) = 1.959488;
args.ubx(13:13:end,1) = 2.919419;

% desired pose constraints
args.lbx(n_states*(N+1)+1:n_states*(N+1)+n_controls*N,1) = -inf; 
args.ubx(n_states*(N+1)+1:n_states*(N+1)+n_controls*N,1) =  inf; 
args.lbx(n_states*(N+1)+1:n_controls:end,1) = 0.3; 
args.ubx(n_states*(N+1)+1:n_controls:end,1) = 0.5; 
args.lbx(n_states*(N+1)+2:n_controls:end,1) = -0.15; 
args.ubx(n_states*(N+1)+2:n_controls:end,1) = 0.1;

args.lbx(n_states*(N+1)+4:n_controls:end,1) = -1; 
args.ubx(n_states*(N+1)+4:n_controls:end,1) = 1;
args.lbx(n_states*(N+1)+5:n_controls:end,1) = -1; 
args.ubx(n_states*(N+1)+5:n_controls:end,1) = 1;
args.lbx(n_states*(N+1)+6:n_controls:end,1) = -1; 
args.ubx(n_states*(N+1)+6:n_controls:end,1) = 1;


%% simulation loop 
if(~ros_init)
t0 = 0;
% load variable from ros
x0 = [0.4066; 0.010; 0.3578; -0.0028; -0.9852 ; -0.1714; q_init];    % initial condition.

xs = [0.4200; 0.05; 0.3578; -0.0028; -0.9852 ; -0.1714; 0.0042; 0.0033; 0.3082]; % Reference posture.

gain = [100;100;100;10;10;10;100;100;100;10;10;10];

xx(:,1) = x0; % xx contains the history of states
t(1) = t0;

u0 = repmat(xs(1:6),1,N);  % control inputs 
X0 = repmat(x0,1,N+1)';  % initialization of the states decision variables

% Start MPC
mpciter = 0;
xx1 = [];
u_cl=[];

mu = 0;

while mpciter < 100
    tic
    args.p   = [x0;xs(1:2);xs(4:6);gain;xs(7:end)]; % set the values of the parameters vector
    % initial value of the optimization variables
    args.x0  = [reshape(X0',n_states*(N+1),1);reshape(u0',n_controls*N,1)]; 
    
    
    sol = solver('x0', args.x0, 'lbx', args.lbx,  'ubx', args.ubx,...
                'lbg', args.lbg, 'ubg', args.ubg,'p',args.p);     
   
    
    f_obj(mpciter+1) = full(sol.f);
    u = reshape(full(sol.x(n_states*(N+1)+1:end))',n_controls,N)'; % get controls only from the solution
    u_cl= [u_cl ; u(1,:)];
    
    t(mpciter+1) = t0;
    [t0, x0, u0] = shift(T, t0, x0, u,f,gain,xs(7:end)); % get the initialization of the next optimization step
    
    xx(:,mpciter+2) = x0;  
    X0 = reshape(full(sol.x(1:n_states*(N+1)))',n_states,N+1)'; % get solution TRAJECTORY
        
    % Shift trajectory to initialize the next step
    X0 = [X0(2:end,:);X0(end,:)];
    mpciter
    mpciter = mpciter + 1;
    a(mpciter) = toc;
    
    Jq = params.J(x0(end-6:end));
    mu(mpciter) = sqrt(det(full(Jq*Jq')));
end
toc
figure
plot(t,xx(1,1:end-1))
hold on; plot([t(1),t(end)],[xs(1),xs(1)],'--');ylim([0.406, 0.52]);xlim([0,5])
plot(t,u_cl(:,1),'-.');
figure; plot(t,xx(2,1:end-1))
hold on; plot([t(1),t(end)],[xs(2),xs(2)],'--');ylim([0.01, 0.28]);xlim([0,5])
plot(t,u_cl(:,2),'-.')
figure; plot(t,mu);ylim([0.094, 0.095]);xlim([0,5])
end     
%% ROS loop
if(ros_init)
% get x0
t = 0;
jointp = joint_sub.LatestMessage.Position;
%jointp = [joints.A1;joints.A2;joints.A3;joints.A4;joints.A5;joints.A6;joints.A7];


pose = pose_sub.LatestMessage.Pose.Pose;

position = [pose.Position.X; pose.Position.Y; pose.Position.Z];
rot = quat2rotm([pose.Orientation.W, pose.Orientation.X, ...
                    pose.Orientation.Y, pose.Orientation.Z]);
eta_r = rot(:,3);

force_des = force_sub.LatestMessage.Fd;

x0 = [position;eta_r;jointp];

xs = [position+[-0.0028; -0.9852 ; -0.1714]*0.05;eta_r; double(force_des(1));double(force_des(2));double(force_des(3))]; % Reference posture.

% be careful; start rc before this
gains_m = gain_sub.LatestMessage;

gain = [gains_m.KF.X;gains_m.KF.Y;gains_m.KF.Z;
        gains_m.KFs.X;gains_m.KFs.Y;gains_m.KFs.Z;
        gains_m.KX.X;gains_m.KX.Y;gains_m.KX.Z;
        gains_m.KAng.X;gains_m.KAng.Y;gains_m.KAng.Z];

xx(:,1) = x0; % xx contains the history of states

u0 = repmat(xs(1:6),1,N);  % control inputs 
X0 = repmat(x0,1,N+1)';  % initialization of the states decision variables

% Start MPC
mpciter = 0;
xx1 = [];
u_cl=[];
scatter(xs(1),xs(2),[],100);
hold on
sat = false;

p_d = [];
p = [];
mu = 0;

reset(r);
while 1
tic
% MPC computing

joints = joint_sub.LatestMessage.Position;
%jointp = [joints.A1;joints.A2;joints.A3;joints.A4;joints.A5;joints.A6;joints.A7];
jointp = joints;

pose = pose_sub.LatestMessage.Pose.Pose;

position = [pose.Position.X; pose.Position.Y; pose.Position.Z];
rot = quat2rotm([pose.Orientation.W, pose.Orientation.X, ...
                    pose.Orientation.Y, pose.Orientation.Z]);
eta_r = rot(:,3);

x0 = [position;eta_r;jointp];

force_des = force_sub.LatestMessage.Fd;

gains_m = gain_sub.LatestMessage;

gain = [gains_m.KF.X;gains_m.KF.Y;gains_m.KF.Z;
        gains_m.KFs.X;gains_m.KFs.Y;gains_m.KFs.Z;
        gains_m.KX.X;gains_m.KX.Y;gains_m.KX.Z;
        gains_m.KAng.X;gains_m.KAng.Y;gains_m.KAng.Z];
    
xs(7:end) = [double(force_des(1));double(force_des(2));double(force_des(3))];
    
args.p   = [x0;xs(1:2);xs(4:6);gain;xs(7:end)]; % set the values of the parameters vector
% initial value of the optimization variables
args.x0  = [reshape(X0',n_states*(N+1),1);reshape(u0',n_controls*N,1)]; 

sol = solver('x0', args.x0, 'lbx', args.lbx,  'ubx', args.ubx,...
                'lbg', args.lbg, 'ubg', args.ubg,'p',args.p);     

f_obj(mpciter+1) = full(sol.f);
u = reshape(full(sol.x(n_states*(N+1)+1:end))',n_controls,N)'; % get controls only from the solution
%xx1(:,1:13,mpciter+1)= reshape(full(sol.x(1:13*(N+1)))',13,N+1)'; % get solution TRAJECTORY
u_cl= [u_cl ; u(1,:)];

X0 = reshape(full(sol.x(1:n_states*(N+1)))',n_states,N+1)'; % get solution TRAJECTORY
% Shift trajectory to initialize the next step
X0 = [X0(2:end,:);X0(end,:)];
u0 = [u(2:size(u,1),:);u(size(u,1),:)];

% Send Results through ros

future_position = full(sol.x(1:n_states*(N+1)));

MPCpose.Pos = u_cl(end,1:3)';
MPCpose.Rot = u_cl(end,4:6)';

p_d(:,mpciter+1) = u(1,:)';
p(:,mpciter+1) = position(1:3,1);

MPCpose_pub.send(MPCpose);
MPCqdot.QDot = zeros(7,1);
MPCqdot_pub.send(MPCqdot);

t(mpciter+2) = t(mpciter+1) + T;

if(sat)
%mpciter
break;
end
mpciter = mpciter + 1;
a(mpciter) = toc;

Jq = params.J(x0(end-6:end));
mu(mpciter) = sqrt(det(full(Jq*Jq')));
toc
waitfor(r);
end


figure;
plot(t(1:end-1),p(1,1:end))
hold on; plot([t(1),t(end-1)],[xs(1),xs(1)],'--');
plot(t(1:end-1),p_d(1,:),'-.');
figure; plot(t(1:end-1),p(2,1:end))
hold on; plot([t(1),t(end)],[xs(2),xs(2)],'--');
plot(t(1:end-1),p_d(2,:),'-.')
figure; plot(t,mu);ylim([0.094, 0.095]);xlim([0,5])
end
