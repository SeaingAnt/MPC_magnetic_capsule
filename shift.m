function [t0, x0, u0] = shift(T, t0, x0, u,f, gains, t)
st = x0;
con = u(1,:)';
k1 = f(st, con, gains,t);   % new 
k2 = f(st + T/2*k1, con, gains,t); % new
k3 = f(st + T/2*k2, con, gains,t); % new
k4 = f(st + T*k3, con, gains,t); % new
st_next_RK4 = T/6*(k1 +2*k2 +2*k3 +k4); % new
st_next_RK4(4:6) = st_next_RK4(4:6);

st_next_RK4(4:6) = [0 -st(6) st(5);
                    st(6) 0 -st(4); 
                   -st(5) st(4) 0]'*st_next_RK4(4:6);

st = st + st_next_RK4;               
x0 = full(st);

t0 = t0 + T;
u0 = [u(2:size(u,1),:);u(size(u,1),:)];
end