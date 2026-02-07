function [v] = LOGq(q)
    qv = [q(2) q(3) q(4)];              %approximation 52a
    v = (acos(q(1))/(norm(qv)+realmin))*qv;       % 51a
end

