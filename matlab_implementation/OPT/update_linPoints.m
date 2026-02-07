function q_lin_ = update_linPoints(q_lin,n)
%UPDATE_LINPOINTS 
    for i=1:size(q_lin,1)
        q_lin_(i,:) = quatmultiply(q_lin(i,:),EXPq(n(i,:)/2));
    end
end

