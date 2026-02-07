function [vn] = rotV(vb,qnb)
    vn_ = quatmultiply(qnb,[0 vb]);
    vn = quatmultiply(vn_,quatinv(qnb));
    vn = vn(2:4);
end

