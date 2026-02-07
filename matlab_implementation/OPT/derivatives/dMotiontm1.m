function der = dMotiontm1(q_lint,q_lintm1)
% dMotion: Returns derivative of the costMotion wrt state at timestap 't-1'
global T;

    der = ((1/T)*dlogdq*quatR(quatmultiply(quatconj(q_lintm1), q_lint))*dexpnCdexpn*dexpndn);
end

