function der = dMotion(q_lint,q_lintm1)
% dMotion: Returns derivative of the costMotion wrt state at timestap 't'
global T;

    der = ((1/T)*dlogdq*quatL(quatmultiply(quatconj(q_lintm1), q_lint))*dexpndn);
end

