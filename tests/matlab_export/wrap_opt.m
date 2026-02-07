function [q_s1, q_s2] = wrap_opt(gyr, gyr_2, C1, C2, Fs, q_init, cov_w, cov_i, cov_lnk, n_iter)
% WRAP_OPT  Run Gauss-Newton optimizer as a callable function.
%   Extracted from OPT_conn.m with all globals eliminated.
%
%   Inputs:
%     gyr      - (3, N) gyroscope data sensor 1
%     gyr_2    - (3, N) gyroscope data sensor 2
%     C1       - (3, N) preprocessed accelerations sensor 1
%     C2       - (3, N) preprocessed accelerations sensor 2
%     Fs       - sampling frequency (Hz)
%     q_init   - (1, 4) initial quaternion [w x y z]
%     cov_w    - (6, 6) gyro noise covariance
%     cov_i    - (3, 3) initial orientation covariance
%     cov_lnk  - (3, 3) link constraint covariance
%     n_iter   - number of GN iterations
%
%   Outputs:
%     q_s1     - (N, 4) orientation quaternions sensor 1
%     q_s2     - (N, 4) orientation quaternions sensor 2

    % Declare globals needed by library functions (integrateGyr uses T,
    % calculateK uses N, approxDerivative uses Fs)
    global T N;
    T = 1/Fs;
    N = size(gyr, 2);

    %% Covariance sub-matrices
    cov_w1 = cov_w(1:3,1:3);
    cov_w2 = cov_w(4:6,4:6);
    cov_a = eye(3)*1e-01;   % NOT USED but needed for calcJac2 signature

    %% Initial orientation
    q_1 = q_init;

    % Initialize by strapdown integration
    q_lin_s1 = integrateGyr(gyr', q_1);
    q_lin_s2 = integrateGyr(gyr_2', q_1);

    %% Gauss-Newton iterations
    for k = 1:n_iter
        n = zeros(2*N, 3);

        %% Weighted errors for current linearization points
        % Sensor 1
        epsilon_i_s1 = costInit(n(1,:), q_lin_s1(1,:), q_1, cov_i);
        epsilon_w_s1 = costMotion(n(1:N,:), gyr, q_lin_s1, cov_w1);
        % Sensor 2
        epsilon_i_s2 = costInit(n(N+1,:), q_lin_s2(1,:), q_1, cov_i);
        epsilon_w_s2 = costMotion(n(N+1:2*N,:), gyr_2, q_lin_s2, cov_w2);

        % Link constraint
        epsilon_lnk = costLnk(q_lin_s1, q_lin_s2, C1, C2, cov_lnk);

        % Order - Sensor 1
        epsilon_w_s1 = epsilon_w_s1'; epsilon_w_s1 = epsilon_w_s1(:);
        epsilon_w_s1(1:3) = epsilon_i_s1;
        epsilon_1 = epsilon_w_s1;

        % Sensor 2
        epsilon_w_s2 = epsilon_w_s2'; epsilon_w_s2 = epsilon_w_s2(:);
        epsilon_w_s2(1:3) = epsilon_i_s2;
        epsilon_2 = epsilon_w_s2;

        % Link under everything
        epsilon_lnk = epsilon_lnk'; epsilon_lnk = epsilon_lnk(:); epsilon_lnk = epsilon_lnk';
        epsilon = [epsilon_1' epsilon_2' epsilon_lnk];

        %% Jacobian
        J_S1 = calcJac2(size(epsilon_1,1), N, n(1:N), q_lin_s1, q_1, cov_i, cov_w1, cov_a);
        J_S2 = calcJac2(size(epsilon_2,1), N, n(N+1:2*N), q_lin_s2, q_1, cov_i, cov_w2, cov_a);
        J_S1S2 = calcJac_Link(size(epsilon_lnk,2), (size(J_S1,2)+size(J_S2,2)), q_lin_s1, q_lin_s2, C1, C2, cov_lnk);

        % Fill Jacobian
        J = sparse((size(epsilon_1,1) + size(epsilon_2,1) + size(epsilon_lnk,2)), (2*3*N));
        J(1:size(J_S1,1), 1:size(J_S1,2)) = J_S1;
        J((size(J_S1,1)+1):size(J_S1,1)+size(J_S2,1), (size(J_S1,2)+1):(size(J_S1,2)+size(J_S2,2))) = J_S2;
        J((size(J_S1,1)+size(J_S2,1)+1):end, 1:end) = J_S1S2;

        %% Gradient & Hessian
        epsilon = epsilon';
        G = J'*epsilon(:);
        H = J'*J;

        %% Apply update
        n = -(H\G);

        %% Update linearization points
        q_lin_s1 = update_linPoints(q_lin_s1, vec2mat(n(1:N*3), 3));
        q_lin_s2 = update_linPoints(q_lin_s2, vec2mat(n((N*3)+1:2*(N*3)), 3));
    end

    q_s1 = q_lin_s1;
    q_s2 = q_lin_s2;
end
