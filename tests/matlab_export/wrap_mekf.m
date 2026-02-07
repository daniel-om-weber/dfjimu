function [q_s1, q_s2] = wrap_mekf(gyr, gyr_2, C1, C2, Fs, q_init, cov_w, cov_i)
% WRAP_MEKF  Run MEKF filter as a callable function.
%   Extracted from MEKF_conn.m with all globals eliminated.
%
%   Inputs:
%     gyr     - (3, N) gyroscope data sensor 1
%     gyr_2   - (3, N) gyroscope data sensor 2
%     C1      - (3, N) preprocessed accelerations sensor 1
%     C2      - (3, N) preprocessed accelerations sensor 2
%     Fs      - sampling frequency (Hz)
%     q_init  - (1, 4) initial quaternion [w x y z]
%     cov_w   - (6, 6) gyro noise covariance (std devs on diagonal)
%     cov_i   - (3, 3) initial orientation covariance
%
%   Outputs:
%     q_s1    - (N, 4) orientation quaternions sensor 1
%     q_s2    - (N, 4) orientation quaternions sensor 2

    % Declare globals needed by library functions (integrateGyr uses T,
    % calculateK uses N, approxDerivative uses Fs)
    global T N;
    T = 1/Fs;
    N = size(gyr, 2);

    %% Initial orientation
    q_1 = q_init;
    q_lin_s1 = q_1;
    q_lin_s2 = q_1;

    P = cov_i(1)*eye(6);           % initial Process covariance matrix
    Q = cov_w;                      % Process noise covariance (from static)

    % Measurement noise covariance (tuned on static window)
    R = eye(3)*2*(.076^2);

    G = T*eye(6);

    % Initialize by strapdown integration
    orientation_s1 = integrateGyr(gyr', q_1);
        orientation_s1 = orientation_s1';
    orientation_s2 = integrateGyr(gyr_2', q_1);
        orientation_s2 = orientation_s2';

    %% Multiplicative EKF loop
    for t = 2:N
        %% A) Time Update
        F = zeros(6,6);
        F(1:3,1:3) = EXPr(-T*gyr(:,t-1));
            F(4:6,4:6) = EXPr(-T*gyr_2(:,t-1));

        q_lin_s1 = quatmultiply(orientation_s1(:,t-1)', EXPq((T/2)*gyr(:,t-1)));
            q_lin_s2 = quatmultiply(orientation_s2(:,t-1)', EXPq((T/2)*gyr_2(:,t-1)));

        P = F*P*F' + G*Q*G';

        %% B) Measurement update
        Rbn_s1 = quat2matrix(quatconj(q_lin_s1));
            Rbn_s2 = quat2matrix(quatconj(q_lin_s2));

        % Link constraint error
        e = zeros(3,1);
        e(1:3,1) = (Rbn_s1'*C1(:,t)) - (Rbn_s2'*C2(:,t));

        % H matrix
        H = zeros(3,6);
        H(1:3,1:3) = Rbn_s1'*crossM(C1(:,t));
        H(1:3,4:6) = -Rbn_s2'*crossM(C2(:,t));

        S = H*P*H' + R;
        K = (P*H')/S;

        n = K*e;
        P_tilde = P - K*S*K';

        %% C) Relinearize (Reset)
        q_lin_s1 = quatmultiply(q_lin_s1, EXPq(n(1:3)/2));
        q_lin_s2 = quatmultiply(q_lin_s2, EXPq(n(4:6)/2));

        J = zeros(6,6);
        J(1:3,1:3) = EXPr(-n(1:3));
        J(4:6,4:6) = EXPr(-n(4:6));
        P = J*P_tilde*J';

        % Output
        orientation_s1(:,t) = q_lin_s1;
            orientation_s2(:,t) = q_lin_s2;
    end

    % Return as (N, 4) — transpose from (4, N)
    q_s1 = orientation_s1';
    q_s2 = orientation_s2';
end
