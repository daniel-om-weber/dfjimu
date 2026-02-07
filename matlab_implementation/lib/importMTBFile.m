function [sensors] = importMTBFile(filename)
    %%------- HELP
    %
    % This script allows the user to understand the step-wise procedure to load a file 
    %
    % The code is divided into two parts:
    %
    % 1) Set-up of the system and read some data
    %
    % 2) Event handler of the MT.
    %
    %%-------- IMPORTANT NOTES
    %
    % - For the code to work properly, make sure the code folder is your current directory in Matlab.
    %
    % - This code supports both 32 and 64 bits Matlab version.
    %
    % - The code requires xsensdeviceapi_com.dll to be registered in the Windows
    %   register (this is done automatically during the Xsens MT SDK installation)
    %
    
    if nargin<1
        filename = uigetfile('*.mtb','Select the MTB file');
    end

 
    %% Launching activex server
    try
        switch computer
        case 'PCWIN'
            h = actxserver('xsensdeviceapi_com32.IXsensDeviceApi')
        case 'PCWIN64'
            h = actxserver('xsensdeviceapi_com64.IXsensDeviceApi')
        otherwise
            error('CMT:os','Unsupported OS');
        end
    catch e
        fprintf('\n Please reinstall MT SDK or check manual,\n Xsens Device Api is not found.\n')
        rethrow(e);
    end
    fprintf( '\n ActiveXsens server - activated \n' );

    %% open the log file
    h.XsControl_openLogFile(filename);

    % getting number of MTs
    deviceID = cell2mat(h.XsControl_mainDeviceIds());
    num_MTs = length(deviceID);
    if num_MTs > 1
        fprintf('\n More than one device found, this script only uses the first one: %s\n',dec2hex(deviceID))
    else
        fprintf('\n Device found: %s\n',dec2hex(deviceID))
    end
    device = h.XsControl_device(deviceID);

    %% Tell the device to retain the data
    h.XsDevice_setOptions(device, h.XsOption_XSO_RetainRecordingData, 0);

    %% Load the log file and wait until it is loaded
    h.registerevent({'onProgressUpdated', @eventhandlerXsens});
    h.XsDevice_loadLogFile(device);
    fileLoaded = 0;
    while  fileLoaded == 0
        % wait untill maxSamples are arrived
        pause(.2)
    end
    fprintf('\nFile fully loaded\n')

    %% start data extracting
    % determine if device is Awinda station/dongle or a seperate device, in
    % case of dongle or station, find a MTw.
    if h.XsDeviceId_isWirelessMaster(deviceID)
        master = device;
        childDevices = h.XsDevice_children(master);
        childDevicesCnt = h.XsDevice_childCount(master);
        for iDevice = 1 : childDevicesCnt
            
            device = childDevices{iDevice};
            deviceID = h.XsDevice_deviceId(device);
            fprintf('\n Device found for data extracting: %s\n',dec2hex(deviceID))
            sensor.id = dec2hex(deviceID);  %this is the device number which corresponds to the number MT Manager
            sensor.fs = h.XsDevice_updateRate(device); % frequency of data
            % get total number of samples
            nSamples = h.XsDevice_getDataPacketCount(device);
            % allocate space
            Gyr_X = zeros(nSamples,1);
            Gyr_Y = zeros(nSamples,1);
            Gyr_Z = zeros(nSamples,1);
            Acc_X = zeros(nSamples,1);
            Acc_Y = zeros(nSamples,1);
            Acc_Z = zeros(nSamples,1);
            Mag_X = zeros(nSamples,1);
            Mag_Y = zeros(nSamples,1);
            Mag_Z = zeros(nSamples,1);
            RSSI = zeros(nSamples,1);
            quaternion = zeros(4,nSamples);
            
            time = zeros(nSamples,1); % if time available, otherwise packet counter
            PacketCounter = zeros(nSamples,1); % if time available, otherwise packet counter
            hasPacketCounter = false;
            hasTimeData = false;
            hasCalibratedData = false;
            readSample = 0;
            % for loop to extract the data
            for iSample = 1:nSamples
                % get data packet
                dataPacket =  h.XsDevice_getDataPacketByIndex(device,iSample-1);
                % check if dataPacket is a data packet
                if dataPacket
                    readSample = readSample+1;
                    % see if data packet contains certain data
                    if h.XsDataPacket_containsCalibratedData(dataPacket)
                        % extract dwiata, data ll always be in cells
                        packet = cell2mat(h.XsDataPacket_calibratedData(dataPacket));
                        
                        RSSI(iSample)= h.XsDataPacket_rssi(dataPacket);
                        quaternion(:,iSample) = cell2mat(h.XsDataPacket_orientationQuaternion_1(dataPacket)); 
                       
                        Acc_X(iSample)=packet(1);
                        Acc_Y(iSample)=packet(2);
                        Acc_Z(iSample)=packet(3);
                        Gyr_X(iSample)=packet(4);
                        Gyr_Y(iSample)=packet(5);
                        Gyr_Z(iSample)=packet(6);
                        Mag_X(iSample)=packet(7);
                        Mag_Y(iSample)=packet(8);
                        Mag_Z(iSample)=packet(9);
                    end
                    if h.XsDataPacket_containsSampleTimeFine(dataPacket)
                        hasTimeData = true;
                        time(readSample) = h.XsDataPacket_sampleTimeFine(dataPacket);
                    elseif h.XsDataPacket_containsPacketCounter(dataPacket)
                        hasPacketCounter = true;
                        PacketCounter(readSample) = h.XsDataPacket_packetCounter(dataPacket);
                    end
                    
                end
            end
            % clean up of allocated space
            %time(readSample+1:end) = [];
            %sdi(readSample+1:end) = [];
            if hasPacketCounter == true
                sensor.data=table(PacketCounter,Acc_X,Acc_Y,Acc_Z,Gyr_X,Gyr_Y,Gyr_Z,Mag_X,Mag_Y,Mag_Z,RSSI);
                sensor.q = quaternion;
            end
            sensors(iDevice) = sensor;
        end
    end
            
            
            %% close port and object
            h.XsControl_close();
            
            delete(h); % release COM-object
    clear h;

%% event handling
    function eventhandlerXsens(varargin)
        % device pointer is zero for progressUpdated
        devicePtr = varargin{3}{1};
        % The current progress
        currentProgress = varargin{3}{2};
        % The total work to be done
        total = varargin{3}{3};
        % Identifier, in the case the file name which is loaded
        identifier = varargin{3}{4};
        if currentProgress == 100
            fileLoaded = 1;
        end
    end
end
