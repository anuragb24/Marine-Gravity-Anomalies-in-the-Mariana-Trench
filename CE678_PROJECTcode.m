% MARINE GRAVITY ANOMALIES
% CE678 PROJECT
% Supervisor- Prof. Balaji Devaraju
% Anurag Basu(241030401)
 
%--------------------------------------------------------------------------
% 1. Show header data of SARAL data from cycle 001 and pass 0075. 
%--------------------------------------------------------------------------

saral_data = 'E:\SEM 1\physical geodesy\008308_saral_ssh_33\001\001_0075ssh.33.nc';
ncdisp(saral_data);

lat = ncread(saral_data,'glat.00'); % Store latitude for cycle 001 and pass 0075.
long = ncread(saral_data,'glon.00'); % Store longitude for cycle 001 and pass 0075.
sea_surface_height = ncread(saral_data,'ssh.33'); % Sea surface height for cycle 001 and pass 0075.

disp("The size of the latitude column from file '001_0075ssh.33.nc' is");
disp(size(lat));
disp("The size of the longitude column from file '001_0075ssh.33.nc' is");
disp(size(long))
disp("The size of the sea surface height column from file '001_0075ssh.33.nc' is");
disp(size(sea_surface_height))



%--------------------------------------------------------------------------
%2. Read all cycles and passes
%--------------------------------------------------------------------------

%Define external forlder path
folder = 'E:\SEM 1\physical geodesy\008308_saral_ssh_33';

% number of cycles
n_cycle = 35; 

%Define passes
file_suf = [47, 62, 75, 90, 133, 176, 161, 262, 247, 290, ...
            305, 348, 333, 376, 447, 434, 505, 462, 533, 520, ...
            591, 548, 677, 634, 705, 720, 763, 748, 791, 834, ...
            877, 892, 963, 920, 991, 978];

%number of passes
n_passes = length(file_suf);

%Preallocate variable to store all data
data = cell(n_cycle, n_passes);

%Loop to read all the cycle, pass and its required variables 
for i = 1:n_cycle
    sf_name = sprintf('%03d', i);
    sf_path = fullfile(folder, sf_name);
    
    for j = 1:n_passes
        file_pre = sprintf('%04d', file_suf(j));
        file_name = [sf_name, '_', file_pre, 'ssh.33.nc'];
        file_path = fullfile(sf_path, file_name);
        
        if isfile(file_path)
            latitude = ncread(file_path, 'glat.00');
            longitude = ncread(file_path, 'glon.00');
            ssh = ncread(file_path, 'ssh.33');
            data{i, j} = struct('latitude', latitude, 'longitude', longitude, 'ssh', ssh);
        end
    end
end

disp('Latitude, Longitude, and SSH data have been successfully stored for all cycles and passes.');


%--------------------------------------------------------------------------
%3. Plot all passes from cycle 2 to identify ascending and descending passes
%--------------------------------------------------------------------------

figure;
hold on;

% Preallocate variable to store legends of pass numbers
data_legends = {};

% Loop to plot all passes from cycle 2
for j = 1:n_passes
    i = 2;
    if ~isempty(data{i, j})
        lat = data{i, j}.latitude;
        lon = data{i, j}.longitude;

        % Alternate colors for passes
        if mod(j, 2) == 1
            plot(lon, lat, 'Color', [0 0.4470 0.7410]); 
        else
            plot(lon, lat, 'Color', [0.8500 0.3250 0.0980]);
        end
        
        % Calculate the midpoint and angle for text rotation
        mid_index = round(length(lat) / 2);  % Find the midpoint index
        if mid_index < length(lat)
            delta_x = lon(mid_index + 1) - lon(mid_index);
            delta_y = lat(mid_index + 1) - lat(mid_index);
            angle = atan2d(delta_y, delta_x);  % Angle in degrees
        end

        % Position text at the midpoint of the line
        text(lon(mid_index), lat(mid_index), ['Pass ', num2str(file_suf(j))], ...
             'FontSize', 8, 'Color', 'k', ...
             'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right', ...
             'Rotation', angle);

        data_legends{end+1} = ['Pass ', num2str(file_suf(j))];
    end
end

title('Line Plot of All Passes (for cycle 2)');
xlabel('Longitude (Degrees)');
ylabel('Latitude (Degrees)');
axis equal;  % Set equal axes
daspect([1, 1, 1]);       % Set data aspect ratio to 1:1
legend(data_legends, 'Location', 'eastoutside');
grid on;
hold off;

disp('All passes from cycle 2 successfully plotted');



%--------------------------------------------------------------------------
%4. Plot all passes from all cycle
%--------------------------------------------------------------------------

figure;
hold on;

% Loop through all cycles and passes
for i = 1:n_cycle
    for j = 1:n_passes
        % Check if data exists for the current cycle and pass
        if ~isempty(data{i, j})
            lat = data{i, j}.latitude;
            lon = data{i, j}.longitude;
            
            % Alternate colors for odd and even passes
            if mod(j, 2) == 1
                plot(lon, lat, 'Color', [0 0.4470 0.7410]);
            else
                plot(lon, lat, 'Color', [0.8500 0.3250 0.0980]);
            end
        end
    end
end

% Customize plot
title('Line Plot of Latitude vs Longitude for All Passes Across All Cycles');
xlabel('Longitude (Degrees)');
ylabel('Latitude (Degrees)');
axis equal;  %Set equal axes
daspect([1, 1, 1]);       % Set data aspect ratio to 1:1
grid on;

hold off;

disp('All passes across all cycles have been successfully plotted.');


%--------------------------------------------------------------------------
%5. Define ascending pass data
%--------------------------------------------------------------------------

%Define ascending passes
ascend_suf = [47, 75, 133, 161, 247, 305, 333, 447, 505,  ...
              533, 591, 677, 705, 763, 791, 877, 963, 991];

%Create subset data for only ascending passes
ascend = cell(n_cycle, round(n_passes / 2));
for i = 1:n_cycle
    ascend(i, :) = data(i, 1:2:n_passes);
end

disp('Subset of data with ascend passes has been successfully created');


%--------------------------------------------------------------------------
%6. Plot defined ascend pass data only for confirmation
%--------------------------------------------------------------------------

figure;
hold on;

%store legends of ascend pass numbers
legends = {};

%Loop to plot all ascend passes from cycle 2
for j = 1:size(ascend, 2)
    i = 2;

    if ~isempty(ascend{i, j})
        lat = ascend{i, j}.latitude;
        lon = ascend{i, j}.longitude;
        
        plot(lon, lat, 'Color', [0 0.4470 0.7410], 'DisplayName', ['Pass ', num2str(file_suf(j))]);
        legends{end + 1} = ['Pass ', num2str(ascend_suf(j))];

        % Calculate the midpoint

        mid_index = round(length(lat) / 2);
        if mid_index < length(lat)
            delta_x = lon(mid_index + 1) - lon(mid_index);
            delta_y = lat(mid_index + 1) - lat(mid_index);
            angle = atan2d(delta_y, delta_x);  % Angle in degrees
        end
        
        % Add a label on the plot for each pass number at the starting point
        text(lon(mid_index), lat(mid_index), ['Pass', num2str(ascend_suf(j))], 'FontSize', 8, 'Color', 'k', ...
             'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right', 'Rotation',angle);
    end
end

title('Line plot of latitude vs longitude for all ascend passes (for Cycle 2)');
xlabel('Longitude (Degrees)');
ylabel('Latitude (Degrees)');
legend(legends, 'Location', 'eastoutside');
axis equal;  %Set equal axes
daspect([1, 1, 1]);       % Set data aspect ratio to 1:1
grid on;

hold off;

disp('All ascend passes from cycle 2 is succesfully plotted');


%--------------------------------------------------------------------------
%7. Plot lat, long and sea surface height of all passes from cycle 2
%--------------------------------------------------------------------------

figure;
hold on;

% Preallocate variable to store legends of ascend pass numbers
legends = {};

% Initialize arrays to store all SSH values
all_ssh_values = [];

% Loop to plot all ascend passes from cycle 2
for j = 1:size(ascend, 2)
    i = 2;  % Cycle 2

    % Check if data exists for the current ascend pass
    if ~isempty(ascend{i, j})
        lat = ascend{i, j}.latitude;
        lon = ascend{i, j}.longitude;
        ssh = ascend{i, j}.ssh;
        
        % Store SSH values for calculating z-axis limits later
        all_ssh_values = [all_ssh_values; ssh];
        
        % Plot 3D line of Longitude, Latitude, and SSH
        plot3(lon, lat, ssh, 'DisplayName', ['Pass ', num2str(ascend_suf(j))]);
        
        % Add legend entry for the current pass
        legends{end + 1} = ['SSH of pass ', num2str(ascend_suf(j))];
    end
end

% Calculate z-axis limits based on the range of SSH values
z_min = min(all_ssh_values);
z_max = max(all_ssh_values);
zlim([z_min, z_max]);

% Customize the plot
title('3D Plot of Latitude, Longitude, and SSH for All ascend Passes (Cycle 2)');
xlabel('Longitude (Degrees)');
ylabel('Latitude (Degrees)');
zlabel('Sea Surface Height (meters)');

% Set data aspect ratio for proportional scaling of lat, lon, and SSH
daspect([1, 1, (z_max - z_min) / max(range(lon), range(lat))]);

% Flip the y-axis
set(gca, 'YDir', 'reverse');

% Add legend
legend(legends, 'Location', 'eastoutside');

% Set the view to 3D for better visualization
view(3);
grid on;

hold off;



%--------------------------------------------------------------------------
%8. Plot lat, long and ssh of all passes from all cycles at once
%--------------------------------------------------------------------------

% % Initialize figure and hold for multiple plots
% figure;
% hold on;
% 
% % Preallocate variable to store legends of ascend pass numbers
% legends = {};
% 
% % Total number of cycles
% n_cycle = 35;
% 
% % Loop through all cycles and plot each ascend pass
% for i = 1:n_cycle
%     % Loop to plot all ascend passes for the current cycle
%     for j = 1:size(ascend, 2)
%         if ~isempty(ascend{i, j})
%             lat = ascend{i, j}.latitude;
%             lon = ascend{i, j}.longitude;
% 
%             % Plot each ascend pass
%             plot(lon, lat, '-o', 'Color', [0 0.4470 0.7410], 'DisplayName', ['Pass ', num2str(j), ', Cycle ', num2str(i)]);
%             legends{end + 1} = ['Pass ', num2str(j), ', Cycle ', num2str(i)];
%         end
%     end
% end
% 
% % Add title, labels, and legend
% title('Line plot of latitude vs longitude for all ascend passes across cycles');
% xlabel('Longitude');
% ylabel('Latitude');
% legend(legends, 'Location', 'eastoutside');
% hold off;
% 
% disp('All ascend passes across cycles have been successfully plotted.');


%-------------------------------------------------------------------------- 
%9. Descriptive analysis of the ascend passes data.
%Calculate Mean, Median, Standard Deviation, and Z-Score for SSH Data 
%-------------------------------------------------------------------------- 

% Preallocate variable for statistics on ascending data
desc_ascend = cell(n_cycle, round(n_passes / 2)); % New cell array for statistics

% Loop to calculate statistics for ascending passes
for i = 1:n_cycle
    for j = 1:round(n_passes / 2)
        if ~isempty(ascend{i, j})
            ssh = ascend{i, j}.ssh; 
            lat = ascend{i, j}.latitude;
            lon = ascend{i, j}.longitude;
            % Calculate max, min, mean, median, std, and z-score
            max_ssh = max(ssh);
            min_ssh = min(ssh);
            mean_ssh = mean(ssh);
            median_ssh = median(ssh);
            std_ssh = std(ssh);
            zscore_ssh = (ssh - mean_ssh) / std_ssh; % Z-score calculation
            % outliers = abs(zscore_ssh) > 2.5;  % Find indices of outliers
            % if outliers > 0
            %     num_outliers = sum(outliers);
            % end
            % total_ssh = numel(ssh);
            % % Calculate and store outlier percentage using Z-score method
            % if total_ssh > 0
            %     outlier_perc = (num_outliers / total_ssh) * 100;
            % end
            % Store the statistics in the desc_ascend cell
            desc_ascend{i, j} = struct('mean_ssh', mean_ssh,'median_ssh', median_ssh, ...
                              'maximum_ssh', max_ssh, 'minimum_ssh', min_ssh, ...
                              'std_ssh', std_ssh,'latitude', lat, 'longitude', lon, 'zscore_ssh', zscore_ssh);
            % , 'outliers', outliers, 'number_of_outliers', num_outliers, ...
            %                   'total_number_of_ssh_values', total_ssh,'outlier_percentage', outlier_perc);
        end
    end
end

disp('Statistics for ascending passes have been calculated and stored in desc_ascend.');


%--------------------------------------------------------------------------
%10. Create 100 * 100 grid of lat, long for all sea surface height data
%--------------------------------------------------------------------------

% Initialize arrays to collect all latitudes, longitudes, and SSH values across all cycles
lat_all = [];
lon_all = [];
ssh_all = [];

% Loop through all ascend passes and cycles
for j = 1:size(ascend, 2)  % Loop through all passes
    for i = 1:size(ascend, 1)  % Loop through all cycles
        % Check if data exists in the current cell
        if ~isempty(ascend{i, j})
            % Append current cycle data to the main arrays
            lat_all = [lat_all; double(ascend{i, j}.latitude)];
            lon_all = [lon_all; double(ascend{i, j}.longitude)];
            ssh_all = [ssh_all; double(ascend{i, j}.ssh)];
        end
    end
end

% Create a grid for longitude and latitude
[lon_grid, lat_grid] = meshgrid(linspace(min(lon_all), max(lon_all), 100), ...
                                linspace(min(lat_all), max(lat_all), 100));


% Initialize the SSH grid with NaNs
ssh_grid_raw = NaN(size(lat_grid));

% Loop through each SSH value to find the corresponding grid index
for k = 1:length(ssh_all)
    [~, lon_idx] = min(abs(lon_grid(1,:) - lon_all(k)));  % Find the closest grid longitude index
    [~, lat_idx] = min(abs(lat_grid(:,1) - lat_all(k))); % Find the closest grid latitude index
    ssh_grid_raw(lat_idx, lon_idx) = ssh_all(k); % Place the SSH value in the correct grid location
end


% Interpolate SSH data over the grid using nearest neighbour
ssh_grid = griddata(lon_all, lat_all, ssh_all, lon_grid, lat_grid, 'natural');


%--------------------------------------------------------------------------
%11. Plot lat, long and ssh of all passes from all cycles at once in 3D
%--------------------------------------------------------------------------

figure;
hold on;

% Plot the interpolated surface for better 3D visualization
surf(lon_grid, lat_grid, ssh_grid, 'EdgeColor', 'none', 'FaceAlpha', 0.8);

% Customize the plot
title('3D Surface Plot of Latitude, Longitude, and SSH for All Cycles in ascend Passes');
xlabel('Longitude (Degrees)');
ylabel('Latitude (Degrees)');
zlabel('Sea Surface Height (meters)');

% Add colorbar to visualize SSH magnitude
cbar = colorbar;  
ylabel(cbar, 'SSH (meters)'); 
colormap(bone);

% Set axis limits based on your data to improve visualization
xlim([min(lon_grid(:)), max(lon_grid(:))]);
ylim([min(lat_grid(:)), max(lat_grid(:))]);
zlim([min(ssh_grid(:)), max(ssh_grid(:))]);

% Calculate the ranges for each axis
x_range = range(lon_grid(:));
y_range = range(lat_grid(:));
z_range = range(ssh_grid(:));

% Use daspect to set the aspect ratio relative to the ranges
daspect([x_range, y_range, z_range]);

% Enhance 3D view
view(3);  % Adjust the view angle for better perspective
grid on;

hold off;

%--------------------------------------------------------------------------
%12. Plot lat, long and ssh of all passes from all cycles at once in 2D
%--------------------------------------------------------------------------

figure;

% Use imagesc or pcolor to represent SSH in a 2D plot with hue
imagesc(lon_grid(1, :), lat_grid(:, 1), ssh_grid);
set(gca, 'YDir', 'normal');  % Ensure latitudes are in ascending order on the y-axis

% Customize the plot
title('2D Plot of Sea Surface Height (ssh) variation in space');
xlabel('Longitude (Degrees)');
ylabel('Latitude (Degrees)');
cbar = colorbar;
colormap(bone);  % Set colormap to represent SSH values
ylabel(cbar,'ssh (meters)');

% Set axis limits and other visual preferences if needed
axis equal;
daspect([1, 1, 1]);

hold off;

%-------------------------------------------------------------------------- 
%13. Descriptive analysis of all the grids.
% Calculate Mean, Median, Standard Deviation, Z-Score, Outlier for SSH Data 
%-------------------------------------------------------------------------- 

% Initialize variables to store statistics for each grid cell
mean_ssh = nan(size(ssh_grid));
median_ssh = nan(size(ssh_grid));
std_ssh = nan(size(ssh_grid));
min_ssh = nan(size(ssh_grid));
max_ssh = nan(size(ssh_grid));
all_total_values = nan(size(ssh_grid)); 
all_ssh_values = cell(size(ssh_grid));
outliers_quant = cell(size(ssh_grid));
all_num_outliers_quant = nan(size(ssh_grid));
all_outlier_percentage_quant = nan(size(ssh_grid));
all_z_scores = cell(size(ssh_grid));
outliers_z = cell(size(ssh_grid));
all_num_outliers_z = nan(size(ssh_grid)); 
all_outlier_percentage_z = nan(size(ssh_grid));
% outlier_grid = false(size(ssh_grid));

% Loop through each grid cell to calculate statistics
for k = 1:numel(ssh_grid)
    % Get SSH values in the current grid cell
    ssh_values = ssh_all((lon_all >= lon_grid(k) - 0.06) & (lon_all < lon_grid(k) + 0.06) & ...
                         (lat_all >= lat_grid(k) - 0.06) & (lat_all < lat_grid(k) + 0.06));

    % Store SSH values for the current grid cell
    all_ssh_values{k} = ssh_values;

    if ~isempty(ssh_values)
        % Calculate statistics
        mean_ssh(k) = mean(ssh_values);
        median_ssh(k) = median(ssh_values);
        std_ssh(k) = std(ssh_values);
        min_ssh(k) = min(ssh_values);
        max_ssh(k) = max(ssh_values);
        
        % Identify outliers using the 1.5 * IQR rule
        q1 = quantile(ssh_values, 0.25); % Q1 for current grid cell
        q3 = quantile(ssh_values, 0.75); % Q3 for current grid cell
        iqr = q3 - q1; % Interquartile Range
        lower_bound = q1 - 1.5 * iqr;
        upper_bound = q3 + 1.5 * iqr;
        
        % Store outliers in the corresponding cell
        outliers_quant{k} = ssh_values(ssh_values < lower_bound | ssh_values > upper_bound);
        
        % Calculate the number of outliers
        num_outliers_quant = numel(outliers_quant{k});
        all_num_outliers_quant(k) = num_outliers_quant; % Store number of outliers for this grid
        
        % Total values in the grid cell
        total_values = numel(ssh_values);
        all_total_values(k) = total_values; % Store total values for this grid

        % Calculate and store outlier percentage using IQR method
        if total_values > 0
            outlier_percentage_quant = (num_outliers_quant / total_values) * 100;
        end

        all_outlier_percentage_quant(k) = outlier_percentage_quant; % Store outlier percentage for this grid

        % Calculate Z-scores for SSH values
        z_scores = (ssh_values - mean_ssh(k)) / std_ssh(k);
        all_z_scores{k} = z_scores;  % Store Z-scores for current grid cell
        
        % Identify outliers based on Z-score threshold of 2.5
        outlier_indices = abs(z_scores) > 2.5;  % Find indices of outliers
        outliers_z{k} = ssh_values(outlier_indices);  % Store outliers in the corresponding cell
        
        % Calculate the number of outliers based on Z-scores
        num_outliers_z = numel(outliers_z{k});  % Make sure to use outliers_z here
        all_num_outliers_z(k) = num_outliers_z; % Store number of outliers for this grid
       
        % Calculate and store outlier percentage using Z-score method
        if total_values > 0
            outlier_percentage_z = (num_outliers_z / total_values) * 100;
        end

        all_outlier_percentage_z(k) = outlier_percentage_z; % Store outlier percentage for this grid

    end
end
       
disp('Statistics for SSH data across all grids have been calculated and stored.');


%-------------------------------------------------------------------------- 
%14. Plot outlier percentge of ssh in all grids
%-------------------------------------------------------------------------- 


% Plot Z-score Outlier Percentage on a 2D spatial grid
figure;

hold on;
imagesc(lon_grid(1, :), lat_grid(:, 1), all_outlier_percentage_z);
set(gca, 'YDir', 'normal'); % Flip Y-axis to have latitudes in ascending order
colormap(flipud(bone)); 
cbar = colorbar;
xlabel('Longitude (Degrees)');
ylabel('Latitude (Degrees)');
title('Outlier Percentage (Z-score Method)');

% Set color limits
caxis([min(all_outlier_percentage_z(:)), max(all_outlier_percentage_z(:))]);
ylabel(cbar, 'Grid Outliers in %');

% Ensure equal scaling on both axes
axis equal;
daspect([1, 1, 1]);

grid on;

hold off;


%-------------------------------------------------------------------------- 
%15. Calculate slopes in all ascending passes 
%-------------------------------------------------------------------------- 

% Preallocate cells for slope data
slope_data = cell(size(ascend));

% Topex Ellipsoid Parameters
a = 6378136.3;       % Semi-major axis (meters)
f = 0.0033528;       % flattening
e2 = 2 * f - f^2; % Eccentricity squared

% Loop over passes and cycles to calculate slopes for given SSH values
for j = 1:size(ascend, 2)
    for i = 1:size(ascend, 1)
        if ~isempty(ascend{i, j})
            latitudes = ascend{i, j}.latitude;
            longitudes = ascend{i, j}.longitude;
            ssh_values = ascend{i, j}.ssh;
            
            % Preallocate slope, latitude and longitude values
            slopes = NaN(length(ssh_values) - 1, 1);
            mid_latitudes = NaN(length(ssh_values) - 1, 1);
            mid_longitudes = NaN(length(ssh_values) - 1, 1);
            slope_distances = NaN(length(ssh_values) - 1, 1); % New variable for slope distance
            azimuth = NaN(length(ssh_values) - 1, 1);
            
            % Loop to calculate all slopes in each pass
            for k = 1:length(ssh_values) - 1
                lat1 = latitudes(k);
                lon1 = longitudes(k);
                lat2 = latitudes(k + 1);
                lon2 = longitudes(k + 1);
                
                % Convert latitudes and longitudes to radians for azimuth calculation
                phi_i = deg2rad(lat1); 
                phi_j = deg2rad(lat2);  
                lambda_i = deg2rad(lon1);
                lambda_j = deg2rad(lon2);
                delta_phi = phi_j - phi_i;
                delta_lambda = lambda_j - lambda_i;

                % Calculate midpoint latitude and longitude
                mid_latitudes(k) = (lat1 + lat2) / 2;
                mid_longitudes(k) = (lon1 + lon2) / 2;

                % Calculate the Haversine distance
                a_hav = sin(delta_phi / 2)^2 + cos(phi_i) * cos(phi_j) * sin(delta_lambda / 2)^2;
                c = 2 * atan2(sqrt(a_hav), sqrt(1 - a_hav));
                distance = a * c; % Distance in meters using the semi-major axis of Topex

                % % Calculate slope distance (3D Euclidean distance using ECEF)
                % N1 = a / sqrt(1 - e2 * sin(phi_i)^2); % Radius of curvature
                % N2 = a / sqrt(1 - e2 * sin(phi_j)^2); % Radius of curvature
                % 
                % % ECEF coordinates for both points
                % X1 = (N1 + ssh_values(k)) * cos(phi_i) * cos(lambda_i);
                % Y1 = (N1 + ssh_values(k)) * cos(phi_i) * sin(lambda_i);
                % Z1 = ((1 - e2) * N1 + ssh_values(k)) * sin(phi_i);
                % 
                % X2 = (N2 + ssh_values(k+1)) * cos(phi_j) * cos(lambda_j);
                % Y2 = (N2 + ssh_values(k+1)) * cos(phi_j) * sin(lambda_j);
                % Z2 = ((1 - e2) * N2 + ssh_values(k+1)) * sin(phi_j);
                % 
                % % Calculate the slope distance (3D Euclidean distance)
                % distance = sqrt((X2 - X1)^2 + (Y2 - Y1)^2 + (Z2 - Z1)^2); % in meters
                
                % Calculate slope (m / degree)
                slopes(k) = (ssh_values(k + 1) - ssh_values(k)) / distance; % no units
                
                
                % Azimuth calculation
                alpha_num = cos(phi_j) * sin(lambda_j - lambda_i);
                alpha_denom = (cos(phi_i) * sin(phi_j) - sin(phi_i) * cos(phi_j) * cos(lambda_j - lambda_i));
                azimuth(k) = atan2(alpha_num, alpha_denom);  % Azimuth in radians
                 
            end
            
            % Store the calculated data for the current pass
            slope_data{i, j} = struct('mid_latitude', mid_latitudes, ...
                                       'mid_longitude', mid_longitudes, ...
                                       'slopes', slopes, ...
                                       'azimuth', azimuth, ...
                                       'slope_distances', distance);  % Include slope distance
        end
    end
end

disp('Slopes and corresponding midpoint coordinates have been successfully calculated and stored for all cycles and passes.');


%--------------------------------------------------------------------------
%16. Create 100 * 100 grid of lat, long, slope and azimuth data from all cycles
%and passes
%--------------------------------------------------------------------------

% Initialize arrays to collect all latitudes, longitudes, slopes, and azimuth values across all cycles
latitude_all = [];
longitude_all = [];
slopes_all = [];
azimuth_all = [];

% Loop through all slope data passes and cycles
for j = 1:size(slope_data, 2)  % Loop through all passes
    for i = 1:size(slope_data, 1)  % Loop through all cycles
        % Check if data exists in the current cell
        if ~isempty(slope_data{i, j})
            % Append current cycle data to the main arrays
            latitude_all = [latitude_all; double(slope_data{i, j}.mid_latitude)];
            longitude_all = [longitude_all; double(slope_data{i, j}.mid_longitude)];
            slopes_all = [slopes_all; double(slope_data{i, j}.slopes)];
            azimuth_all = [azimuth_all; double(slope_data{i, j}.azimuth)];
        end
    end
end

% Create a grid for longitude and latitude
[longitude_grid, latitude_grid] = meshgrid(linspace(min(longitude_all), max(longitude_all), 100), ...
                                           linspace(min(latitude_all), max(latitude_all), 100));

% Initialize a cell array to store slopes and azimuths in each grid cell
slope_grid_raw = cell(size(latitude_grid));

% Populate the grid with slopes and azimuths
for k = 1:length(slopes_all)
    [~, longitude_idx] = min(abs(longitude_grid(1, :) - longitude_all(k)));  
    [~, latitude_idx] = min(abs(latitude_grid(:, 1) - latitude_all(k))); 

    if isempty(slope_grid_raw{latitude_idx, longitude_idx})
        % Initialize a struct for storing slopes and azimuths in this cell
        slope_grid_raw{latitude_idx, longitude_idx} = struct('slopes', slopes_all(k), 'azimuth', azimuth_all(k));
    else
        % Append the new slope and azimuth values to existing data in this cell
        slope_grid_raw{latitude_idx, longitude_idx}.slopes = ...
            [slope_grid_raw{latitude_idx, longitude_idx}.slopes; slopes_all(k)];
        slope_grid_raw{latitude_idx, longitude_idx}.azimuth = ...
            [slope_grid_raw{latitude_idx, longitude_idx}.azimuth; azimuth_all(k)];
    end
end

% Check whether all data are mapped or not
total_elements_count = sum(cellfun(@(x) numel(x.slopes), slope_grid_raw(~cellfun('isempty', slope_grid_raw))), 'all');
disp(['Total number of elements in the grid: ', num2str(total_elements_count)]);

total_values = numel(slopes_all);
disp(['Total number of values in slopes_all: ', num2str(total_values)]);


%--------------------------------------------------------------------------
%17. Calculate xi and eta for avaiable slope cells in 100 * 100 grid
%--------------------------------------------------------------------------

function adjusted_data = find_xi_eta(slope_grid_raw)
    % Preallocate cell to store adjusted results
    adjusted_data = cell(size(slope_grid_raw));
    
    % Loop over each cell in slope_grid_raw
    for i = 1:numel(slope_grid_raw)
        if ~isempty(slope_grid_raw{i}) && length(slope_grid_raw{i}.slopes) > 1
            % Extract slopes and azimuth angles
            slopes = slope_grid_raw{i}.slopes;
            alphas = slope_grid_raw{i}.azimuth;
            
            % Design matrix A, observation vector L
            A = [cos(alphas), sin(alphas)];  % Use sind and cosd for degree inputs
            L = slopes;
            
            % Solve for ξ and η using least squares
            N = A' * A;
            U = A' * L;
            x = inv(N) * U;  % Compute directly without handling singularity
            
            % Calculate residuals (ε)
            epsilon = L - (A * x);
            
            % Store results in a struct for each cell
            adjusted_data{i} = struct('xi', x(1), 'eta', x(2), 'residuals', epsilon);
        else
            % Skip cells with a single slope value or empty data
            adjusted_data{i} = struct('xi', NaN, 'eta', NaN, 'residuals', NaN);
        end
    end
end

% Call the function with slope_grid_raw and store the result
adjusted_data = find_xi_eta(slope_grid_raw);

disp("DOV's calculated successfully for all possible grid cells");


%--------------------------------------------------------------------------
%18. Analysing and cleaning raw DOV (xi, eta) data over 100 * 100 grid
%--------------------------------------------------------------------------

% Call the function to obtain adjusted_data
adjusted_data = find_xi_eta(slope_grid_raw);

% Extract longitude and latitude
lon_values = [];
lat_values = [];
xi_values = [];
eta_values = [];

% Populate arrays for scatter plot
for i = 1:numel(adjusted_data)
    if ~isnan(adjusted_data{i}.xi) && ~isnan(adjusted_data{i}.eta)
        % Convert linear index to 2D grid coordinates
        [row, col] = ind2sub(size(longitude_grid), i);
        lon_values(end + 1) = longitude_grid(row, col);
        lat_values(end + 1) = latitude_grid(row, col);
        xi_values(end + 1) = adjusted_data{i}.xi;
        eta_values(end + 1) = adjusted_data{i}.eta;
    end
end


% Line plot for all xi_values
figure;
plot(xi_values, 'b-', 'LineWidth', 1.5); % Line plot for cleaned xi values
title('Line Plot of E-W DOV \xi Values (All)');
xlabel('Index');
ylabel('\xi (rad)');
grid on;

% Line plot for all eta_values
figure;
plot(eta_values, 'g-', 'LineWidth', 1.5); % Line plot for cleaned eta values
title('Line Plot of N-S DOV \eta Values (All)');
xlabel('Index');
ylabel('\eta (rad)');
grid on;

% DOV data analysis and removing outlier values
count_out_xi = sum(xi_values > 0.1); % Count xi_values are greater than 0.1
count_total_xi = numel(xi_values); % Count total xi_values
DOV_out_perc_xi = (count_out_xi/count_total_xi) * 100; % DOV outlier percentage xi


count_out_eta = sum(eta_values > 0.1); % Count eta_values are greater than 0.1
count_total_eta = numel(eta_values); % Count total eta_values
DOV_out_perc_eta =  (count_out_eta/count_total_eta) * 100; % DOV Outlier percentage

valid_indices = (abs(xi_values) <= 0.1) & (abs(eta_values) <= 0.1); % Remove values greater than 0.1 in xi_values and eta_values

% Create cleaned data
lon_values_cleaned = lon_values(valid_indices);
lat_values_cleaned = lat_values(valid_indices);
xi_values_cleaned = xi_values(valid_indices);
eta_values_cleaned = eta_values(valid_indices);

% Display the DOV analysis outcomes
disp(['Number of xi values greater than 0.1: ', num2str(count_out_xi)]);
disp(['Number of eta values greater than 0.1: ', num2str(count_out_eta)]);
disp(['Total number of xi values: ', num2str(count_total_xi)]);
disp(['Total number of eta values: ', num2str(count_total_eta)]);
disp(['Percentage of values removed in xi: ', num2str(DOV_out_perc_xi)]);
disp(['Percentage of values removed in eta: ', num2str(DOV_out_perc_eta)]);

% Line plot for xi_values_cleaned
figure;
plot(xi_values_cleaned, 'b-', 'LineWidth', 1.5); % Line plot for cleaned xi values
title('Line Plot of E-W DOV \xi Values (Cleaned Data)');
xlabel('Index');
ylabel('\xi (rad)');
grid on;

% Line plot for eta_values_cleaned
figure;
plot(eta_values_cleaned, 'g-', 'LineWidth', 1.5); % Line plot for cleaned eta values
title('Line Plot of N-S DOV \eta Values (Cleaned Data)');
xlabel('Index');
ylabel('\eta (rad)');
grid on;


%%
%--------------------------------------------------------------------------
%19. Plot raw DOV (xi, eta) data over 100 * 100 grid
%--------------------------------------------------------------------------

addpath('E:\SEM 1\physical geodesy\colorbrew');


colormap_xi = brewermap([], 'RdBu'); % Selected Red-Blue diverging color map
colormap_eta = brewermap([], 'PiYG'); % Selected Purple-Green diverging color map

% Plotting 2D scatter for xi (cleaned data)
figure;
scatter(lon_values_cleaned, lat_values_cleaned, 18, xi_values_cleaned, 'filled'); % 18 is the size of the markers
colormap(colormap_xi); % Apply the diverging color map
colorbar; % Add colorbar to indicate the value of xi
caxis([min(xi_values_cleaned), max(xi_values_cleaned)]); % Set color limits based on cleaned data range
title('2D Scatter Plot of E-W DOV \xi Values (Cleaned Data)');
xlabel('Longitude (degrees)');
ylabel('Latitude (degrees)');
ylabel(colorbar, '\xi (rad)'); % Label for colorbar

% Plotting 2D scatter for eta (cleaned data)
figure;
scatter(lon_values_cleaned, lat_values_cleaned, 18, eta_values_cleaned, 'filled'); % 18 is the size of the markers
colormap(colormap_eta); % Apply the diverging color map
colorbar; % Add colorbar to indicate the value of eta
caxis([min(eta_values_cleaned), max(eta_values_cleaned)]); % Set color limits based on cleaned data range
title('2D Scatter Plot of N-S DOV \eta Values (Cleaned Data)');
xlabel('Longitude (degrees)');
ylabel('Latitude (degrees)');
ylabel(colorbar, '\eta (rad)'); % Label for colorbar

%--------------------------------------------------------------------------
%20. Interpolate cleaned DOV (xi, eta) data over 100 * 100 grid
%--------------------------------------------------------------------------

lon_grid_flat = longitude_grid(:);
lat_grid_flat = latitude_grid(:);
grid_points = [lon_grid_flat, lat_grid_flat];

gpr_model_xi = fitrgp([lon_values_cleaned', lat_values_cleaned'], xi_values_cleaned', ...
                      'KernelFunction', 'squaredexponential');

gpr_model_eta = fitrgp([lon_values_cleaned', lat_values_cleaned'], eta_values_cleaned', ...
                       'KernelFunction', 'squaredexponential');

xi_interpolated = predict(gpr_model_xi, grid_points);
eta_interpolated = predict(gpr_model_eta, grid_points);

xi_grid = reshape(xi_interpolated, size(longitude_grid));

disp(max(max(xi_grid)));
disp(min(min(xi_grid)));

eta_grid = reshape(eta_interpolated, size(longitude_grid));

%--------------------------------------------------------------------------
%21. Plot interpolated cleaned DOV (xi, eta) data over 100 * 100 grid
%--------------------------------------------------------------------------

% Plotting interpolated xi values
figure;
imagesc(longitude_grid(1,:), latitude_grid(:,1), xi_grid);
colormap(brewermap([], 'RdBu'));
colorbar_xi_int = colorbar; % Create colorbar
ylabel(colorbar_xi_int, '\xi (radians)'); % Label colorbar with unit in radians
title('Interpolated E-W DOV (\xi) Values');
xlabel('Longitude (degrees)');
ylabel('Latitude (degrees)');
axis equal; % Set axis to equal for geographic scaling
set(gca, 'YDir', 'normal'); % Ensure latitude increases upwards

% Plotting interpolated eta values
figure;
imagesc(longitude_grid(1,:), latitude_grid(:,1), eta_grid);
colormap(brewermap([], 'PiYG'));
colorbar_eta_int = colorbar; % Create colorbar
ylabel(colorbar_eta_int, '\eta (radians)'); % Label colorbar with unit in radians
title('Interpolated N-S DOV (\eta) Values');
xlabel('Longitude (degrees)');
ylabel('Latitude (degrees)');
axis equal; % Set axis to equal for geographic scaling
set(gca, 'YDir', 'normal'); % Ensure latitude increases upwards

%--------------------------------------------------------------------------
%22. Find Normal Gravity for 100 * 100 grid
%--------------------------------------------------------------------------

b = a * (1-f);       % semi minor axis
gamma_a = 9.7803253359;   % Gravity at the equator (m/s²)
gamma_b = 9.8321849378;   % Gravity at the poles (m/s²)

% Initialize gamma_grid as a numeric array to store normal gravity values
gamma_grid = zeros(size(latitude_grid));

% Calculate normal gravity for each point in the grid using a loop
for i = 1:size(latitude_grid, 1)
    for j = 1:size(latitude_grid, 2)
        % Extract latitude in degrees and convert to radians
        lat_deg = latitude_grid(i, j);  % Use regular indexing instead of brace indexing
        lat_rad = deg2rad(lat_deg);
        
        % Calculate normal gravity using the Somigliana-Pizzetti formula
        gamma_grid(i, j) = (a * gamma_a * cos(lat_rad)^2 + b * gamma_b * sin(lat_rad)^2) / ...
                           sqrt(a^2 * cos(lat_rad)^2 + b^2 * sin(lat_rad)^2);
    end
end

% Display the gravity grid as a surface plot
figure;
imagesc(longitude_grid(1,:), latitude_grid(:,1), gamma_grid); % Plot using regular indexing for latitude_grid
colormap(brewermap([], 'OrRd'));
colorbar_label = colorbar;
xlabel('Longitude (degrees)');
ylabel('Latitude (degrees)');
ylabel(colorbar_label, 'Normal Gravity (\gamma) [m/s²]'); % Add units to colorbar
title('Normal Gravity (\gamma) over Grid [m/s²]');
axis equal;

% Invert the y-axis for proper geographic display
set(gca, 'YDir', 'normal');

%--------------------------------------------------------------------------
%23. Find Gravity Anomaly for 100 * 100 grid
 


% Constants
G = 6.67430e-11;         % Gravitational constant (m^3 kg^-1 s^-2)
rho_water = 1025;        % Density of seawater (kg/m^3)
gamma = 9.81;            % Approximate gravitational acceleration (m/s^2)

% Assume xi_grid and eta_grid have been defined from previous interpolation
% Define grid size
[Ny, Nx] = size(xi_grid); 

% Compute spatial resolutions
x_length = max(longitude_grid(:)) - min(longitude_grid(:));
y_length = max(latitude_grid(:)) - min(latitude_grid(:));
dx = x_length / (Nx - 1);
dy = y_length / (Ny - 1);

% Create wave numbers for FFT
k_x = 2 * pi * (0:(Nx-1)) / x_length;
k_y = 2 * pi * (0:(Ny-1)) / y_length;
[kx, ky] = meshgrid(k_x, k_y);

% Adjust wave numbers for negative frequencies
kx_shifted = ifftshift(kx - pi / dx);
ky_shifted = ifftshift(ky - pi / dy);

% Calculate |k| in Fourier domain
k_magnitude = sqrt(kx_shifted.^2 + ky_shifted.^2);

% Set k_magnitude to a small number where zero to avoid division by zero
k_magnitude(k_magnitude == 0) = 1e-10;

% FFT of DOV components
xi_fft = fft2(xi_grid);
eta_fft = fft2(eta_grid);

% Inverse Vening Meinesz formula in Fourier domain
gravity_anomaly_fft = (1i ./ k_magnitude) .* (kx_shifted .* xi_fft + ky_shifted .* eta_fft);

% Convert gravity anomaly back to spatial domain using inverse FFT
gravity_anomaly_grid = real(ifft2(gravity_anomaly_fft));

% Convert gravity anomaly from m/s^2 to mGal
gravity_anomaly_grid_mGal = gravity_anomaly_grid * 1e5;

% Plot the gravity anomaly grid
figure;
imagesc(longitude_grid(1,:), latitude_grid(:,1), gravity_anomaly_grid_mGal);
colormap(brewermap([], 'RdYlBu'));
colorbar_label = colorbar;
ylabel(colorbar_label, '\Delta g (mGal)');
title('Marine Gravity Anomaly (\Delta g)');
xlabel('Longitude (degrees)');
ylabel('Latitude (degrees)');
axis equal;
set(gca, 'YDir', 'normal');
%%

