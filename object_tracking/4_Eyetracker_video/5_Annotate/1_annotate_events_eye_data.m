function annotate_eye_data()
    events = readtable('events.csv', 'VariableNamingRule', 'preserve');
    gaze = readtable('gaze.csv', 'VariableNamingRule', 'preserve');
    blinks = readtable('blinks.csv', 'VariableNamingRule', 'preserve');
    fixations = readtable('fixations.csv', 'VariableNamingRule', 'preserve');
    saccades = readtable('saccades.csv', 'VariableNamingRule', 'preserve');
    imu = readtable('imu.csv', 'VariableNamingRule', 'preserve');


    % Sort events by 'timestamp [ns]'
    events_sorted = sortrows(events, 'timestamp [ns]');
    
    % Annotate the gaze data with event names based on 'timestamp [ns]'
    gaze.event_name = arrayfun(@(x) find_event_for_timestamp(x, events_sorted), gaze.('timestamp [ns]'), 'UniformOutput', false);

    % Annotate blinks, fixations, saccades, and imu data
    blinks.event_name = arrayfun(@(x) find_event_for_timestamp(x, events_sorted), blinks.('start timestamp [ns]'), 'UniformOutput', false);
    fixations.event_name = arrayfun(@(x) find_event_for_timestamp(x, events_sorted), fixations.('start timestamp [ns]'), 'UniformOutput', false);
    saccades.event_name = arrayfun(@(x) find_event_for_timestamp(x, events_sorted), saccades.('start timestamp [ns]'), 'UniformOutput', false);
    imu.event_name = arrayfun(@(x) find_event_for_timestamp(x, events_sorted), imu.('timestamp [ns]'), 'UniformOutput', false);

    writetable(gaze, 'annotated_gaze.csv');
    writetable(blinks, 'annotated_blinks.csv');
    writetable(fixations, 'annotated_fixations.csv');
    writetable(saccades, 'annotated_saccades.csv');
    writetable(imu, 'annotated_imu.csv');


  
end

function event_name = find_event_for_timestamp(timestamp, events_sorted)
    % Find the name of the event occurring at or just after (if it is '') a given timestamp
    idx = find(events_sorted.('timestamp [ns]') >= timestamp, 1);
    
    if ~isempty(idx)
        if idx == 1
            event_name = events_sorted.name{idx};
        else
            if events_sorted.('timestamp [ns]')(idx-1) <= timestamp
                event_name = events_sorted.name{idx-1};
            else
                event_name = '';
            end
        end
    else
        if events_sorted.('timestamp [ns]')(end) <= timestamp
            event_name = events_sorted.name{end};
        else
            event_name = '';
        end
    end
end

