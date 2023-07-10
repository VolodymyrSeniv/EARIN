% Predicate to convert date to timestamp
date_to_timestamp(Date, Timestamp) :-
    date_time_stamp(Date, Timestamp).

% Predicate to convert timestamp to date
timestamp_to_date(Timestamp, Date) :-
    stamp_date_time(Timestamp, DateTime, 'UTC'),
    date_time_value(date, DateTime, Date).

% Predicate to add days to a given date
add_date(Date, DaysToAdd, NewDate) :-
    DaysToAdd =< 365,
    date_to_timestamp(Date, Timestamp),
    NewTimestamp is Timestamp + DaysToAdd * 86400,  % Convert days to seconds
    timestamp_to_date(NewTimestamp, NewDate),
    NewDate = date(Year, _, _),
    (Year =:= 2023 ; Year =:= 2024). % Ensure the year is either 2023 or 2024

% Predicate to subtract days from a given date
sub_date(Date, DaysToSubtract, NewDate) :-
    DaysToSubtract =< 365,
    date_to_timestamp(Date, Timestamp),
    NewTimestamp is Timestamp - DaysToSubtract * 86400,  % Convert days to seconds
    timestamp_to_date(NewTimestamp, NewDate),
    NewDate = date(Year, _, _),
    (Year =:= 2022 ; Year =:= 2023). % Ensure the year is either 2022 or 2023

% Predicate to validate date
validate_date(date(Year, Month, Day)) :-
    Year =:= 2023,
    Month >= 1, Month =< 12,
    days_in_month(Month, DaysInMonth),
    Day >= 1, Day =< DaysInMonth.

% Facts for days in each month
days_in_month(1, 31). % January
days_in_month(2, 28). % February (non-leap year)
days_in_month(3, 31). % March
days_in_month(4, 30). % April
days_in_month(5, 31). % May
days_in_month(6, 30). % June
days_in_month(7, 31). % July
days_in_month(8, 31). % August
days_in_month(9, 30). % September
days_in_month(10, 31). % October
days_in_month(11, 30). % November
days_in_month(12, 31). % December
