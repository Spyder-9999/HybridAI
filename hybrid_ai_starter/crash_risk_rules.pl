/*
    File: crash_risk_rules.pl
    Student: dobi0016
    Course: COMP3742/9742 - Artificial Intelligence

    Purpose:
    Combine machine-learning classification facts with symbolic
    rules about traffic volume, pedestrian crossings and data quality.

    IMPORTANT:
    These are illustrative prototype rules, not validated road-safety
    standards. The ML model classifies historical crash labels; it does
    not independently predict future crashes.
*/


% ---------------------------------------------------------
% 1. Load generated facts
% ---------------------------------------------------------

load_site_facts(FactsFile) :-
    consult(FactsFile).

load_ml_facts(FactsFile) :-
    consult(FactsFile).


% ---------------------------------------------------------
% 2. Check data availability and freshness
% ---------------------------------------------------------

volume_stale_site(Site) :-
    current_predicate(volume_stale/1),
    volume_stale(Site).


% Missing traffic volume.
data_issue(Site, missing_traffic_volume) :-
    site(Site),
    \+ traffic_volume(Site, _).

% Traffic volume is unavailable according to its availability flag.
data_issue(Site, traffic_volume_unavailable) :-
    site(Site),
    \+ has_volume(Site),
    \+ traffic_volume(Site, _).

% Traffic volume is recorded as stale.
data_issue(Site, traffic_volume_stale) :-
    site(Site),
    volume_stale_site(Site).

% Freshness information is missing or unknown.
data_issue(Site, traffic_freshness_unknown) :-
    site(Site),
    \+ volume_current(Site),
    \+ volume_stale_site(Site).

% Traffic volume exists, but the availability flag is missing or false.
data_issue(Site, volume_availability_unknown) :-
    site(Site),
    traffic_volume(Site, Volume),
    number(Volume),
    \+ has_volume(Site).

% Availability and freshness flags indicate valid data, but
% the traffic-volume fact is missing.
data_issue(Site, inconsistent_volume_data) :-
    site(Site),
    has_volume(Site),
    volume_current(Site),
    \+ traffic_volume(Site, _).

% A traffic-volume value exists but is not numeric.
data_issue(Site, invalid_traffic_volume) :-
    site(Site),
    traffic_volume(Site, Volume),
    \+ number(Volume).


% A site is usable only when traffic data is present, numeric,
% current and not affected by a recognised data-quality issue.
usable_site(Site) :-
    site(Site),
    has_volume(Site),
    volume_current(Site),
    traffic_volume(Site, Volume),
    number(Volume),
    \+ data_issue(Site, _).


% ---------------------------------------------------------
% 3. Define contextual conditions
% ---------------------------------------------------------

% Illustrative threshold only; not an official safety standard.
higher_traffic_exposure(Site) :-
    traffic_volume(Site, Volume),
    number(Volume),
    Volume >= 20000.


pedestrian_crossing_present(Site) :-
    pedestrian_crossings(Site, Count),
    number(Count),
    Count > 0.


school_crossing_present(Site) :-
    school_crossings(Site, Count),
    number(Count),
    Count > 0.


% Any one of these conditions can warrant contextual consideration.
contextual_concern(Site) :-
    higher_traffic_exposure(Site).

contextual_concern(Site) :-
    pedestrian_crossing_present(Site).

contextual_concern(Site) :-
    school_crossing_present(Site).


% ---------------------------------------------------------
% 4. Read and validate machine-learning output
% ---------------------------------------------------------

% Expected generated facts:
%
% ml_classification(Site, historical_fsi_present).
% ml_positive_probability(Site, Probability).
%
% Probability represents the estimated probability of the positive
% historical any_fsi class, not the probability of a future crash.


ml_prediction_available(Site) :-
    ml_classification(Site, Class),
    member(Class, [
        historical_fsi_present,
        historical_fsi_absent
    ]),
    ml_positive_probability(Site, Probability),
    number(Probability),
    Probability >= 0.0,
    Probability =< 1.0.


ml_positive_class(Site) :-
    ml_classification(Site, historical_fsi_present).


ml_negative_class(Site) :-
    ml_classification(Site, historical_fsi_absent).


% Descriptive probability predicate.
% The classification supplied by the ML model remains authoritative.
ml_probability_at_least_50_percent(Site) :-
    ml_positive_probability(Site, Probability),
    number(Probability),
    Probability >= 0.5,
    Probability =< 1.0.


% ---------------------------------------------------------
% 5. Generate the hybrid assessment
% ---------------------------------------------------------

% Rule 1: Data-quality issues take priority.
% Return the first applicable data issue.
site_assessment(Site, data_review, Explanation) :-
    data_issue(Site, Explanation),
    !.


% Rule 2: Positive historical ML classification combined with
% at least one contextual concern warrants a priority review.
site_assessment(
    Site,
    priority_safety_review,
    historical_fsi_with_contextual_concern
) :-
    usable_site(Site),
    ml_prediction_available(Site),
    ml_positive_class(Site),
    contextual_concern(Site),
    !.


% Rule 3: Positive historical ML classification without an
% additional contextual trigger still warrants a safety review.
site_assessment(
    Site,
    safety_review,
    ml_positive_historical_class
) :-
    usable_site(Site),
    ml_prediction_available(Site),
    ml_positive_class(Site),
    !.


% Rule 4: Negative historical ML classification does not mean
% that the site is safe. Review any additional contextual concerns.
site_assessment(
    Site,
    contextual_review,
    contextual_concern_with_negative_ml_class
) :-
    usable_site(Site),
    ml_prediction_available(Site),
    ml_negative_class(Site),
    contextual_concern(Site),
    !.


% Rule 5: A negative historical ML classification with no
% defined contextual trigger receives routine monitoring.
site_assessment(
    Site,
    routine_monitoring,
    negative_ml_class_no_contextual_trigger
) :-
    usable_site(Site),
    ml_prediction_available(Site),
    ml_negative_class(Site),
    \+ contextual_concern(Site),
    !.


% Rule 6: If ML output is unavailable, contextual concerns
% can still trigger a review using the available site facts.
site_assessment(
    Site,
    contextual_review,
    ml_unavailable_with_contextual_concern
) :-
    usable_site(Site),
    \+ ml_prediction_available(Site),
    contextual_concern(Site),
    !.


% Rule 7: If ML output is unavailable and no contextual trigger
% is identified, use routine review rather than assuming safety.
site_assessment(
    Site,
    routine_review,
    ml_unavailable_no_contextual_trigger
) :-
    usable_site(Site),
    \+ ml_prediction_available(Site),
    \+ contextual_concern(Site),
    !.


% ---------------------------------------------------------
% 6. Compatibility predicates
% ---------------------------------------------------------

% Preserve recommendation/3 for existing queries.
% The assessment rules above are the single source of decisions.
recommendation(Site, Action, Explanation) :-
    site_assessment(Site, Action, Explanation).


% Preserve reason/2 for queries that need the selected explanation.
reason(Site, Explanation) :-
    site_assessment(Site, _, Explanation).