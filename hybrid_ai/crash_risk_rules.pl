/*
    Student Name: Patrick
    Student FAN: dobi0016
    File: prepare_site.py
    Date: 03-11-2026

    Description:
    Combine machine-learning classification facts with symbolic
    reasoning about traffic volume, pedestrian crossings and
    data quality.

    The rules demonstrate how a symbolic inference engine can
    interpret machine-learning output alongside contextual facts.
*/


% ---------------------------------------------------------
% 1. Load generated facts
% ---------------------------------------------------------

load_site_facts(FactsFile) :-
    consult(FactsFile).

load_ml_facts(FactsFile) :-
    consult(FactsFile).


% ---------------------------------------------------------
% 2. Traffic data validation
% ---------------------------------------------------------

% Illustrative traffic-volume threshold.
% This is configurable within the rule base.

high_traffic_threshold(20000).


% Check whether the traffic-volume data is marked as stale.

volume_stale_site(Site) :-
    current_predicate(volume_stale/1),
    volume_stale(Site).


% Traffic volume is missing entirely.

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


% Traffic volume exists, but its availability flag is missing or false.

data_issue(Site, volume_availability_unknown) :-
    site(Site),
    traffic_volume(Site, Volume),
    number(Volume),
    \+ has_volume(Site).


% Availability and freshness flags indicate valid data,
% but the traffic-volume fact is missing.

data_issue(Site, inconsistent_volume_data) :-
    site(Site),
    has_volume(Site),
    volume_current(Site),
    \+ traffic_volume(Site, _).


% Traffic volume exists but contains a non-numeric value.

data_issue(Site, invalid_traffic_volume) :-
    site(Site),
    traffic_volume(Site, Volume),
    \+ number(Volume).


% A traffic-volume value cannot be negative.

data_issue(Site, negative_traffic_volume) :-
    site(Site),
    traffic_volume(Site, Volume),
    number(Volume),
    Volume < 0.


% A site can only be used when its traffic data is available,
% numeric, current and free of recognised data-quality issues.

usable_site(Site) :-
    site(Site),
    has_volume(Site),
    volume_current(Site),
    traffic_volume(Site, Volume),
    number(Volume),
    Volume >= 0,
    \+ data_issue(Site, _).


% ---------------------------------------------------------
% 3. Crossing data validation
% ---------------------------------------------------------

% Missing pedestrian crossing information.

data_issue(Site, pedestrian_crossing_data_unknown) :-
    site(Site),
    \+ pedestrian_crossings(Site, _).


% Missing school crossing information.

data_issue(Site, school_crossing_data_unknown) :-
    site(Site),
    \+ school_crossings(Site, _).


% Invalid pedestrian crossing count.

data_issue(Site, invalid_pedestrian_crossing_data) :-
    site(Site),
    pedestrian_crossings(Site, Count),
    \+ (
        number(Count),
        Count >= 0
    ).


% Invalid school crossing count.

data_issue(Site, invalid_school_crossing_data) :-
    site(Site),
    school_crossings(Site, Count),
    \+ (
        number(Count),
        Count >= 0
    ).


% ---------------------------------------------------------
% 4. Define contextual conditions
% ---------------------------------------------------------

% Higher traffic exposure is identified using the
% configurable threshold defined above.

higher_traffic_exposure(Site) :-
    traffic_volume(Site, Volume),
    number(Volume),
    high_traffic_threshold(Threshold),
    Volume >= Threshold.


% A pedestrian crossing is present when its count is positive.

pedestrian_crossing_present(Site) :-
    pedestrian_crossings(Site, Count),
    number(Count),
    Count > 0.


% A school crossing is present when its count is positive.

school_crossing_present(Site) :-
    school_crossings(Site, Count),
    number(Count),
    Count > 0.


% A contextual concern exists when at least one
% defined contextual condition is satisfied.

contextual_concern(Site) :-
    higher_traffic_exposure(Site).

contextual_concern(Site) :-
    pedestrian_crossing_present(Site).

contextual_concern(Site) :-
    school_crossing_present(Site).


% Identify the individual contextual conditions
% contributing to an assessment.

contextual_reason(Site, higher_traffic_exposure) :-
    higher_traffic_exposure(Site).

contextual_reason(Site, pedestrian_crossing_present) :-
    pedestrian_crossing_present(Site).

contextual_reason(Site, school_crossing_present) :-
    school_crossing_present(Site).


% ---------------------------------------------------------
% 5. Validate machine-learning output
% ---------------------------------------------------------

% Expected generated facts:
%
% ml_classification(Site, historical_fsi_present).
% ml_positive_probability(Site, Probability).
%
% The probability represents the estimated probability of the
% positive historical any_fsi class.


% A valid ML prediction requires an accepted classification
% and a numeric probability between 0 and 1.

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


% A positive classification is accepted only when
% the complete ML prediction is valid.

ml_positive_class(Site) :-
    ml_prediction_available(Site),
    ml_classification(Site, historical_fsi_present).


% A negative classification is accepted only when
% the complete ML prediction is valid.

ml_negative_class(Site) :-
    ml_prediction_available(Site),
    ml_classification(Site, historical_fsi_absent).


% ---------------------------------------------------------
% 6. Identify machine-learning data issues
% ---------------------------------------------------------

% Completely missing ML output is treated as unavailable.
% This allows the contextual fallback rules to operate.


% A classification exists but its value is invalid.

data_issue(Site, invalid_ml_classification) :-
    site(Site),
    ml_classification(Site, Class),
    \+ member(Class, [
        historical_fsi_present,
        historical_fsi_absent
    ]).


% A valid classification exists but its probability is missing.

data_issue(Site, ml_probability_missing) :-
    site(Site),
    ml_classification(Site, Class),
    member(Class, [
        historical_fsi_present,
        historical_fsi_absent
    ]),
    \+ ml_positive_probability(Site, _).


% A probability exists but the classification is missing.

data_issue(Site, ml_classification_missing) :-
    site(Site),
    \+ ml_classification(Site, _),
    ml_positive_probability(Site, _).


% A probability exists but is not numeric or is outside
% the accepted range of 0 to 1.

data_issue(Site, invalid_ml_probability) :-
    site(Site),
    ml_positive_probability(Site, Probability),
    \+ (
        number(Probability),
        Probability >= 0.0,
        Probability =< 1.0
    ).


% ---------------------------------------------------------
% 7. Retrieve all data-quality issues
% ---------------------------------------------------------

% Return all recognised data issues for a site.
% Sorting removes duplicate explanations and provides
% a consistent result.

site_data_issues(Site, Issues) :-
    findall(
        Issue,
        data_issue(Site, Issue),
        FoundIssues
    ),
    sort(FoundIssues, Issues).


% ---------------------------------------------------------
% 8. Generate the hybrid assessment
% ---------------------------------------------------------

% Rule 1:
% Data-quality issues take priority over assessment rules.
% The first identified issue becomes the main explanation.
% All issues can also be retrieved using site_data_issues/2.

site_assessment(Site, data_review, Explanation) :-
    data_issue(Site, Explanation),
    !.


% Rule 2:
% A positive historical ML classification combined with
% at least one contextual concern warrants priority review.

site_assessment(
    Site,
    priority_safety_review,
    historical_fsi_with_contextual_concern
) :-
    usable_site(Site),
    ml_positive_class(Site),
    contextual_concern(Site),
    !.


% Rule 3:
% A positive historical ML classification without an
% additional contextual trigger warrants a safety review.

site_assessment(
    Site,
    safety_review,
    ml_positive_historical_class
) :-
    usable_site(Site),
    ml_positive_class(Site),
    !.


% Rule 4:
% A negative historical ML classification does not mean
% that a site is safe. Contextual concerns still warrant review.

site_assessment(
    Site,
    contextual_review,
    contextual_concern_with_negative_ml_class
) :-
    usable_site(Site),
    ml_negative_class(Site),
    contextual_concern(Site),
    !.


% Rule 5:
% A negative historical ML classification without a defined
% contextual trigger receives routine monitoring.

site_assessment(
    Site,
    routine_monitoring,
    negative_ml_class_no_contextual_trigger
) :-
    usable_site(Site),
    ml_negative_class(Site),
    \+ contextual_concern(Site),
    !.


% Rule 6:
% If ML output is completely unavailable, contextual concerns
% can still trigger a review using the available site facts.

site_assessment(
    Site,
    contextual_review,
    ml_unavailable_with_contextual_concern
) :-
    usable_site(Site),
    \+ ml_prediction_available(Site),
    \+ ml_output_issue(Site),
    contextual_concern(Site),
    !.


% Rule 7:
% If ML output is completely unavailable and no contextual
% trigger is identified, use routine review.

site_assessment(
    Site,
    routine_review,
    ml_unavailable_no_contextual_trigger
) :-
    usable_site(Site),
    \+ ml_prediction_available(Site),
    \+ ml_output_issue(Site),
    \+ contextual_concern(Site),
    !.


% ---------------------------------------------------------
% 9. Identify incomplete or invalid ML output
% ---------------------------------------------------------

% These conditions distinguish completely missing ML output
% from partially supplied or invalid ML output.

ml_output_issue(Site) :-
    data_issue(Site, invalid_ml_classification).

ml_output_issue(Site) :-
    data_issue(Site, ml_probability_missing).

ml_output_issue(Site) :-
    data_issue(Site, ml_classification_missing).

ml_output_issue(Site) :-
    data_issue(Site, invalid_ml_probability).


% ---------------------------------------------------------
% 10. Compatibility predicates
% ---------------------------------------------------------

% Preserve recommendation/3 for existing queries.
% The assessment rules remain the single source of decisions.

recommendation(Site, Action, Explanation) :-
    site_assessment(Site, Action, Explanation).

% Preserve reason/2 for queries that need the selected explanation.

reason(Site, Explanation) :-
    site_assessment(Site, _, Explanation).
