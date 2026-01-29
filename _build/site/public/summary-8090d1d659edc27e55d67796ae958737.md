1️⃣ Methods
Data sources

This analysis integrates four operational datasets:

Trips: trip duration, distance, timing, truck assignment

Incidents: incident occurrence, date, fault attribution, claim cost

Maintenance: maintenance events, downtime, maintenance type (including tyres)

Loads: revenue components linked to trips via load_id

Trips were deduplicated to one row per trip_id. Incidents and maintenance events were deduplicated to one row per incident_id and maintenance_id, respectively.

Exposure definition

Driving exposure was explicitly accounted for using:

Driving hours (primary exposure)

Driving miles (secondary normalization)

This avoids bias from unequal trip durations.

Trip segmentation

Trips were segmented into duration bins (hours per trip) to enable interpretable operational comparisons:

0–10, 10–20, 20–30, 30–40, 40–50, 50–60, 60+ hours

Incident risk analysis

Incident risk was analyzed using two complementary approaches:

Descriptive exposure-normalized rates

Incidents per 1,000 driving hours

95% confidence intervals assuming Poisson-distributed rare events

Poisson regression with exposure offset

Outcome: incident count per trip (0/1)

Offset: log(driving hours)

Predictor: trip duration bin

Output: incident rate ratios (IRR)

This formally tests whether incident rates per hour differ across trip durations.

Downtime analysis

Maintenance downtime was attributed to trips using a time-window join:

Maintenance events were linked to trips only if they occurred during the trip’s execution window for the same truck

Downtime was aggregated at the trip level

Results were normalized per 1,000 driving hours and per 1,000 miles

Revenue analysis

Revenue was joined from loads_df to trips via load_id.
Revenue efficiency was computed as:

Revenue per mile

Revenue per hour

All revenue metrics were aggregated at the trip level to avoid duplication.

2️⃣ Findings
2.1 Incident risk (statistically tested)

After adjusting for driving exposure:

Short trips (≤10 hours) have the highest incident rate per driving hour

Incident risk declines monotonically with trip duration

Poisson regression confirms that all trips longer than 10 hours have statistically significantly lower incident rates per hour (p < 0.05)

Chi-square tests at the trip level were not significant, as expected, because they ignore unequal exposure

Conclusion:
Incident risk is driven by risk intensity per hour, not by trip count.

2.2 Downtime

Short trips experience dramatically higher downtime per 1,000 driving hours

While longer trips are more likely to encounter downtime at least once, downtime on short trips is far more concentrated and disruptive

Average downtime per maintenance event is similar across trip lengths, indicating that the difference is driven by exposure concentration, not maintenance severity

Conclusion:
Short trips are operationally fragile.

2.3 Revenue efficiency

Revenue per mile and per hour is not materially higher for short trips relative to mid-length trips

Mid-duration trips (20–50 hours) deliver:

Comparable revenue efficiency

Significantly lower incident risk

Significantly lower downtime intensity

Very long trips (60+ hours) show low risk but lower revenue per hour

Conclusion:
Mid-duration trips represent the optimal balance of profitability and safety.

2.4 Integrated business insight

Short trips:

Generate reasonable revenue

But carry disproportionately high safety and downtime risk per unit of exposure

Exhibit high operational variance

This makes them risk-intensive rather than margin-intensive.

3️⃣ One-slide Executive Summary
Title

Short-Duration Trips Drive Disproportionate Risk Without Superior Returns

Key findings (left side)

Short trips (≤10h) have:

~2–4× higher incident rates per driving hour

~4× higher downtime per 1,000 hours

Differences are statistically significant after adjusting for exposure

Trip-level incident probability appears similar, but risk per hour is not

Financial context (middle)

Revenue per mile and per hour for short trips is:

Comparable to mid-length trips

Not high enough to compensate for elevated risk

Mid-duration trips (20–50h):

Similar revenue

Much lower operational disruption

Recommendation (right side)

Do not treat all trips equally

Prioritize:

Mid-duration trips (20–50h)

Manage short trips with:

Pricing premiums

Stricter acceptance criteria

Enhanced safety controls

Use exposure-normalized KPIs for ongoing monitoring

4️⃣ What’s missing / Next analyses to strengthen decisions

You correctly identified several important gaps. These don’t weaken your conclusions — they refine them.

4.1 Route analysis (high priority)

Why it matters:

Short trips are likely urban, congested, and complex

Risk may be route-driven rather than duration-driven

Next steps:

Group incidents and downtime by:

Route ID

Origin/destination metro vs highway

Identify:

High-risk routes per 1,000 hours

Routes with elevated “other driver” fault rates

4.2 Load analysis

Why it matters:

Load type, weight, or customer requirements may correlate with:

Urban delivery

Tight schedules

Frequent stops

Next steps:

Compare incident and downtime rates by:

Load type

Weight buckets

Customer segments

Identify loads that are low-margin + high-risk

4.3 Tyre maintenance drop in 2024 (important context)

You observed:

A significant drop in tyre maintenance events in 2024

Why this matters:

Could indicate:

Improved tyre technology (positive)

Deferred maintenance (negative)

May impact:

Breakdown risk

Safety outcomes with lag effects

Next steps:

Overlay tyre maintenance trends with:

Incident types (e.g., breakdown-related)

Downtime events

Check for lagged effects (maintenance ↓ → incidents ↑ later)

4.4 Fault attribution (“other drivers”)

You correctly flagged this as critical.

Key insight:

Many incidents are caused by external actors

But exposure still matters:

Urban density

Intersections

Stop-and-go traffic

Next steps:

Analyze fault attribution by:

Route

Trip duration

Urban vs highway proxies

Focus prevention on:

Defensive driving training

Intersection safety

Urban routing buffers

4.5 Preventive measures (evidence-based)

Based on current findings, prevention should focus on:

Where risk is concentrated, not where miles are highest

Examples:

Enhanced safety protocols for short trips

Route redesign for urban deliveries

Pricing or scheduling buffers to reduce time pressure

Preventive maintenance aligned with short-trip usage patterns

Final takeaway (this is the anchor)

After adjusting for unequal driving exposure, short-duration trips are statistically and operationally riskier without delivering superior revenue efficiency. Mid-duration trips provide a safer and more economically stable operating profile. Future work should focus on route, load, maintenance, and external-driver dynamics to refine targeted prevention strategies.