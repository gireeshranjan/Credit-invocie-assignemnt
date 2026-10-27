# Portfolio analytics

14,685 loans; 10,000 customers; 2,364 labelled defaults (16.10%).

## Key insights

1. Mortgage concentration: mortgages are 9.8% of loans but 52.3% of original lending, and default at 24.7% (1.5x the 16.1% portfolio rate). Action: review mortgage affordability and concentration limits first.
2. Income: default rates fall as income rises, from 38.7% under USD 30k to 7.8% at USD 120k+. Below USD 60k (3,215 loans) the rate is 25.3%. Action: tighten affordability and loan-size checks below USD 60k.
3. Employment x income: unemployed customers default at 35.0%; unemployed and below USD 60k reach 37.8% (547 loans, 207 defaults), more than double the portfolio rate. Action: route this segment to manual underwriting review.
4. Prior write-offs: risk rises with each debt another lender has written off: 14.6% with none, 17.7% with one, 21.9% with two and 31.4% with three or more. 29% of loans belong to customers with at least one. Action: make a prior write-off a mandatory review trigger at application.
5. Age: default rates decline steadily with age, from 21.6% under 30 (22% of loans) to 6.8% at 60+. Action: apply affordability buffers and lower starting limits for younger borrowers.

## Trends over time

- Lending volume was steady: 569-663 loans per origination quarter from 2018Q1 to 2023Q4.
- Cohort default rates ranged 13.6%-19.9% with no significant trend (-0.08 percentage points per year, p = 0.69). Outside the +/-2 standard-error control band: 2018Q1 (19.9%), 2022Q2 (19.2%); about 1 of 24 quarters would fall outside by chance alone, so these merit review rather than alarm.
- Loans that later defaulted were 30+ days past due more often from the first month (6.4% vs 4.3%), and the gap holds across all 12 months: delinquency is an early signal rather than something that builds with loan age.
- Calendar-month delinquency stayed between 3.8% and 5.9% of active loans (2018-04 to 2024-10) with no sustained deterioration. Action: track cohort default and 30+ DPD rates against these control limits each quarter.

## Further observations

- Missed payments as early warning: 12.9% default with no missed instalments, 18.1% with one, 25.3% with two and 34.1% with three or more. This is the same 12-month window as the outcome, so use it as a monitoring trigger, not an application rule.
- Credit activity: 67% of customers hold active credit elsewhere (15.7% defaults). Customers with no bureau record default at 17.6%; of customers whose outside credit is all closed, 42% have a recorded write-off.
- What does not drive risk: region (15.0%-16.9%) and origination year (14.9%-17.7%) barely move default rates. The one hotspot is South mortgages at 27.3% (381 loans).
- Gender: female loans 16.3%, male loans 16.0%; differences vary by income group, and gender is excluded from the model.

Observed associations in a synthetic, retrospective dataset: not causal, and not lending rules on their own. Unknowns are retained, groups under 100 loans are flagged, and no expected-loss or profit claim is made.

Full tables: exports/excel/portfolio_analytics.xlsx; CSVs and charts: reports/analytics.