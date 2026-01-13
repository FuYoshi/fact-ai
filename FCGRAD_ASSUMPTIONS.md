# FCGrad Implementation

## Matches Paper
- Separate networks per agent
- Lower return = disadvantaged = prioritized
- Collective return = mean of all agents' TD(λ) targets
- Conflict detection: `g_ind · g_col < 0`
- Projection onto normal plane formula
- Collective advantage = mean of all agents' advantages (for collective gradient)

## Assumption (not in paper)
- **No-conflict case:** Use individual gradient unchanged
