# Observable-SAC freeze — NOT GRANTED

**used_final_eval_data: false**

This file exists to record that the freeze gate **did not pass**.

The 27-D observable observation and leakage tests are in place, but
`success_obs` **frequently awards +8 and ends the episode while GT
`recovered()` is false** because residual `v_rel` is still 0.02–0.09 m/s
(pad CoP ê-dot is a weak slip proxy). That is a **BLOCKER_BEFORE_SAC**.

This remains a **simulated spatial-tactile** formulation. It is **not**
hardware validation.

Do **not** start the first SAC smoke until the terminal condition is
resolved without GT in the policy or reward.

See `OBSERVABLE_POLICY_INTEGRATION.md`.
