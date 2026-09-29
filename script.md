### Title

Good evening. My project is **Residual Reinforcement Learning for Adaptive Grasp Recovery under Dynamics and Grasp Uncertainty**.

### Slide 1 — Problem Definition and Relevant Literature

The problem I focus on is grasp deterioration during lifting.

Even if the robot successfully grasps an object, uncertainty in object mass, friction, or grasp position may cause the object to slip, rotate, or eventually drop during lifting.

Existing feedback grasp control can react to unstable grasps using contact information. Residual reinforcement learning provides another approach, where a learned policy modifies an existing controller instead of replacing it.

So my research question is: **can a learned recovery policy adapt its corrective actions to different grasp-deterioration states?**

### Slide 2 — Proposed Solution

My system uses a nominal Cartesian controller for the normal manipulation process: approach, descend, close, and lift.

During lifting, I calculate a ground-truth deterioration score, \(D_t\), using the relative motion between the object and gripper together with contact information.

When \(D_t\) exceeds a threshold, the system switches from normal lifting into recovery mode.

In recovery, SAC outputs two bounded residual actions: a correction to vertical velocity and a correction to gripping force.

This allows the robot to react differently depending on the current state. For example, it can increase the gripping force, slow down the lift, or even temporarily move downward.

Once the grasp becomes stable again, the system returns to the normal lifting controller.

For the experiments, I randomize object mass, friction, and grasp offset. The policy is not directly given the mass or friction values.

### Slide 3 — Evaluation and Timeline

For evaluation, I compare three methods: the nominal controller without recovery, a hand-designed reactive recovery controller, and the SAC recovery controller.

The primary metric is **recovery rate**, which measures how many deteriorating grasps can be successfully stabilized.

I will also measure drop rate, relative slip, recovery time, and gripping effort.

By the project milestone on **October 20**, I plan to complete the main SAC training and initial comparison.

After that, I will focus on full evaluation, ablation experiments, and analysis for the final report and presentation around **December 1st**.

Thank you.