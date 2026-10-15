"""Launch-neighborhood and contact-order labels for ball-impact episodes."""

from __future__ import annotations

def near(spec, other):
    daz = abs((float(spec["azimuth_deg"]) - float(other["azimuth_deg"]) + 180.0) % 360.0 - 180.0)
    return (
        daz < 2.0
        and abs(float(spec["speed_mps"]) - float(other["speed_mps"])) < 0.20
        and abs(float(spec["z_off_m"]) - float(other["z_off_m"])) < 0.002
        and abs(float(spec["phi_deg"]) - float(other["phi_deg"])) < 5.0
        and abs(float(spec["delay_s"]) - float(other.get("delay_s", 0.0))) < 0.05
    )

def cylinder_first(evidence, blob):
    other = evidence.get("first_other")
    contact_t = evidence.get("contact_t")
    if evidence.get("category") != "valid_impact_recovery_ic":
        return evidence, blob
    if other is not None and contact_t is not None and float(other["t"]) < float(contact_t) - 1e-6:
        evidence = dict(evidence)
        evidence["category"] = "robot_or_environment_before_cylinder"
        evidence["reason"] = (
            f"{other.get('kind')} contact at t={other.get('t')} s preceded the cylinder contact at t={contact_t} s"
        )
        return evidence, None
    if other is not None and contact_t is not None and abs(float(other["t"]) - float(contact_t)) <= 1e-6:
        evidence = dict(evidence)
        evidence["simultaneous_other"] = True
    return evidence, blob

def mechanism(row):
    multi = int(row.get("none_intervals") or 0) >= 2
    if row.get("family") == "sliding" and not multi:
        return "contact_preserving_slide"
    if row.get("family") == "sliding":
        return "sliding"
    if multi:
        return "repeated_recatch"
    if row.get("strategy") == "chase_lost":
        return "airborne_chase"
