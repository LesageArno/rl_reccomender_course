import numpy as np

def minimumInclusionDegree(learner:np.ndarray[float], jobs:np.ndarray[float]) -> float:
    # Initialise the result
    res = 0
    
    # Provider
    ep = learner[:,0]
    clp = learner[:,1]
    crp = learner[:,2]
    
    # Requirements
    er = jobs[:,:,0]
    clr = jobs[:,:,1]
    
    # Keep only the one that are not disjoint
    mask = np.all(ep+crp>=clr, axis=1)
    er = er[mask]
    clr = clr[mask]
    
    # If any expertise does not overlap for all the job, then the result will be 0 (early exit)
    if er.shape[0]==0:
        return 0
    
    
    
    # Intersections for all skills
    ix1 = (er*clp+clr*ep-er*ep)/(clr+clp-er)
    ix2 = (crp*er - clr*ep + ep*er)/(crp - clr + er)
    
    #Area of the triangle
    xp = (crp + clp)/2
    
    ...
    
if __name__ == "__main__":
    # (1st skill) Fully Disjoint [OK] 
    # (2nd skill) Fully Included [OK]
    # (3rd skill) Phantom Intersection (left) [OK]
    # (4th skill) Phantom Intersection (right) [OK]
    # (5th skill) Phantom Intersection (none) [OK]
    # (6th skill) Partial Inclusion (Requirement First) [OK]
    # (7th skill) Partial Inclusion (Provider First) [OK]
    # (8th skill) Partial Inclusion (Full Step, Provider Max out) [OK]
    # (9th skill) Partial Inclusion (Full Step, Provider Max in) [OK]
    # (10th skill) Partial Inclusion (Crisp Provider Max Out) [OK]
    learner_expertise = np.array([0.15, 0.70, 0.35, 0.50, 0.50, 0.50, 0.70, 0.55, 0.65, 0.50]) 
    learner_left =      np.array([0.10, 0.10, 0.10, 0.35, 0.30, 0.10, 0.50, 0.50, 0.50, 0.00])
    learner_right =     np.array([0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.00])
   
    job_expertise =     np.array([
                                 [0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60],  # (1st skill) Fully Disjoint
                                 [0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60],  # (2nd skill) Fully Included
                                 [0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60],  # (3rd skill) Phantom Intersection (left)
                                 [0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60],  # (4th skill) Phantom Intersection (right)
                                 [0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60],  # (5th skill) Phantom Intersection (none)
                                 [0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60],  # (6th skill) Partial Inclusion (Requirement First)
                                 [0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60],  # (7th skill) Partial Inclusion (Provider First)
                                 [0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60],  # (8th skill) Partial Inclusion (Full Step, Provider Max Out)
                                 [0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60],  # (9th skill) Partial Inclusion (Full Step, Provider Max In)
                                 [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],  # (10th skill) Partial Inclusion (Crisp Provider Max Out)
                                 
    ])
    
    job_confidence =    np.array([
                                 [0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30],  # (1st skill) Fully Disjoint
                                 [0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30],  # (2nd skill) Fully Included
                                 [0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30],  # (3rd skill) Phantom Intersection (left)
                                 [0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30],  # (4th skill) Phantom Intersection (right)
                                 [0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30],  # (5th skill) Phantom Intersection (none)
                                 [0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30],  # (6th skill) Partial Inclusion (Requirement First)
                                 [0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30],  # (7th skill) Partial Inclusion (Provider First)
                                 [0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60],  # (8th skill) Partial Inclusion (Full Step, Provider Max Out)
                                 [0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60],  # (9th skill) Partial Inclusion (Full Step, Provider Max In)
                                 [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],  # (10th skill) Partial Inclusion (Crisp Provider Max Out)

    ])
    
    learner = np.column_stack((learner_expertise, learner_left, learner_right)) # (S,3)
    jobs = np.stack((job_expertise, job_confidence), axis=2) # (J,S,2)

    def InclusionDegree(
        learner: np.ndarray[float],
        jobs: np.ndarray[float]
    ) -> np.ndarray[float]:
        
        ep = learner[None, :]
        
        # Provider
        ep  = learner[None, :, 0]      # (1,S)
        cpl = learner[None, :, 1]
        cpr = learner[None, :, 2]

        # Requirement
        er = jobs[:, :, 0]          # (J,S)
        cr = jobs[:, :, 1]

        Xp = (cpl + cpr) / 2

        out = np.empty_like(er, dtype=float)

        # ------------------------------------------------------------------
        # Full disjunction
        # ------------------------------------------------------------------
        mask = ep + cpr < cr
        out[mask] = 0.0

        # ------------------------------------------------------------------
        # Full inclusion
        # ------------------------------------------------------------------
        mask2 = (ep - cpl >= cr) & (ep >= er)
        out[mask2] = 1.0

        remaining = ~(mask | mask2)

        if not np.any(remaining):
            return out

        # ------------------------------------------------------------------
        # Common functions
        # ------------------------------------------------------------------
        with np.errstate(divide='ignore', invalid='ignore'):
            R_ix = lambda x: (x - cr) / (er - cr)
            P1 = lambda x: 1 + (x - ep) / cpl
            P2 = lambda x: 1 - (x - ep) / cpr

            ix1 = (cpl*er + cr*ep - er*ep) / (cr + cpl - er)
            ix2 = (cpr*er - cr*ep + ep*er) / (cpr - cr + er)

        undef = (cr + cpl - er) == 0

        # ------------------------------------------------------------------
        # No triangular width
        # ------------------------------------------------------------------
        mask_xp0 = remaining & (Xp == 0)
        out[mask_xp0] = ((ep - cr) / (er - cr))[mask_xp0]

        remaining &= ~(Xp == 0)

        # ------------------------------------------------------------------
        # Full step
        # ------------------------------------------------------------------
        mask_step = remaining & (cr == er)

        if np.any(mask_step):

            left = mask_step & (ep <= er)
            right = mask_step & (ep > er)

            integral = np.zeros_like(out)

            integral[left] = (
                (ep + cpr - ix2) * P2(ix2) / 2
            )[left]

            integral[right] = (
                (P1(ix1) * (ep - ix1) + ep - ix1 + cpr) / 2
            )[right]

            out[mask_step] = (integral / Xp)[mask_step]

        remaining &= ~(cr == er)

        # ------------------------------------------------------------------
        # Phantom intersection
        # ------------------------------------------------------------------
        phantom = (
            remaining &
            (
                undef |
                (ix1 <= np.minimum(ep - cpl, cr)) |
                (ix1 >= er)
            )
        )

        if np.any(phantom):

            integral = R_ix(ix2) * (ep + cpr - cr) / 2
            out[phantom] = (integral / Xp)[phantom]

        remaining &= ~phantom

        # ------------------------------------------------------------------
        # Requirement start first
        # ------------------------------------------------------------------
        req_first = remaining & (cr <= ep - cpl)

        if np.any(req_first):

            integral = (
                R_ix(ix1) * (cpl - ep + ix2)
                + R_ix(ix2) * (cpr + ep - ix1)
            ) / 2

            out[req_first] = (integral / Xp)[req_first]

        remaining &= ~req_first

        # ------------------------------------------------------------------
        # Provider start first
        # ------------------------------------------------------------------
        prov_first = remaining

        if np.any(prov_first):

            integral = (
                R_ix(ix1) * (ep - cr)
                + ep - ix1 + cpr
            ) / 2

            out[prov_first] = (integral / Xp)[prov_first]

        return out,Xp
        
        
    out, Xp = InclusionDegree(learner, jobs)
    out.round(5)
