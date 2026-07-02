import numpy as np

def InclusionDegree(learner: np.ndarray[float], jobs: np.ndarray[float]) -> np.ndarray[float]:
        
    # Provider
    ep  = learner[None, :, 0] # Learner expertise (1,#S)
    clp = learner[None, :, 1] # Learner left confidence (1,#S)
    crp = learner[None, :, 2] # Learner right confidence (1,#S)

    # Requirement
    er = jobs[:, :, 0] # Requirement expertise for each job and skills (#J,#S)
    cr = jobs[:, :, 1] # Requirement confidence for each job and skills (#J,#S)

    # Areas of the learner expertise
    Xp = (clp + crp) / 2

    # Initialise the object to return
    out = np.empty_like(er, dtype=float)

    ##### FULL DISJUNCTION CASE (put disjunction to 0)
    fullDisjunctionMask = ep + crp < cr
    out[fullDisjunctionMask] = 0

    ##### FULL INCLUSION CASE (put inclusion to 1)
    fullInclusionMask = (ep - clp >= cr) & (ep >= er)
    out[fullInclusionMask] = 1

    # Remaining is the mask to avoid modifying twice the same area
    remaining = ~(fullDisjunctionMask | fullInclusionMask)

    # Early Exit if not any job was found
    if not np.any(remaining):
        return out

    ##### HELPER FUNCTIONS (error environment to handle division by zero error)
    with np.errstate(divide='ignore', invalid='ignore'):
        R = lambda x: (x - cr) / (er - cr) # Requirement Ramp function
        P1 = lambda x: 1 + (x - ep) / clp # Provider triangle left function
        P2 = lambda x: 1 - (x - ep) / crp # Provider triangle right function

        # Compute intersection
        ix1 = (clp*er + cr*ep - er*ep) / (cr + clp - er)
        ix2 = (crp*er - cr*ep + ep*er) / (crp - cr + er)

    # Mask to handle ix1 undefined behaviour
    undef = (cr + clp - er) == 0

    ##### DEGENERATE TRIANGLE CASE (no width)
    degenerateTriMask = remaining & (Xp == 0)
    out[degenerateTriMask] = R(ep)[degenerateTriMask]

    # Remove degenerate case from remaining
    remaining &= ~(Xp == 0)

    ##### FULL STEP CASE
    fullStepMask = remaining & (cr == er)

    # Do not perform computation if no step case exists
    if np.any(fullStepMask):
        ## Compute provider OUT and provider IN subcase masks
        outProviderMask = fullStepMask & (ep <= er)
        inProviderMask = fullStepMask & (ep > er)

        # Initialise the results
        integral = np.zeros_like(out)

        # OUT case, then IN case
        integral[outProviderMask] = ((ep + crp - ix2) * P2(ix2) / 2)[outProviderMask]
        integral[inProviderMask] = ((P1(ix1) * (ep - ix1) + ep - ix1 + crp) / 2)[inProviderMask]

        # Compute the inclusion
        out[fullStepMask] = (integral / Xp)[fullStepMask]

    # Remove full step from remaining
    remaining &= ~(cr == er)

    ##### PHANTOM CASE (Right, Left and Parallel)
    phantomMask = (remaining & (undef | (ix1 <= np.minimum(ep - clp, cr)) | (ix1 >= er)))

    # Compute the results only if phantoms cases exists
    if np.any(phantomMask):
        integral = R(ix2) * (ep + crp - cr) / 2
        out[phantomMask] = (integral / Xp)[phantomMask]

    # Remove phantoms from remaining
    remaining &= ~phantomMask

    ##### NON PHANTOM REQUIREMENT FIRST
    reqFirstMask = remaining & (cr <= ep - clp)
        
    # Computation iff this case exists
    if np.any(reqFirstMask):
        integral = (R(ix1) * (clp - ep + ix2) + R(ix2) * (crp + ep - ix1)) / 2
        out[reqFirstMask] = (integral / Xp)[reqFirstMask]

    # Remove non phantom requirement first from remaining
    remaining &= ~reqFirstMask

    ##### NON PHANTOM PROVIDER FIRST
    provFirstMask = remaining

    # Computation iff this case exists
    if np.any(provFirstMask):
        integral = (R(ix1) * (ep - cr) + ep - ix1 + crp) / 2
        out[provFirstMask] = (integral / Xp)[provFirstMask]

    # Return the matrix of inclusions for each corresponding pair of skills requirement and provide
    return out

def minimumInclusionDegree(learner:np.ndarray[float], jobs:np.ndarray[float]) -> float:
    return InclusionDegree(learner, jobs).min(axis=1).sum()
    
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
    # (11th job) INCLUSION FOR EVERYTHING
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
                                 [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]   # (11th job) INCLUSION FOR EVERYTHING        
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
                                 [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]   # (11th job) INCLUSION FOR EVERYTHING
    ])
    
    learner = np.column_stack((learner_expertise, learner_left, learner_right)) # (S,3)
    jobs = np.stack((job_expertise, job_confidence), axis=2) # (J,S,2)
    
    minimumInclusionDegree(learner, jobs)


