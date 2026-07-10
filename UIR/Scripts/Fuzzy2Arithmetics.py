import numpy as np
from numba import njit
from . import helperBenchmark as hb

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
    
    # Initialise some operations to avoid recomputing
    left = ep - clp
    right = ep + crp
    

    ##### FULL DISJUNCTION CASE (put disjunction to 0)
    fullDisjunctionMask = right < cr
    out[fullDisjunctionMask] = 0

    ##### FULL INCLUSION CASE (put inclusion to 1)
    fullInclusionMask = (left >= cr) & (ep >= er)
    out[fullInclusionMask] = 1

    # Remaining is the mask to avoid modifying twice the same area
    remaining = ~(fullDisjunctionMask | fullInclusionMask)

    # Early Exit if not any job was found
    if not np.any(remaining):
        return out

    ##### HELPER FUNCTIONS (error environment to handle division by zero error)
    with np.errstate(divide="ignore", invalid="ignore"):
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
    with np.errstate(divide="ignore", invalid="ignore"):
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
        with np.errstate(divide="ignore", invalid="ignore"):
            integral[outProviderMask] = ((right - ix2) * P2(ix2) / 2)[outProviderMask]
            integral[inProviderMask] = ((P1(ix1) * (ep - ix1) + right - ix1) / 2)[inProviderMask]

        # Compute the inclusion
        with np.errstate(divide="ignore", invalid="ignore"):
            out[fullStepMask] = (integral / Xp)[fullStepMask]

    # Remove full step from remaining
    remaining &= ~(cr == er)

    ##### PHANTOM CASE (Right, Left and Parallel)
    phantomMask = (remaining & (undef | (ix1 <= np.minimum(left, cr)) | (ix1 >= er)))

    # Compute the results only if phantoms cases exists
    if np.any(phantomMask):
        with np.errstate(divide="ignore", invalid="ignore"):
            integral = R(ix2) * (right - cr) / 2
            out[phantomMask] = (integral / Xp)[phantomMask]

    # Remove phantoms from remaining
    remaining &= ~phantomMask

    ##### NON PHANTOM REQUIREMENT FIRST
    reqFirstMask = remaining & (cr <= left)
        
    # Computation iff this case exists
    if np.any(reqFirstMask):
        with np.errstate(divide="ignore", invalid="ignore"):
            integral = (R(ix1) * (clp - ep + ix2) + R(ix2) * (right - ix1)) / 2
            out[reqFirstMask] = (integral / Xp)[reqFirstMask]

    # Remove non phantom requirement first from remaining
    remaining &= ~reqFirstMask

    ##### NON PHANTOM PROVIDER FIRST
    provFirstMask = remaining

    # Computation iff this case exists
    if np.any(provFirstMask):
        with np.errstate(divide="ignore", invalid="ignore"):
            integral = (R(ix1) * (ep - cr) + right - ix1) / 2
            out[provFirstMask] = (integral / Xp)[provFirstMask]

    # Return the matrix of inclusions for each corresponding pair of skills requirement and provide
    return out

def minimumInclusionDegree(learner:np.ndarray[float], jobs:np.ndarray[float], inverted:bool = False) -> np.ndarray[float]:
    if not inverted:
        return InclusionDegree(learner, jobs).min(axis=1)
    return InvertedInclusionDegree(learner, jobs).min(axis=1)

def InvertedInclusionDegree(learner: np.ndarray[float], providers: np.ndarray[float]) -> np.ndarray[float]:
    # Provider
    ep  = learner[None, :, 0] # Learner expertise (1,#S)
    clp = learner[None, :, 1] # Learner left confidence (1,#S)
    crp = learner[None, :, 2] # Learner right confidence (1,#S)

    # Requirement
    er = providers[:, :, 0] # Requirement expertise for each job and skills (#J,#S)
    cr = providers[:, :, 1] # Requirement confidence for each job and skills (#J,#S)

    # Areas of the learner expertise [OK]
    Xp = (clp + crp) / 2

    # Initialise the object to return [OK]
    out = np.empty_like(er, dtype=float)

    ##### FULL DISJUNCTION CASE (put disjunction to 0) [TRANSFORMED]
    fullDisjunctionMask = ep - clp > cr
    out[fullDisjunctionMask] = 0

    ##### FULL INCLUSION CASE (put inclusion to 1) [TRANSFORMED]
    fullInclusionMask = (ep + crp <= cr) & (ep <= er)
    out[fullInclusionMask] = 1

    # Remaining is the mask to avoid modifying twice the same area [OK]
    remaining = ~(fullDisjunctionMask | fullInclusionMask)

    # Early Exit if not any job was found [OK]
    if not np.any(remaining):
        return out

    ##### HELPER FUNCTIONS (error environment to handle division by zero error) [OK]
    with np.errstate(divide="ignore", invalid="ignore"):
        R = lambda x: (x - er) / (cr - er) # Requirement Ramp function [TRANSFORMED]
        P1 = lambda x: 1 + (x - ep) / clp # Provider triangle left function [OK]
        P2 = lambda x: 1 - (x - ep) / crp # Provider triangle right function [OK]

        # Compute intersection
        ix1 = (clp*er + cr*ep - er*ep) / (cr + clp - er) # [OK]
        ix2 = (crp*er - cr*ep + ep*er) / (crp - cr + er) # [OK]

    # Mask to handle ix2 undefined behaviour [TRANSFORMED]
    undef = (crp - cr + er) == 0

    ##### DEGENERATE TRIANGLE CASE (no width) [OK]
    degenerateTriMask = remaining & (Xp == 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[degenerateTriMask] = R(ep)[degenerateTriMask]

    # Remove degenerate case from remaining [OK]
    remaining &= ~(Xp == 0)

    ##### FULL STEP CASE [OK]
    fullStepMask = remaining & (cr == er)

    # Do not perform computation if no step case exists
    if np.any(fullStepMask):
        ## Compute provider OUT and provider IN subcase masks [TRANSFORMED]
        outProviderMask = fullStepMask & (ep >= er)
        inProviderMask = fullStepMask & (ep < er)

        # Initialise the results [OK]
        integral = np.zeros_like(out)

        # OUT case, then IN case [OK]
        with np.errstate(divide="ignore", invalid="ignore"):
            integral[outProviderMask] = ((cr - ep + clp) * P1(cr) / 2)[outProviderMask] # [TRANSFORMED]
            integral[inProviderMask] = ((clp + cr - ep + P2(cr)*(cr-ep)) / 2)[inProviderMask] # [TRANSFORMED]

        # Compute the inclusion [OK]
        with np.errstate(divide="ignore", invalid="ignore"):
            out[fullStepMask] = (integral / Xp)[fullStepMask]

    # Remove full step from remaining [OK]
    remaining &= ~(cr == er)

    ##### PHANTOM CASE (Right, Left and Parallel) [TRANSFORMED]
    phantomMask = (remaining & (undef | (ix2 >= np.maximum(ep + crp, cr)) | (ix2 <= er))) 

    # Compute the results only if phantoms cases exists [OK]
    if np.any(phantomMask):
        with np.errstate(divide="ignore", invalid="ignore"):
            integral = R(ix1) * (cr-ep+clp) / 2 # [TRANSFORMED]
            out[phantomMask] = (integral / Xp)[phantomMask] # [OK]

    # Remove phantoms from remaining {OK}
    remaining &= ~phantomMask

    ##### NON PHANTOM REQUIREMENT LAST
    reqLastMask = remaining & (cr >= ep + crp)
        
    # Computation iff this case exists [OK]
    if np.any(reqLastMask):
        with np.errstate(divide="ignore", invalid="ignore"):
            integral = (R(ix1) * (clp - ep + ix2) + R(ix2) * (ep - ix1 + crp)) / 2 # [TRANSFORMED]
            out[reqLastMask] = (integral / Xp)[reqLastMask]

    # Remove non phantom requirement first from remaining [OK]
    remaining &= ~reqLastMask

    ##### NON PHANTOM PROVIDER LAST
    provLastMask = remaining

    # Computation iff this case exists
    if np.any(provLastMask):
        with np.errstate(divide="ignore", invalid="ignore"):
            integral = (clp + ix2 - ep + R(ix2)*(cr-ep)) / 2
            out[provLastMask] = (integral / Xp)[provLastMask]

    # Return the matrix of inclusions for each corresponding pair of skills requirement and provide
    return out

def DeltaRampTriangle(jobs:np.ndarray[float], learner:np.ndarray[float]) -> np.ndarray[float]:
    # Provider
    ep  = learner[None, :, 0] # Learner expertise (1,#S)
    clp = learner[None, :, 1] # Learner left confidence (1,#S)
    crp = learner[None, :, 2] # Learner right confidence (1,#S)

    # Requirement
    er = jobs[:, :, 0] # Requirement expertise for each job and skills (#J,#S)
    cr = jobs[:, :, 1] # Requirement confidence for each job and skills (#J,#S)
    
    # Computation
    x1 = np.where(er-ep-crp<0, 0, er-ep-crp)
    x2 = np.where(er-ep<0, 0, er-ep)
    x3 = np.where(cr-ep+clp<0, 0, er-ep+clp)
    
    # Sort
    X = np.sort(np.stack((x1, x2, x3), axis=2), axis=2)
    
    # Reconstitute triangles
    return np.stack((X[:,:,1], X[:,:,1]-X[:,:,0], X[:,:,2]-X[:,:,1]), axis=2)

def DeltaTriangleTriangle(training:np.ndarray[float], subdelta:np.ndarray[float]) -> np.ndarray[float]:
    # Training Provider
    et  = training[None, :, 0] # Training expertise (1,#S)
    clt = training[None, :, 1] # Training left confidence (1,#S)
    crt = training[None, :, 2] # Training right confidence (1,#S)
    
    # SubDelta Provider
    ed  = subdelta[:, :, 0] # Sub Delta expertise (#J, #S)
    cld = subdelta[:, : ,1] # Sub Delta left confidence (#J, #S)
    crd = subdelta[:, :, 2] # Sub Delta right confidence (#J, #S)
    
    x1 = np.maximum(0, et+crt-ed-crd)
    x2 = np.maximum(0, et-ed)
    x3 = np.maximum(0, et-clt-ed+cld)
    
    # Sort
    X = np.sort(np.stack((x1, x2, x3), axis=2), axis=2)
    
    # Reconstitute triangles
    return np.stack((X[:,:,1], X[:,:,1]-X[:,:,0], X[:,:,2]-X[:,:,1]), axis=2)
    
def TriangleUnion(courseAcquire:np.ndarray[float], learner:np.ndarray[float]) -> np.ndarray[float]:
    ep1 = courseAcquire[:,0]
    l1 = ep1-courseAcquire[:,1]
    r1 = ep1+courseAcquire[:,2]
    
    ep2 = learner[:,0]
    l2 = ep2-learner[:,1]
    r2 = ep2+learner[:,2]
    
    z = np.sort(np.stack((np.maximum(ep1, ep2), np.maximum(l1,l2), np.maximum(r1,r2)), axis=1), axis=1)
    epz = z[:,1]
    clz = epz-z[:,0]
    crz = z[:,2]-epz
    
    return np.column_stack((epz,clz,crz))

def TrianglesToRamps(triangles:np.ndarray[float], inverted:bool = False) -> np.ndarray[float]:
    ep  = triangles[None, :, 0] # Triangles expertise (1,#S)
    if not inverted:
        clp = triangles[None, :, 1] # Triangles left confidence (1,#S)
        return np.stack((ep, ep-clp), axis=2) #(1,#S,2)
    crp = triangles[None, :, 2]
    return np.stack((ep, ep+crp), axis=2)

def TrianglesToRamps2(triangles:np.ndarray[float], inverted:bool = False) -> np.ndarray[float]:
    ep = triangles[:,:,0] # Triangle expertise (#J, #S)
    if not inverted:
        clp = triangles[:,:,1] # Triangle left confidence (#J, #S)
        return np.stack((ep, ep-clp), axis=2)
    crp = triangles[:,:,2]
    return np.stack((ep, ep+crp), axis=2)
      
def TrianglesSum(triangles:np.ndarray[float]) -> np.ndarray[float]:
    return triangles.sum(axis=0)

def TriangleDivision(t1:np.ndarray[float], t2:np.ndarray[float]) -> np.ndarray[float]:
    if t2[1] == 0 and t2[2] == 0:
        return np.array([t1[0]/t2[0], 0, 0])
    elif t2[1] == 0:
        return np.array([t1[0]/t2[0], 0, max(t1[1]/t2[2], t1[2]/t2[2])])
    elif t2[2] == 0:
        return np.array([t1[0]/t2[0], min(t1[1]/t2[1], t1[2]/t2[1]), 0])
    else:
        vals = {t1[1]/t2[1], t1[1]/t2[2], t1[2]/t2[1], t1[2]/t2[2]}
        return np.array([t1[0]/t2[0], min(vals), max(vals)])

def TriangleScalarAddition(t1:np.ndarray[float], t2:float) -> np.ndarray[float]:
    return np.array([t1[0]+t2, t1[1], t1[2]])

def TriangleScalarMultiplication(t1:np.ndarray[float], m:float) -> np.ndarray[float]:
    x = t1[0]
    ul = x-t1[1]
    ur = x+t1[2]
    mx = m*x
    vals = {m*ul, m*ur}
    
    return np.array([mx, mx-min(vals), max(vals)-mx])

def ClipTriangle(triangles:np.ndarray[float], ep_clip:list[float] = [0,1]) -> np.ndarray[float]:
    ep = triangles[:,0].clip(*ep_clip)
    clp = np.where(triangles[:,1]>ep, ep, triangles[:,1])
    crp = np.where(triangles[:,2]>1-ep, 1-ep, triangles[:,2])
    return np.column_stack((ep, clp, crp))

def TriangleCentroidDefuzzification(triangle:np.ndarray[float]) -> np.ndarray[float]:
    et = triangle[0]
    L = et - triangle[1]
    R = et + triangle[2]
    
    return (et + L + R)/3

def RampSum(ramps:np.ndarray[float]) -> np.ndarray[float]:
    er = ramps.sum(axis=0)[:,0]
    cr = er-(ramps[:,:,0]-ramps[:,:,1]).sum(axis=0)
    return np.column_stack((er, cr))

def ClipRamp(ramps:np.ndarray[float], clip:list[float] = [0,1]) -> np.ndarray[float]:
    er = ramps[:,0].clip(*clip)
    cr = ramps[:,1].clip(*clip)
    return np.column_stack((er,cr)) 

@njit(inline="always")
def _nb_R(x:float, er:float, cr:float) -> float:
    return (x - cr) / (er - cr)

@njit(inline="always")
def _nb_RInverted(x:float, er:float, cr:float) -> float:
    return (x - er) / (cr - er)

@njit(inline="always")
def _nb_P1(x:float, ep:float, clp:float) -> float:
    return 1.0 + (x - ep) / clp

@njit(inline="always")
def _nb_P2(x:float, ep:float, crp:float) -> float:
    return 1.0 - (x - ep) / crp

@njit(cache=True)
def _nb_InclusionDegree(learner:np.ndarray[float], jobs:np.ndarray[float]) -> np.ndarray[float]:
    # Retrieve shapes
    J = jobs.shape[0]
    S = jobs.shape[1]

    # Initialise out
    out = np.empty((J, S), dtype=np.float64)

    # For each job and subsequent skill
    for j in range(J):
        for s in range(S):
            # Retrieve the skill expertise state
            ep  = learner[s, 0]
            clp = learner[s, 1]
            crp = learner[s, 2]

            # Retrieve the job expertise state
            er = jobs[j, s, 0]
            cr = jobs[j, s, 1]

            # Compute area of main triangle
            Xp = (clp + crp) / 2

            # Variables to avoid recomputation
            left  = ep - clp
            right = ep + crp

            # Full disjunction
            if right < cr:
                out[j, s] = 0.0
                continue

            # Full inclusion
            if left >= cr and ep >= er:
                out[j, s] = 1.0
                continue

            # Degenerate triangle
            if Xp == 0.0:
                out[j, s] = _nb_R(ep, er, cr)
                continue

            # Step case
            if cr == er:
                # OUT CASE
                if ep <= er:
                    ix2 = (crp*er - cr*ep + ep*er) / (crp - cr + er)
                    integral = ((right - ix2) * _nb_P2(ix2, ep, crp)) / 2
                # IN CASE
                else:
                    ix1 = (clp*er + cr*ep - er*ep) / (cr + clp - er)
                    integral = (_nb_P1(ix1, ep, clp) * (ep - ix1) + right - ix1) / 2

                out[j, s] = integral / Xp
                continue
            
            # Phantom Case
            undef = (cr + clp - er) == 0
            if not undef:
                ix1 = (clp*er + cr*ep - er*ep) / (cr + clp - er)
            ix2 = (crp*er - cr*ep + ep*er) / (crp - cr + er)
            if undef or ix1 <= min(left, cr) or ix1 >= er:
                integral = _nb_R(ix2, er, cr) * (right - cr) / 2
                out[j, s] = integral / Xp
                continue
            
            # Non Phantom Requirement First
            if cr <= left:
                integral = (_nb_R(ix1, er, cr) * (clp - ep + ix2) + _nb_R(ix2, er, cr) * (right - ix1)) / 2
                out[j, s] = integral / Xp
                continue
            
            # Non Phantom Provider First
            if cr >= left:
                integral = ((_nb_R(ix1, er, cr) * ep - cr) + right - ix1) / 2
                out[j, s] = integral / Xp
                continue
    return out

@njit(cache=True)
def _nb_InvertedInclusionDegree(learner:np.ndarray[float], jobs:np.ndarray[float]) -> np.ndarray[float]:
    # Retrieve shapes
    J = jobs.shape[0]
    S = jobs.shape[1]

    # Initialise out
    out = np.empty((J, S), dtype=np.float64)

    # For each job and subsequent skill
    for j in range(J):
        for s in range(S):
            # Retrieve the skill expertise state
            ep  = learner[s, 0]
            clp = learner[s, 1]
            crp = learner[s, 2]

            # Retrieve the job expertise state
            er = jobs[j, s, 0]
            cr = jobs[j, s, 1]

            # Compute area of main triangle
            Xp = (clp + crp) / 2

            # Variables to avoid recomputation
            left  = ep - clp
            right = ep + crp

            # Full disjunction [TRANSFORMED]
            if left > cr:
                out[j, s] = 0.0
                continue

            # Full inclusion [TRANSFORMED]
            if (right <= cr) and (ep <= er):
                out[j, s] = 1.0
                continue

            # Degenerate triangle [OK]
            if Xp == 0.0:
                out[j, s] = _nb_RInverted(ep, er, cr)
                continue

            # Step case
            if cr == er:
                # OUT CASE
                if ep >= er:
                    integral = ((cr - ep + clp) * _nb_P1(cr, ep, clp) / 2)
                # IN CASE
                else:
                    integral = ((clp + cr - ep + _nb_P2(cr, ep, crp)*(cr-ep)) / 2)
                out[j, s] = integral / Xp
                continue
            
            # Phantom Case
            undef = (crp - cr + er) == 0
            if not undef:
                ix2 = (crp*er - cr*ep + ep*er) / (crp - cr + er)
            if undef or ix2 <= max(right, cr) or ix2 <= er:
                integral = _nb_RInverted(ix2, er, cr) * (cr-ep+clp) / 2
                out[j, s] = integral / Xp
                continue
            
            # Non Phantom Requirement Last
            if cr >= right:
                ix1 = (clp*er + cr*ep - er*ep) / (cr + clp - er)
                integral = (_nb_RInverted(ix1, er, cr) * (clp - ep + ix2) + _nb_RInverted(ix2, er, cr) * (right - ix1)) / 2
                out[j, s] = integral / Xp
                continue
            
            if cr <= right:
                integral = (clp + ix2 - ep + _nb_RInverted(ix2, er, cr)*(cr-ep)) / 2
                out[j, s] = integral / Xp
                continue
    return out



#if __name__ == "__main__":
#    # (1st skill) Fully Disjoint [OK] 
#    # (2nd skill) Fully Included [OK]
#    # (3rd skill) Phantom Intersection (left) [OK]
#    # (4th skill) Phantom Intersection (right) [OK]
#    # (5th skill) Phantom Intersection (none) [OK]
#    # (6th skill) Partial Inclusion (Requirement First) [OK]
#    # (7th skill) Partial Inclusion (Provider First) [OK]
#    # (8th skill) Partial Inclusion (Full Step, Provider Max out) [OK]
#    # (9th skill) Partial Inclusion (Full Step, Provider Max in) [OK]
#    # (10th skill) Partial Inclusion (Crisp Provider Max Out) [OK]
#    # (11th job) INCLUSION FOR EVERYTHING
#    learner_expertise = np.array([0.15, 0.70, 0.35, 0.50, 0.50, 0.50, 0.70, 0.55, 0.65, 0.50]) 
#    learner_left =      np.array([0.10, 0.10, 0.10, 0.35, 0.30, 0.10, 0.50, 0.50, 0.50, 0.00])
#    learner_right =     np.array([0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.00])
#   
#    job_expertise =     np.array([
#                                 [0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60],  # (1st skill) Fully Disjoint
#                                 [0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60],  # (2nd skill) Fully Included
#                                 [0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60],  # (3rd skill) Phantom Intersection (left)
#                                 [0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60],  # (4th skill) Phantom Intersection (right)
#                                 [0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60],  # (5th skill) Phantom Intersection (none)
#                                 [0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60],  # (6th skill) Partial Inclusion (Requirement First)
#                                 [0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60],  # (7th skill) Partial Inclusion (Provider First)
#                                 [0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60],  # (8th skill) Partial Inclusion (Full Step, Provider Max Out)
#                                 [0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60],  # (9th skill) Partial Inclusion (Full Step, Provider Max In)
#                                 [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],  # (10th skill) Partial Inclusion (Crisp Provider Max Out)
#                                 [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]   # (11th job) INCLUSION FOR EVERYTHING        
#    ])
#    
#    job_confidence =    np.array([
#                                 [0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30],  # (1st skill) Fully Disjoint
#                                 [0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30],  # (2nd skill) Fully Included
#                                 [0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30],  # (3rd skill) Phantom Intersection (left)
#                                 [0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30],  # (4th skill) Phantom Intersection (right)
#                                 [0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30],  # (5th skill) Phantom Intersection (none)
#                                 [0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30],  # (6th skill) Partial Inclusion (Requirement First)
#                                 [0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30],  # (7th skill) Partial Inclusion (Provider First)
#                                 [0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60],  # (8th skill) Partial Inclusion (Full Step, Provider Max Out)
#                                 [0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60],  # (9th skill) Partial Inclusion (Full Step, Provider Max In)
#                                 [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],  # (10th skill) Partial Inclusion (Crisp Provider Max Out)
#                                 [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]   # (11th job) INCLUSION FOR EVERYTHING
#    ])
#    
#    learner = np.column_stack((learner_expertise, learner_left, learner_right)) # (S,3)
#    jobs = np.stack((job_expertise, job_confidence), axis=2) # (J,S,2)
#    
#    minimumInclusionDegree(learner, jobs)


