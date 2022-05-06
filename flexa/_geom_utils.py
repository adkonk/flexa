import numpy as np

def angle(a, b):
		"""Returns angle between vectors a and b"""
		return(np.arccos(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)))

def align(a, ref):
    """Orients a in direction of ref"""
    if np.dot(a, ref) < 0:
        a *= -1
    return(a)

def tri_normal(a, b=None, c=None, ref=np.array([0, 0, 1])):
    """Returns normal vector to plane defined by 3 rows in a or vecs abc"""
    # TODO: check inputs better
    if (len(a.shape) == 1 and a.size == 2) \
        or (len(a.shape) == 2 and a.shape[1] == 2): 
        return ref 

    if b is None and c is None:
        assert len(a.shape) == 2 and a.shape[0] == 3
        b = a[1, :]
        c = a[2, :]
        a = a[0, :]

    if isinstance(ref, str): # type check first to avoid warnings
        if ref == 'ori': # vector from mean to origin
            n = (a + b + c) / -3
            return(n / np.linalg.norm(n))
        else: 
            raise ValueError("only 'ori' is supported as string ref")

    n = np.cross(b - a, c - a)
    n = n / np.linalg.norm(n)
    return(align(n, ref))

def face_normal(r, ref=np.array([0, 0, 1])):
    """Returns normal to points (rows) in r by least squares plane approx"""
    if r.shape[0] == 3:
        return(tri_normal(r, ref=ref))

    if isinstance(ref, str): # type check first to avoid warnings
        if ref == 'ori': # vector from mean to origin
            n = np.mean(r, axis=0)
            return(n / -np.linalg.norm(n))
        else: 
            raise ValueError("only 'ori' is supported as string ref")
    
    # solve least squares approximation z_i = (x_i, y_i) * (v_1, v_2) + c
    # so then the normal vector for plane is (v_1, v_2, -1)
    x = np.concatenate((r[:, :2], np.ones((r.shape[0], 1))), 
                        axis = 1)
    v = np.linalg.inv(x.T @ x) @ x.T @ r[:, [2]]
    v = np.array([v[0, 0], v[1, 0], -1]) # OLS coefficients with -1 for z 

    v /= np.linalg.norm(v)

    return(align(v, ref))

def closest_point(a, b, c, p):
    """Returns the closest point to p on the plane defined by a b and c"""
    m = np.cross(b - a, c - a) # normal vector of plane given by abc
    k = np.dot(m, a) # plane equation is m dot x = k (here x = a)

    c = (k - np.dot(m, p)) / np.dot(m, m) # closest point on ab, n
                                            # plane to p is given by p + cm
    
    return(p + c * m)