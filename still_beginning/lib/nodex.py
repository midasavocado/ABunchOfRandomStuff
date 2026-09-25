"""nodex — tiny expression compiler for Blender shader node trees (EARTH & ORBIT team helper).

Write math with Python operators; each op becomes a Math / Vector Math node. Python numbers are
constant-folded. Works in material, world and group node trees.

    X = Ctx(node_tree)
    d = X.vec(texcoord.outputs['Generated'])
    b = d.dot(o)                  # scalar
    c = (b * b - 3.0).max(0).sqrt()
    col = X.v3((1, .5, .2)) * c   # vector * scalar
    out_socket = col.s
"""
import math

E = math.e


class Ctx:
    def __init__(self, nt, x0=0):
        self.nt = nt
        self.n = nt.nodes
        self.l = nt.links
        self._x = x0
        self._sep = {}

    # --------------------------------------------------------------- node plumbing
    def node(self, t, **props):
        nd = self.n.new(t)
        for k, v in props.items():
            setattr(nd, k, v)
        self._x += 1
        nd.location = ((self._x // 60) * 220, -(self._x % 60) * 60)
        nd.hide = True
        return nd

    def _in(self, sock, v):
        if isinstance(v, (S, Vc)) and isinstance(v.s, (tuple, int, float)):
            v = v.s
        if isinstance(v, (S, Vc)):
            self.l.new(v.s, sock)
        elif isinstance(v, (int, float)):
            sock.default_value = v
        elif isinstance(v, (tuple, list)):
            sock.default_value = tuple(v) if len(v) == len(sock.default_value) else (*v, 1.0)
        else:  # raw socket
            self.l.new(v, sock)

    def f(self, v):
        """Wrap a raw socket / number as scalar."""
        if isinstance(v, S):
            return v
        if isinstance(v, (int, float)):
            return self.value(float(v))[0]
        return S(self, v)

    def vec(self, v):
        if isinstance(v, Vc):
            return v
        if isinstance(v, (tuple, list)):
            return Vc(self, tuple(float(x) for x in v))
        return Vc(self, v)

    v3 = vec

    def value(self, v=0.0, name=None):
        nd = self.node('ShaderNodeValue')
        nd.outputs[0].default_value = v
        if name:
            nd.name = nd.label = name
        return S(self, nd.outputs[0]), nd

    def combine(self, x, y, z):
        if all(isinstance(t, (int, float)) for t in (x, y, z)):
            return Vc(self, (float(x), float(y), float(z)))
        nd = self.node('ShaderNodeCombineXYZ')
        for i, t in enumerate((x, y, z)):
            self._in(nd.inputs[i], t)
        return Vc(self, nd.outputs[0])

    def vinput(self, v=(0, 0, 0), name=None):
        """Animatable vector parameter (CombineXYZ with default values)."""
        nd = self.node('ShaderNodeCombineXYZ')
        for i in range(3):
            nd.inputs[i].default_value = v[i]
        if name:
            nd.name = nd.label = name
        return Vc(self, nd.outputs[0]), nd

    def math(self, op, a, b=None, c=None, clamp=False):
        vals = [a, b, c]
        if all(isinstance(t, (int, float)) or t is None for t in vals):
            r = _fold(op, a, b, c)
            if r is not None:
                return r
        nd = self.node('ShaderNodeMath', operation=op, use_clamp=clamp)
        for i, t in enumerate(vals):
            if t is not None:
                self._in(nd.inputs[i], t)
        return S(self, nd.outputs[0])

    def vmath(self, op, a, b=None, c=None, scale=None):
        nd = self.node('ShaderNodeVectorMath', operation=op)
        for i, t in enumerate((a, b, c)):
            if t is None:
                continue
            if isinstance(t, (int, float)):
                t = (t, t, t)
            self._in(nd.inputs[i], t)
        if scale is not None:
            self._in(nd.inputs[3], scale)
        if op in ('DOT_PRODUCT', 'LENGTH', 'DISTANCE'):
            return S(self, nd.outputs[1])
        return Vc(self, nd.outputs[0])

    def sep(self, v):
        key = v.s if not isinstance(v.s, tuple) else None
        if key is not None and key in self._sep:
            return self._sep[key]
        nd = self.node('ShaderNodeSeparateXYZ')
        self._in(nd.inputs[0], v)
        r = (S(self, nd.outputs[0]), S(self, nd.outputs[1]), S(self, nd.outputs[2]))
        if key is not None:
            self._sep[key] = r
        return r

    # --------------------------------------------------------------- helpers
    def mix(self, t, a, b):
        """a*(1-t)+b*t for scalars or vectors."""
        if isinstance(a, Vc) or isinstance(b, Vc) or isinstance(a, tuple) or isinstance(b, tuple):
            a = self.vec(a); b = self.vec(b)
            return a + (b - a) * t
        return a + (b - a) * t if isinstance(a, S) or isinstance(b, S) or isinstance(t, S) else a + (b - a) * t

    def smoothstep(self, e0, e1, x):
        t = ((self.f(x) - e0) / (e1 - e0)).clamp01()
        return t * t * (t * -2.0 + 3.0)

    def lin(self, e0, e1, x):
        return ((self.f(x) - e0) / (e1 - e0)).clamp01()

    def noise(self, vec, scale=1.0, detail=4.0, rough=0.5, lac=2.0, dist=0.0, ntype='FBM', dim='3D',
              offset=0.0, gain=1.0, normalize=True):
        nd = self.node('ShaderNodeTexNoise')
        nd.noise_dimensions = dim
        nd.noise_type = ntype
        nd.normalize = normalize
        self._in(nd.inputs['Vector'], vec)
        for k, v in (('Scale', scale), ('Detail', detail), ('Roughness', rough), ('Lacunarity', lac),
                     ('Distortion', dist), ('Offset', offset), ('Gain', gain)):
            sk = nd.inputs.get(k)
            if sk is None:
                sk = [i for i in nd.inputs if i.name == k]
                sk = sk[0] if sk else None
            if sk is not None:
                self._in(sk, v)
        return S(self, nd.outputs['Fac']), Vc(self, nd.outputs['Color'])

    def voronoi(self, vec, scale=1.0, feature='F1', rand=1.0, metric='EUCLIDEAN', detail=0.0, rough=0.5):
        nd = self.node('ShaderNodeTexVoronoi')
        nd.voronoi_dimensions = '3D'
        nd.feature = feature
        nd.distance = metric
        self._in(nd.inputs['Vector'], vec)
        self._in(nd.inputs['Scale'], scale)
        self._in(nd.inputs['Randomness'], rand)
        try:
            self._in(nd.inputs['Detail'], detail)
            self._in(nd.inputs['Roughness'], rough)
        except KeyError:
            pass
        return (S(self, nd.outputs['Distance']), Vc(self, nd.outputs['Color']), Vc(self, nd.outputs['Position']))

    def rotate_euler(self, v, rot):
        nd = self.node('ShaderNodeVectorRotate', rotation_type='EULER_XYZ')
        self._in(nd.inputs['Vector'], v)
        self._in(nd.inputs['Rotation'], rot)
        return Vc(self, nd.outputs[0]), nd

    def where(self, cond, a, b):
        """cond in [0,1] -> mix(b, a) (a when cond=1)."""
        return self.mix(cond, b, a)


def _fold(op, a, b, c):
    try:
        if op == 'ADD': return a + b
        if op == 'SUBTRACT': return a - b
        if op == 'MULTIPLY': return a * b
        if op == 'DIVIDE': return a / b if b else 0.0
        if op == 'POWER': return a ** b
        if op == 'SQRT': return math.sqrt(max(a, 0))
        if op == 'EXPONENT': return math.exp(a)
        if op == 'MINIMUM': return min(a, b)
        if op == 'MAXIMUM': return max(a, b)
        if op == 'ABSOLUTE': return abs(a)
        if op == 'MULTIPLY_ADD': return a * b + c
    except Exception:
        return None
    return None


class S:
    """Scalar expression (socket or python float)."""
    def __init__(self, X, s):
        self.X = X
        self.s = s

    def _o(self, op, other, rev=False):
        a, b = (other, self) if rev else (self, other)
        if isinstance(other, Vc) or isinstance(other, tuple):
            ov = self.X.vec(other)
            if op == 'MULTIPLY':
                return ov * self
            av = self.X.combine(self, self, self)
            return self.X.vmath({'ADD': 'ADD', 'SUBTRACT': 'SUBTRACT', 'DIVIDE': 'DIVIDE'}[op], ov if rev else av, av if rev else ov)
        return self.X.math(op, a if not isinstance(a, S) else a, b)

    def __add__(self, o): return self._o('ADD', o)
    def __radd__(self, o): return self._o('ADD', o, True)
    def __sub__(self, o): return self._o('SUBTRACT', o)
    def __rsub__(self, o): return self._o('SUBTRACT', o, True)
    def __mul__(self, o): return self._o('MULTIPLY', o)
    def __rmul__(self, o): return self._o('MULTIPLY', o, True)
    def __truediv__(self, o): return self._o('DIVIDE', o)
    def __rtruediv__(self, o): return self._o('DIVIDE', o, True)
    def __pow__(self, o): return self._o('POWER', o)
    def __neg__(self): return self.X.math('MULTIPLY', self, -1.0)

    def exp(self): return self.X.math('EXPONENT', self)
    def sqrt(self): return self.X.math('SQRT', self)
    def abs(self): return self.X.math('ABSOLUTE', self)
    def log(self, base=E): return self.X.math('LOGARITHM', self, base)
    def max(self, o): return self.X.math('MAXIMUM', self, o)
    def min(self, o): return self.X.math('MINIMUM', self, o)
    def clamp01(self): return self.X.math('ADD', self, 0.0, clamp=True)
    def clamp(self, a, b): return self.max(a).min(b)
    def sin(self): return self.X.math('SINE', self)
    def cos(self): return self.X.math('COSINE', self)
    def atan2(self, o): return self.X.math('ARCTAN2', self, o)
    def acos(self): return self.X.math('ARCCOSINE', self)
    def fract(self): return self.X.math('FRACT', self)
    def floor(self): return self.X.math('FLOOR', self)
    def smax(self, o, k): return self.X.math('SMOOTH_MAX', self, o, k)
    def smin(self, o, k): return self.X.math('SMOOTH_MIN', self, o, k)
    def madd(self, m, a): return self.X.math('MULTIPLY_ADD', self, m, a)
    def gt(self, o): return self.X.math('GREATER_THAN', self, o)
    def lt(self, o): return self.X.math('LESS_THAN', self, o)
    def tanh(self): return self.X.math('TANH', self)


class Vc:
    """Vector / colour expression."""
    def __init__(self, X, s):
        self.X = X
        self.s = s  # socket or tuple

    def _o(self, op, other, rev=False):
        X = self.X
        if isinstance(other, (int, float)):
            if op == 'MULTIPLY':
                if isinstance(self.s, tuple):
                    return Vc(X, tuple(c * other for c in self.s))
                return X.vmath('SCALE', self, scale=other)
            if op == 'DIVIDE' and not rev:
                return self._o('MULTIPLY', 1.0 / other)
            other = (other, other, other)
        if isinstance(other, S):
            if op == 'MULTIPLY':
                if isinstance(self.s, tuple):
                    return X.vmath('SCALE', self, scale=other)
                return X.vmath('SCALE', self, scale=other)
            other = X.combine(other, other, other)
        if isinstance(other, tuple) and isinstance(self.s, tuple):
            f = {'ADD': lambda a, b: a + b, 'SUBTRACT': lambda a, b: a - b, 'MULTIPLY': lambda a, b: a * b,
                 'DIVIDE': lambda a, b: a / b if b else 0.0}[op]
            a, b = (other, self.s) if rev else (self.s, other)
            return Vc(X, tuple(f(p, q) for p, q in zip(a, b)))
        a, b = (other, self) if rev else (self, other)
        return X.vmath(op, a, b)

    def __add__(self, o): return self._o('ADD', o)
    def __radd__(self, o): return self._o('ADD', o, True)
    def __sub__(self, o): return self._o('SUBTRACT', o)
    def __rsub__(self, o): return self._o('SUBTRACT', o, True)
    def __mul__(self, o): return self._o('MULTIPLY', o)
    def __rmul__(self, o): return self._o('MULTIPLY', o, True)
    def __truediv__(self, o): return self._o('DIVIDE', o)
    def __rtruediv__(self, o): return self._o('DIVIDE', o, True)
    def __neg__(self): return self._o('MULTIPLY', -1.0)

    def dot(self, o): return self.X.vmath('DOT_PRODUCT', self, self.X.vec(o))
    def cross(self, o): return self.X.vmath('CROSS_PRODUCT', self, self.X.vec(o))
    def length(self): return self.X.vmath('LENGTH', self)
    def normalize(self): return self.X.vmath('NORMALIZE', self)
    def vmax(self, o): return self.X.vmath('MAXIMUM', self, o)
    def vmin(self, o): return self.X.vmath('MINIMUM', self, o)
    def vpow(self, o): return self.X.vmath('POWER', self, o)

    def exp(self):
        """component-wise e^v"""
        return self.X.vmath('POWER', (E, E, E), self)

    @property
    def x(self): return self.X.sep(self)[0]
    @property
    def y(self): return self.X.sep(self)[1]
    @property
    def z(self): return self.X.sep(self)[2]
