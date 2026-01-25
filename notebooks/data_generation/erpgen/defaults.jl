const DEFAULT_PINK_NOISELEVEL_MEAN = 15
const DEFAULT_PINK_NOISELEVEL_SD = 2
const DEFAULT_WHITE_NOISELEVEL_MEAN = 15
const DEFAULT_WHITE_NOISELEVEL_SD = 2
const DEFAULT_RED_NOISELEVEL_MEAN = 15
const DEFAULT_RED_NOISELEVEL_SD = 2
const DEFAULT_EXP_NOISELEVEL_MEAN = 15
const DEFAULT_EXP_NOISELEVEL_SD = 2
const DEFAULT_EXP_NOISE_TAU = 10
const DEFAULT_NOISELEVEL_DISTS = Dict(
    PinkNoise => Normal(DEFAULT_PINK_NOISELEVEL_MEAN, DEFAULT_PINK_NOISELEVEL_SD),
    WhiteNoise => Normal(DEFAULT_WHITE_NOISELEVEL_MEAN, DEFAULT_WHITE_NOISELEVEL_SD),
    RedNoise => Normal(DEFAULT_RED_NOISELEVEL_MEAN, DEFAULT_RED_NOISELEVEL_SD),
    ExponentialNoise => Normal(DEFAULT_EXP_NOISELEVEL_MEAN, DEFAULT_EXP_NOISELEVEL_SD),
)
const DEFAULT_CROP_MEAN_MS = 100
const DEFAULT_CROP_SD_MS = 25
const DEFAULT_CROP_DIST = Normal(DEFAULT_CROP_MEAN_MS, DEFAULT_CROP_SD_MS)
const DEFAULT_DROPOUT_RATE_MEAN = 2000
const DEFAULT_DROPOUT_RATE_SD = 250
const DEFAULT_NOISE_POOL = [PinkNoise(), WhiteNoise(), RedNoise(), ExponentialNoise(τ = DEFAULT_EXP_NOISE_TAU)]
const FILTER_BORDER = "reflect"

const RESIZE_METHOD_SPECS = [
    (name = :nearest,
        method = Interpolations.Constant(),
        params = "none",
        notes = "Nearest-neighbor (piecewise constant). Equivalent to BSpline(Constant())."),
    (name = :linear,
        method = Interpolations.Linear(),
        params = "none",
        notes = "Linear/bilinear interpolation. Equivalent to BSpline(Linear())."),
    (name = :quadratic_line_ongrid,
        method = Interpolations.Quadratic(Interpolations.Line(Interpolations.OnGrid())),
        params = "bc::BoundaryCondition (Flat/Line/Free/Reflect/Periodic/Throw) + gridstyle (OnGrid/OnCell)",
        notes = "Quadratic B-spline with boundary condition and gridstyle."),
    (name = :cubic_line_ongrid,
        method = Interpolations.Cubic(Interpolations.Line(Interpolations.OnGrid())),
        params = "bc::BoundaryCondition (Flat/Line/Free/Reflect/Periodic/Throw) + gridstyle (OnGrid/OnCell)",
        notes = "Cubic B-spline with boundary condition and gridstyle."),
    (name = :lanczos4_opencv,
        method = Lanczos4OpenCV(),
        params = "none",
        notes = "Lanczos 4 windowed-sinc interpolation (OpenCV compatible)."),
]
const DEFAULT_RESIZE_METHODS = map(spec -> spec.method, RESIZE_METHOD_SPECS)
