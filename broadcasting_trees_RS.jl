module P

using QuadGK
using LinearAlgebra, Statistics
using ExtractMacro
using DelimitedFiles

include("common.jl")

###### INTEGRATION ######
const ∞  = 20.0
const dx = 0.01
const interval = map(x -> sign(x)*abs(x)^2, -1:dx:1) .* ∞

∫D(f, int=interval) = quadgk(z -> begin
        r = G(z) * f(z)
        isfinite(r) ? r : 0.0
    end, int..., atol=1e-7, maxevals=10^7)[1]

############### SPECTRUM (binary tree broadcasting) ################

"""
Binary tree broadcasting covariance spectrum at depth L (number of leaf layers),
with θ = 1 - 2eps.

Returns eigenvalues λ_k and normalized weights p_k (sum to 1),
and ν0 = top eigenvalue (aligned with the all-ones vector).
"""
struct Spectrum
    λ::Vector{Float64}
    p::Vector{Float64}
    ν0::Float64
end

function tree_spectrum(eps::Float64, L::Int)
    θ = 1.0 - 2.0*eps
    a = 2.0*θ^2
    denom = 1.0 - a

    λ = zeros(Float64, L)
    for k in 1:L
        # λ_k = (1-θ^2) * sum_{j=0}^{k-1} a^j
        if abs(denom) < 1e-12
            λ[k] = (1.0 - θ^2) * k
        else
            λ[k] = (1.0 - θ^2) * (1.0 - a^k) / denom
        end
    end

    w = zeros(Float64, L)
    for k in 1:(L-1)
        w[k] = 2.0^(L-k)
    end
    w[L] = 2.0
    p = w ./ (2.0^L) # prefactor

    ν0 = λ[L]
    return Spectrum(λ, p, ν0)
end

@inline spec_avg(sp::Spectrum, f) = sum(sp.p .* map(f, sp.λ))

# d = number of leaves = 2^L, θ = 1 - 2eps
s_tree(eps::Float64, L::Int) = sqrt(2.0^L) * (1.0 - 2.0*eps)^L

############### PARAMS ################

mutable struct OrderParams <: AbstractParams
    m::Float64
    q::Float64
    δQ::Float64

    mh::Float64
    qh::Float64
    δQh::Float64
end

mutable struct ExtParams <: AbstractParams
    α::Float64
    λreg::Float64
    s::Float64   # signal scale
    eps::Float64
    L::Int
    sp::Spectrum
end

mutable struct Params <: AbstractParams
    ϵ::Float64  # stopping criterion
    ψ::Float64  # damping
    maxiters::Int
    verb::Int
end

Base.show(io::IO, op::OrderParams) = shortshow(io, op)
Base.show(io::IO, ep::ExtParams)   = shortshow(io, ep)

############### Logistic prox + energetic averages ################

@inline σ(x) = 1.0 / (1.0 + exp(-x))

# solve t = u + δQ * σ(-t) by Newton
function tstar_logistic(u::Float64, δQ::Float64; tol=1e-12, maxit=80)
    t = u
    for _ in 1:maxit
        p  = σ(-t)
        f  = t - u - δQ*p
        fp = 1.0 + δQ*p*(1.0 - p)   # because d/dt σ(-t) = -p(1-p)
        Δ  = f / fp
        t -= Δ
        if abs(Δ) < tol*(1.0 + abs(t))
            return t
        end
    end
    return t
end

# E[-σ(-t*)]
E1(m,q,δQ, s) = ∫D(η -> begin # gaussian integral because u is gaussian
    u = s*m + sqrt(max(q,0.0))*η
    t = tstar_logistic(u, δQ)
    -σ(-t)
end)

# E[σ(-t*)^2]
E2(m,q,δQ, s) = ∫D(η -> begin
    u = s*m + sqrt(max(q,0.0))*η
    t = tstar_logistic(u, δQ)
    p = σ(-t)
    p*p
end)

# E[-σ(-t*) * η]
Eη(m,q,δQ, s) = ∫D(η -> begin
    u = s*m + sqrt(max(q,0.0))*η
    t = tstar_logistic(u, δQ)
    -σ(-t) * η
end)

############### RHS of saddle equations ################
# update hats from (m,q,δQ)
fmh(op, ep) = ep.α * ep.s * E1(op.m, op.q, op.δQ, ep.s)

fqh(op, ep) = - ep.α * E2(op.m, op.q, op.δQ, ep.s)   # derivative wrt δQ + envelope theorem

fδQh(op, ep) = begin
    q = op.q
    if q < 1e-14
        0.0
    else
        (ep.α / sqrt(q)) * Eη(op.m, op.q, op.δQ, ep.s)
    end
end

# update non-hats from hats + spectrum
fm(op, ep) = - op.mh / (ep.λreg + ep.sp.ν0 * op.δQh)

fδQ(op, ep) = spec_avg(ep.sp, ν -> ν / (ep.λreg + ν * op.δQh))

fq(op, ep) = begin
    ν0 = ep.sp.ν0
    term_signal = op.mh^2 * ν0 / (ep.λreg + ν0*op.δQh)^2
    term_noise  = op.qh * spec_avg(ep.sp, ν -> (ν^2) / (ep.λreg + ν*op.δQh)^2)
    term_signal - term_noise
end

# --- loss + Moreau envelope (T=0 energetic term) ---

@inline ℓ_logistic(t::Float64) = log1p(exp(-t))  # stable logistic loss

@inline function moreau_logistic(u::Float64, δQ::Float64)
    t = tstar_logistic(u, δQ)
    return (t - u)^2 / (2.0 * δQ) + ℓ_logistic(t)
end

# \bar g_E(m,q,δQ) = - Eη [ M_{δQ}[ℓ](u(η)) ], u = s m + sqrt(q) η
gEbar(op::OrderParams, ep::ExtParams) = -∫D(η -> begin
    u = ep.s * op.m + sqrt(max(op.q, 0.0)) * η
    moreau_logistic(u, op.δQ)
end)

# \bar g_S( mhat0, qhat, δQhat )
gSbar(op::OrderParams, ep::ExtParams) = begin
    ν0 = ep.sp.ν0
    term1 = -0.5 * spec_avg(ep.sp, ν -> (ν * op.qh) / (ep.λreg + ν * op.δQh))
    term2 =  0.5 * (op.mh^2) / (ep.λreg + ν0 * op.δQh)
    term1 + term2
end

# G_I = mhat0 m + 1/2( q δQhat + δQ qhat )
gI(op::OrderParams, ep::ExtParams) = op.mh * op.m + 0.5 * (op.q * op.δQh + op.δQ * op.qh)

# Full RS potential pieces
GE(op::OrderParams, ep::ExtParams) = ep.α * gEbar(op, ep)
GS(op::OrderParams, ep::ExtParams) = gSbar(op, ep)
GI(op::OrderParams, ep::ExtParams) = gI(op, ep)
FT0(op::OrderParams, ep::ExtParams) = GI(op,ep) + GS(op,ep) + GE(op,ep)

#################  SOLVER  ##################

function converge!(op::OrderParams, ep::ExtParams, pars::Params)
    @extract pars: maxiters verb ϵ ψ
    Δ = Inf
    ok = false
    iters = 0

    for k = 1:maxiters
        iters = k
        Δ = 0.0
        verb > 1 && println("it=$k")

        # conjugates (depend on current overlaps)
        @update op.mh  fmh  Δ ψ verb op ep
        @update op.qh   fqh   Δ ψ verb op ep
        @update op.δQh  fδQh  Δ ψ verb op ep

        # overlaps (depend on conjugates + spectrum)
        @update op.m      fm      Δ ψ verb op ep
        @update op.δQ     fδQ     Δ ψ verb op ep
        @update op.q      fq      Δ ψ verb op ep

        #op.q = max(op.q, 0.0) # clamp (need q nonnegative)

        verb > 1 && println(" Δ=$Δ\n")
        @assert isfinite(Δ)
        ok = (Δ < ϵ)
        ok && break
    end
    return ok, Δ, iters
end

function converge(;
    # initial OPs
    m=0.0, q=0.1, δQ=1.0,
    mh=0.0, qh=0.1, δQh=0.0,
    # problem params
    α=0.2, λreg=1.0, s=1.0,
    eps=0.1, L=13,
    # solver params
    ϵ=1e-6, maxiters=50_000, verb=2, ψ=0.2
)
    sp = tree_spectrum(eps, L)
    op = OrderParams(m,q,δQ, mh,qh,δQh)
    s  = s_tree(eps, L)

    ep  = ExtParams(α, λreg, s, eps, L, sp)

    pars = Params(ϵ, ψ, maxiters, verb)
    ok, Δ, iters = converge!(op, ep, pars)
    return ok, Δ, iters, op, ep, pars
end

# allow α to be either a number or an iterable (range/array)
firstval(x) = x isa Number ? x : first(x)

# helper to flatten OrderParams to a numeric tuple for CSV
function op_row(op)
    return (op.m, op.q, op.δQ, op.mh, op.qh, op.δQh)
end

function span(;
    # initial conditions
    m=0.0, q=0.1, δQ=1.0, mh=0.0, qh=0.1, δQh=0.0,

    # sweep controls
    L = 10:13,
    eps = 0.00:0.01:0.30,
    α = 0.2,
    λreg = 1.0,

    # solver params
    ϵ=1e-6, maxiters=50_000, verb=2, ψ=0.2,

    # output
    resfile = nothing, 
    csvfile = "RS_tree_logistic.csv",

    # behavior
    #continuation = false,
    continuation = true,
    break_on_fail = false,
    retry_with_cold_start = true,
)
    op = OrderParams(m, q, δQ, mh, qh, δQh)
    op0 = deepcopy(op)  # for cold starts

    # create a dummy ep; will be overwritten each grid point
    L0   = first(L)
    eps0 = first(eps)
    sp0  = tree_spectrum(eps0, L0)
    s0  = s_tree(eps0, L0)
    α0   = firstval(α)
    λ0   = (λreg isa AbstractVector) ? first(λreg) : λreg

    ep = ExtParams(α0, λ0, s0, eps0, L0, sp0)
    pars = Params(ϵ, ψ, maxiters, verb)

    return span!(op, op0, ep, pars;
        L=L, eps=eps, α=α, λreg=λreg,
        resfile=resfile, csvfile=csvfile,
        continuation=continuation,
        break_on_fail=break_on_fail,
        retry_with_cold_start=retry_with_cold_start
    )
end


function span!(op::OrderParams, op0::OrderParams, ep::ExtParams, pars::Params;
    L=10:13, eps=0.00:0.01:0.30, α = 0.2, λreg=1.0,
    resfile=nothing, csvfile="RS_tree_logistic.csv",
    #continuation=false,
    continuation=true,
    break_on_fail=false,
    retry_with_cold_start=true
)
    # prepare outputs 
    # CSV header
    if !isfile(csvfile)
        open(csvfile, "w") do io
            println(io, join((
                "L","N","eps","alpha","lambda_reg","s","nu0",
                "ok","Delta","iters",
                "m","q","deltaQ","mh","qh","deltaQh",
                "G_I","G_S","G_E","F_T0"
            ), ","))
        end
    end

    # optional plain-text log header
    if !(resfile === nothing) && !isfile(resfile)
        open(resfile, "w") do f
            allheadersshow(f, ExtParams, "ok", "Δ", "iters", OrderParams)
        end
    end

    results = []

    # sweep
    λgrid = (λreg isa AbstractVector) ? λreg : (λreg,)

    for αcur in α, Lcur in L, epscur in eps, λcur in λgrid

        # implement continuation flag:
        # if false -> reset to cold start every grid point
        if !continuation
            op.m, op.q, op.δQ, op.mh, op.qh, op.δQh = op0.m, op0.q, op0.δQ, op0.mh, op0.qh, op0.δQh
        end
        
        ep.α = αcur
        ep.L = Lcur
        ep.eps = epscur
        ep.λreg = λcur
        ep.sp = tree_spectrum(epscur, Lcur)
        ep.s = s_tree(epscur, Lcur)

        Ncur = big(2)^Lcur 

        println("# NEW ITER: α=$(ep.α), L=$(ep.L), eps=$(ep.eps), λreg=$(ep.λreg) (N=$Ncur)")

        ok, Δ, iters = converge!(op, ep, pars)

        # if fail and we were continuing, optionally retry from cold start once
        if !ok && continuation && retry_with_cold_start
            op.m, op.q, op.δQ, op.mh, op.qh, op.δQh = op0.m, op0.q, op0.δQ, op0.mh, op0.qh, op0.δQh
            ok, Δ, iters = converge!(op, ep, pars)
        end

        push!(results, (ok, deepcopy(ep), Δ, iters, deepcopy(op)))

        # write CSV row
        open(csvfile, "a") do io
            m,q,δQ,mh,qh,δQh = op_row(op)

            GIv = ok ? GI(op, ep) : NaN
            GSv = ok ? GS(op, ep) : NaN
            GEv = ok ? GE(op, ep) : NaN
            Fv  = ok ? (GIv + GSv + GEv) : NaN

            row = (
                ep.L, Ncur, ep.eps, ep.α, ep.λreg, ep.s, ep.sp.ν0,
                ok, Δ, iters,
                m, q, δQ, mh, qh, δQh,
                GIv, GSv, GEv, Fv
            )

            @assert length(row) == 20
            println(io, join(row, ","))
        end

        # write the plainshow log only on success
        if ok && !(resfile === nothing)
            open(resfile, "a") do rf
                println(rf, plainshow(ep), " ", ok, " ", Δ, " ", iters, " ", plainshow(op))
            end
        end

        if !ok && break_on_fail
            break
        end
    end

    return results
end


end # module