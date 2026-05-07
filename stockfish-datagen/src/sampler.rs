//! Temperature-controlled softmax over Stockfish MultiPV / evallegal candidates.
//!
//! `temperature` is interpreted in pawn-units (`scale = 100 * T`), so a 100-cp
//! gap shifts probabilities by an e-fold at `temperature=1.0`. `temperature <= 0`
//! falls back to argmax. Single-candidate input short-circuits without consuming
//! RNG state.
//!
//! `sample_score` chooses which per-move score to softmax over: `Cp`
//! (normalized centipawns; default) or `V` (raw NNUE Value, only available
//! from the `evallegal` protocol). The 100-factor in `scale` is constant
//! across both modes — meaning the same nominal `T` is *sharper* under `V`
//! because raw `v` magnitudes are 3–5× larger than `cp`. Pick higher `T`
//! under `V` if you want comparable exploration.

use rand::Rng;

use crate::config::SampleScore;
use crate::stockfish::Candidate;

/// Read the sampling score off a candidate per the chosen `SampleScore`.
/// `V` mode panics on `None` — config validation guarantees this only
/// happens when the candidates came from the `evallegal` protocol, where
/// `score_v` is always populated.
fn pick_score(c: &Candidate, score: SampleScore) -> f32 {
    match score {
        SampleScore::Cp => c.score_cp,
        SampleScore::V => c
            .score_v
            .expect("sample_score=V requires searchless=true; config validation should prevent this"),
    }
}

/// Pick one move from `candidates`. Returns `None` only if the slice is empty.
///
/// Consumes exactly one `f64` from the RNG when sampling actually happens
/// (i.e. `len > 1` and `temperature > 0`); otherwise the RNG is untouched.
/// This deterministic consumption matters for reproducibility from
/// `game_seed`.
pub fn softmax_sample<'a, R: Rng + ?Sized>(
    candidates: &'a [Candidate],
    score: SampleScore,
    temperature: f32,
    rng: &mut R,
) -> Option<&'a Candidate> {
    if candidates.is_empty() {
        return None;
    }
    if candidates.len() == 1 {
        return Some(&candidates[0]);
    }
    if temperature <= 0.0 {
        // argmax — deterministic, no RNG draw.
        return candidates
            .iter()
            .max_by(|a, b| pick_score(a, score).total_cmp(&pick_score(b, score)));
    }

    let max_s = candidates.iter().map(|c| pick_score(c, score)).fold(f32::NEG_INFINITY, f32::max);
    let scale = 100.0 * temperature;

    // Accumulate exponentials in f64 for numerical room — mate scores at
    // ±30000 keep the dynamic range modest but f64 is free here.
    let mut exps = Vec::with_capacity(candidates.len());
    let mut total = 0.0_f64;
    for c in candidates {
        let e = (((pick_score(c, score) - max_s) / scale) as f64).exp();
        total += e;
        exps.push(e);
    }

    let r: f64 = rng.gen::<f64>() * total;
    let mut acc = 0.0_f64;
    for (i, e) in exps.iter().enumerate() {
        acc += e;
        if r <= acc {
            return Some(&candidates[i]);
        }
    }
    // Numerical fall-through: pick the last (essentially never hit).
    Some(candidates.last().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn cands(scores: &[f32]) -> Vec<Candidate> {
        scores
            .iter()
            .enumerate()
            .map(|(i, &s)| Candidate {
                uci: format!("a{}b{}", (i % 8) + 1, ((i + 1) % 8) + 1),
                score_cp: s,
                score_v: None,
            })
            .collect()
    }

    #[test]
    fn empty_returns_none() {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        assert!(softmax_sample(&cands(&[]), SampleScore::Cp, 1.0, &mut rng).is_none());
    }

    #[test]
    fn single_returns_only() {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let c = cands(&[42.0]);
        assert_eq!(
            softmax_sample(&c, SampleScore::Cp, 1.0, &mut rng).unwrap().uci,
            c[0].uci,
        );
    }

    #[test]
    fn zero_temperature_is_argmax() {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let c = cands(&[10.0, 50.0, 30.0]);
        let pick = softmax_sample(&c, SampleScore::Cp, 0.0, &mut rng).unwrap();
        assert!((pick.score_cp - 50.0).abs() < 1e-6);
    }

    #[test]
    fn deterministic_under_same_seed() {
        let c = cands(&[10.0, 20.0, 30.0, 40.0, 50.0]);
        let mut rng_a = ChaCha8Rng::seed_from_u64(123);
        let mut rng_b = ChaCha8Rng::seed_from_u64(123);
        for _ in 0..100 {
            let a = softmax_sample(&c, SampleScore::Cp, 1.0, &mut rng_a).unwrap().score_cp;
            let b = softmax_sample(&c, SampleScore::Cp, 1.0, &mut rng_b).unwrap().score_cp;
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn high_temperature_distributes_picks() {
        // With temperature high, all candidates should be picked at least
        // once over many trials.
        let c = cands(&[0.0, 0.0, 0.0, 0.0]);
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let mut counts = [0u32; 4];
        for _ in 0..1000 {
            let pick = softmax_sample(&c, SampleScore::Cp, 1.0, &mut rng).unwrap();
            for (i, ci) in c.iter().enumerate() {
                if ci.uci == pick.uci {
                    counts[i] += 1;
                    break;
                }
            }
        }
        assert!(counts.iter().all(|&c| c > 100), "uneven distribution: {counts:?}");
    }

    /// V mode reads `score_v`, so a candidate with `score_v = Some(huge)` and
    /// `score_cp = 0` should be argmax-picked at T=0 even though the cp tie
    /// would otherwise let any candidate win.
    #[test]
    fn v_mode_uses_score_v_not_cp() {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        // All cp=0 (tied), but the second move has v=1000 (clearly best).
        let c = vec![
            Candidate { uci: "a1a2".into(), score_cp: 0.0, score_v: Some(0.0) },
            Candidate { uci: "b1b2".into(), score_cp: 0.0, score_v: Some(1000.0) },
            Candidate { uci: "c1c2".into(), score_cp: 0.0, score_v: Some(-500.0) },
        ];
        let pick = softmax_sample(&c, SampleScore::V, 0.0, &mut rng).unwrap();
        assert_eq!(pick.uci, "b1b2");
    }

    /// Same candidate set under Cp mode picks based on cp ties only — argmax
    /// is well-defined to be one of them; what we actually check is that V
    /// picks differ from Cp picks when scores diverge.
    #[test]
    fn v_mode_diverges_from_cp_mode() {
        // cp prefers a1a2 (cp=10), v prefers b1b2 (v=100).
        let c = vec![
            Candidate { uci: "a1a2".into(), score_cp: 10.0, score_v: Some(20.0) },
            Candidate { uci: "b1b2".into(), score_cp: 5.0, score_v: Some(100.0) },
        ];
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let cp_pick = softmax_sample(&c, SampleScore::Cp, 0.0, &mut rng).unwrap();
        let v_pick = softmax_sample(&c, SampleScore::V, 0.0, &mut rng).unwrap();
        assert_eq!(cp_pick.uci, "a1a2");
        assert_eq!(v_pick.uci, "b1b2");
    }
}
