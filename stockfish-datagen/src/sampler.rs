//! Temperature-controlled softmax over Stockfish MultiPV candidates.
//!
//! `temperature` is in pawns, so a 100-cp gap shifts probabilities by an
//! e-fold at temperature=1.0. `temperature <= 0` falls back to argmax.
//! Single-candidate input short-circuits without consuming RNG state.

use rand::Rng;

use crate::stockfish::Candidate;

/// Pick one move from `candidates`. Returns `None` only if the slice is empty.
///
/// Consumes exactly one `f64` from the RNG when sampling actually happens
/// (i.e. `len > 1` and `temperature > 0`); otherwise the RNG is untouched.
/// This deterministic consumption matters for reproducibility from
/// `game_seed`.
pub fn softmax_sample<'a, R: Rng + ?Sized>(
    candidates: &'a [Candidate],
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
            .max_by(|a, b| a.score_cp.total_cmp(&b.score_cp));
    }

    let max_s = candidates.iter().map(|c| c.score_cp).fold(f32::NEG_INFINITY, f32::max);
    let scale = 100.0 * temperature;

    // Accumulate exponentials in f64 for numerical room — mate scores at
    // ±30000 keep the dynamic range modest but f64 is free here.
    let mut exps = Vec::with_capacity(candidates.len());
    let mut total = 0.0_f64;
    for c in candidates {
        let e = (((c.score_cp - max_s) / scale) as f64).exp();
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
        assert!(softmax_sample(&cands(&[]), 1.0, &mut rng).is_none());
    }

    #[test]
    fn single_returns_only() {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let c = cands(&[42.0]);
        assert_eq!(softmax_sample(&c, 1.0, &mut rng).unwrap().uci, c[0].uci);
    }

    #[test]
    fn zero_temperature_is_argmax() {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let c = cands(&[10.0, 50.0, 30.0]);
        let pick = softmax_sample(&c, 0.0, &mut rng).unwrap();
        assert!((pick.score_cp - 50.0).abs() < 1e-6);
    }

    #[test]
    fn deterministic_under_same_seed() {
        let c = cands(&[10.0, 20.0, 30.0, 40.0, 50.0]);
        let mut rng_a = ChaCha8Rng::seed_from_u64(123);
        let mut rng_b = ChaCha8Rng::seed_from_u64(123);
        for _ in 0..100 {
            let a = softmax_sample(&c, 1.0, &mut rng_a).unwrap().score_cp;
            let b = softmax_sample(&c, 1.0, &mut rng_b).unwrap().score_cp;
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
            let pick = softmax_sample(&c, 1.0, &mut rng).unwrap();
            for (i, ci) in c.iter().enumerate() {
                if ci.uci == pick.uci {
                    counts[i] += 1;
                    break;
                }
            }
        }
        assert!(counts.iter().all(|&c| c > 100), "uneven distribution: {counts:?}");
    }
}
