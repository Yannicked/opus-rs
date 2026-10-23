//! Kernel microbenchmarks through the crate's public functions.
use opus_rs::bands::{compute_band_energies, denormalise_bands, haar1, renormalise_vector};
use opus_rs::modes::default_mode;
use opus_rs::pvq::{alg_unquant, renormalise_vector as pvq_renorm};
use opus_rs::range_coder::RangeCoder;
use std::hint::black_box;
use std::time::Instant;

fn time<F: FnMut()>(mut f: F, iters: usize) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..15 {
        let t = Instant::now();
        for _ in 0..iters { f(); }
        best = best.min(t.elapsed().as_secs_f64() / iters as f64);
    }
    best * 1e9
}

fn main() {
    let mut rng = 12345u32;
    let mut rnd = || { rng = rng.wrapping_mul(1664525).wrapping_add(1013904223); (rng >> 8) as f32 / 16777216.0 - 0.5 };
    let base: Vec<f32> = (0..2048).map(|_| rnd()).collect();
    let m = default_mode();
    for &n in &[8usize, 16, 24, 36, 48, 96, 176] {
        let mut x = base[..n].to_vec();
        let ns = time(|| { x.copy_from_slice(&base[..n]); renormalise_vector(black_box(&mut x), n, 1.0); }, 20000);
        println!("bands_renorm n={n:3} {ns:8.1} ns");
        let ns = time(|| { x.copy_from_slice(&base[..n]); pvq_renorm(black_box(&mut x), n, 1.0); }, 20000);
        println!("pvq_renorm   n={n:3} {ns:8.1} ns");
        let ns = time(|| { x.copy_from_slice(&base[..n]); haar1(black_box(&mut x), n, 1); }, 20000);
        println!("haar1        n={n:3} {ns:8.1} ns");
    }
    // alg_unquant over a fixed bitstream
    let data: Vec<u8> = (0..1000).map(|i| (i * 37 % 251) as u8).collect();
    for &(n, k) in &[(16usize, 6i32), (36, 10), (96, 20), (176, 40)] {
        let mut x = vec![0f32; n];
        let ns = time(|| { let mut rc = RangeCoder::new_decoder(&data); for _ in 0..10 { alg_unquant(black_box(&mut x), n, k, 0, 1, &mut rc, 1.0); } }, 2000) / 10.0;
        println!("alg_unquant  n={n:3} k={k:2} {ns:8.1} ns");
    }
    for &n in &[24usize, 64, 160, 480] {
        let ns = time(|| { black_box(opus_rs::pitch::inner_prod(black_box(&base[..n]), black_box(&base[7..7 + n]), n)); }, 20000);
        println!("inner_prod   n={n:3} {ns:8.1} ns");
    }
    for &shift in &[0usize, 3] {
        let n = m.mdct.n >> shift;
        let input: Vec<f32> = base.iter().cycle().take(1920).copied().collect();
        let mut out = vec![0f32; n / 2 + m.overlap];
        let ns = time(|| m.mdct.backward(black_box(&input), &mut out, m.window, m.overlap, shift, 1 << shift), 5000);
        println!("mdct_backward shift={shift} {ns:8.1} ns");
    }
    let x: Vec<f32> = base.iter().cycle().take(1920).copied().collect();
    let mut be = vec![0f32; 42];
    let ns = time(|| compute_band_energies(m, black_box(&x), &mut be, 21, 2, 3), 5000);
    println!("band_energies 2ch lm3 {ns:8.1} ns");
    let mut freq = vec![0f32; 1920];
    let ns = time(|| denormalise_bands(m, black_box(&x), &mut freq, &be, 0, 21, 2, 8), 5000);
    println!("denormalise   2ch lm3 {ns:8.1} ns");
}

#[allow(dead_code)]
fn unused() {}
