// pvq_search on an all-zero band with K > 4 and N not a multiple of 8.
// The AVX2 path picks a padding lane (index >= N) on the score tie and
// writes the pulse past the end of `y`.
fn main() {
    for n in [9usize, 12, 15, 20, 31] {
        let x = vec![0.0f32; n];
        // Guard words after y so the stray write is visible.
        let mut buf = vec![0i32; n + 16];
        let (y, guard) = buf.split_at_mut(n);
        opus_rs::pvq::pvq_search(&x, y, 5, n);
        let sum: i32 = y.iter().map(|v| v.abs()).sum();
        let hit: Vec<(usize, i32)> = guard.iter().enumerate().filter(|(_, v)| **v != 0).map(|(i, v)| (n + i, *v)).collect();
        println!("n={n:2} K=5: sum|y| = {sum}, writes past y: {hit:?}");
    }
}
