//! Bit-identity and timing harness for opus-rs safety work.
//!
//! `h hash <wav>`  : print a hash per config of encoder bytes and decoder
//!                   samples (clean, lossy, mutated and random packets).
//! `h bench <wav> <reps>` : time decode-only and encode-only loops.
use opus_rs::{Application, OpusDecoder, OpusEncoder};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

struct Fnv(u64);
impl Fnv {
    fn new() -> Self {
        Fnv(0xcbf29ce484222325)
    }
    fn bytes(&mut self, b: &[u8]) {
        for &x in b {
            self.0 ^= x as u64;
            self.0 = self.0.wrapping_mul(0x100000001b3);
        }
    }
    fn f32s(&mut self, s: &[f32]) {
        for &x in s {
            self.bytes(&x.to_bits().to_le_bytes());
        }
    }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
}

fn read_wav(path: &str) -> Vec<f32> {
    let b = std::fs::read(path).expect("wav");
    let mut i = 12;
    while i + 8 <= b.len() {
        let len = u32::from_le_bytes(b[i + 4..i + 8].try_into().unwrap()) as usize;
        if &b[i..i + 4] == b"data" {
            return b[i + 8..(i + 8 + len).min(b.len())]
                .chunks_exact(2)
                .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
                .collect();
        }
        i += 8 + len;
    }
    panic!("no data chunk")
}

/// Signal at `sr` Hz, `ch` channels, from the 16 kHz mono source: linear
/// interpolation, plus a second channel that is a delayed, scaled copy mixed
/// with a tone so stereo coding has real side information.
fn signal(src: &[f32], sr: u32, ch: usize, secs: f32) -> Vec<f32> {
    let n = (sr as f32 * secs) as usize;
    let mut out = Vec::with_capacity(n * ch);
    for i in 0..n {
        let t = i as f64 * 16000.0 / sr as f64;
        let k = t as usize % (src.len() - 1);
        let f = (t - t.floor()) as f32;
        let s = src[k] * (1.0 - f) + src[k + 1] * f;
        out.push(s);
        if ch == 2 {
            let k2 = (k + src.len() - 37) % (src.len() - 1);
            let tone = (i as f64 * 2.0 * std::f64::consts::PI * 1234.0 / sr as f64).sin() as f32;
            out.push(src[k2] * 0.6 + tone * 0.05);
        }
    }
    out
}

#[derive(Clone, Copy, Debug)]
struct Cfg {
    sr: u32,
    ch: usize,
    app: Application,
    br: i32,
    cx: i32,
    fms10: usize, // frame length in tenths of ms
    cbr: bool,
    fec: bool,
    hybrid: bool,
    i16in: bool,
}

fn configs() -> Vec<Cfg> {
    let mut v = Vec::new();
    let base = Cfg {
        sr: 16000,
        ch: 1,
        app: Application::Voip,
        br: 20000,
        cx: 10,
        fms10: 200,
        cbr: false,
        fec: false,
        hybrid: false,
        i16in: false,
    };
    for &sr in &[8000u32, 12000, 16000, 24000, 48000] {
        for &ch in &[1usize, 2] {
            for &(app, br) in &[
                (Application::Voip, 12000),
                (Application::Voip, 24000),
                (Application::Audio, 64000),
                (Application::Audio, 128000),
                (Application::RestrictedLowDelay, 48000),
            ] {
                for &cx in &[0, 5, 10] {
                    v.push(Cfg { sr, ch, app, br, cx, ..base });
                }
            }
        }
    }
    for &fms10 in &[25usize, 50, 100, 200, 400, 600] {
        for &sr in &[16000u32, 48000] {
            for &app in &[Application::Voip, Application::Audio] {
                v.push(Cfg { sr, fms10, app, br: 32000, ..base });
            }
        }
    }
    for &br in &[6000, 8000, 10000, 16000, 32000] {
        v.push(Cfg { br, fec: true, ..base });
        v.push(Cfg { br, fec: true, cbr: true, sr: 8000, ..base });
        v.push(Cfg { br, cbr: true, ch: 2, sr: 48000, ..base });
    }
    for &br in &[24000, 32000, 48000] {
        v.push(Cfg { br, hybrid: true, sr: 48000, ..base });
        v.push(Cfg { br, hybrid: true, sr: 24000, ch: 2, ..base });
        v.push(Cfg { br, hybrid: true, sr: 48000, i16in: true, ..base });
    }
    v.push(Cfg { i16in: true, ..base });
    v.push(Cfg { i16in: true, sr: 48000, ch: 2, app: Application::Audio, br: 96000, ..base });
    // fms 2.5/5 only valid for CELT; the encoder rejects others, fine.
    v
}

fn encode_all(cfg: &Cfg, pcm: &[f32]) -> Result<Vec<Vec<u8>>, String> {
    let mut enc = OpusEncoder::new(cfg.sr as i32, cfg.ch, cfg.app).map_err(|e| e.to_string())?;
    enc.bitrate_bps = cfg.br;
    enc.complexity = cfg.cx;
    enc.use_cbr = cfg.cbr;
    enc.use_inband_fec = cfg.fec;
    if cfg.fec {
        enc.packet_loss_perc = 10;
    }
    if cfg.hybrid {
        enc.enable_hybrid_mode().map_err(|e| e.to_string())?;
    }
    let fs = cfg.sr as usize * cfg.fms10 / 10000;
    let mut out = vec![0u8; 1500];
    let mut pk = Vec::new();
    for frame in pcm.chunks_exact(fs * cfg.ch) {
        let r = if cfg.i16in {
            let v: Vec<i16> = frame.iter().map(|&x| (x * 32768.0).clamp(-32768.0, 32767.0) as i16).collect();
            enc.encode_i16(&v, fs, &mut out)
        } else {
            enc.encode(frame, fs, &mut out)
        };
        match r {
            Ok(n) => pk.push(out[..n].to_vec()),
            Err(e) => return Err(e.to_string()),
        }
    }
    Ok(pk)
}

/// Decode a packet list; `drop_every` loses every n-th packet. Hashes samples
/// and the Ok/Err/panic outcome of each call.
fn decode_all(sr: u32, ch: usize, fs: usize, pk: &[Vec<u8>], drop_every: usize, h: &mut Fnv) {
    let mut dec = OpusDecoder::new(sr as i32, ch).unwrap();
    let mut pcm = vec![0f32; 5760 * 2];
    for (i, p) in pk.iter().enumerate() {
        let inp: &[u8] = if drop_every > 0 && i % drop_every == drop_every - 1 { &[] } else { p };
        let r = catch_unwind(AssertUnwindSafe(|| dec.decode(inp, fs, &mut pcm)));
        match r {
            Ok(Ok(n)) => {
                h.bytes(b"ok");
                h.f32s(&pcm[..(n * ch).min(pcm.len())]);
            }
            Ok(Err(e)) => {
                h.bytes(b"err");
                h.bytes(e.as_bytes());
            }
            Err(_) => {
                h.bytes(b"panic");
                // a panicked decoder may be in a broken state; start over
                dec = OpusDecoder::new(sr as i32, ch).unwrap();
            }
        }
    }
}

fn hash_mode(wav: &str) {
    if std::env::var("SHOWPANIC").is_err() {
        std::panic::set_hook(Box::new(|_| {}));
    }
    let src = read_wav(wav);
    let mut total = Fnv::new();
    let mut all_packets: Vec<Vec<u8>> = Vec::new();
    let only: Option<usize> = std::env::var("CFG").ok().map(|v| v.parse().unwrap());
    // CORPUS_OUT writes this build's packets; CORPUS_IN decodes a saved set
    // instead, so decoders can be compared on identical input even where
    // the encoders differ.
    let corpus_in: Option<Vec<u8>> = std::env::var("CORPUS_IN").ok().map(|p| std::fs::read(p).unwrap());
    let mut cin = 0usize;
    let mut corpus_out: Vec<u8> = Vec::new();
    for (ci, cfg) in configs().iter().enumerate() {
        if only.is_some_and(|o| o != ci) {
            continue;
        }
        let pcm = signal(&src, cfg.sr, cfg.ch, 3.0);
        let mut he = Fnv::new();
        let mut hd = Fnv::new();
        let enc = encode_all(cfg, &pcm);
        match &enc {
            Ok(pk) => {
                for p in pk {
                    he.bytes(&(p.len() as u32).to_le_bytes());
                    he.bytes(p);
                }
            }
            Err(e) => he.bytes(e.as_bytes()),
        }
        let pk: Option<Vec<Vec<u8>>> = if let Some(c) = &corpus_in {
            let rd = |c: &[u8], at: &mut usize| {
                let v = u32::from_le_bytes(c[*at..*at + 4].try_into().unwrap());
                *at += 4;
                v
            };
            let count = rd(c, &mut cin);
            if count == u32::MAX {
                None
            } else {
                let mut v = Vec::new();
                for _ in 0..count {
                    let n = rd(c, &mut cin) as usize;
                    v.push(c[cin..cin + n].to_vec());
                    cin += n;
                }
                Some(v)
            }
        } else {
            enc.ok()
        };
        match &pk {
            Some(pk) => {
                corpus_out.extend((pk.len() as u32).to_le_bytes());
                for p in pk {
                    corpus_out.extend((p.len() as u32).to_le_bytes());
                    corpus_out.extend(p);
                }
            }
            None => corpus_out.extend(u32::MAX.to_le_bytes()),
        }
        if let Some(pk) = pk {
            let fs = cfg.sr as usize * cfg.fms10 / 10000;
            decode_all(cfg.sr, cfg.ch, fs, &pk, 0, &mut hd);
            decode_all(48000, 2, 48 * cfg.fms10 / 10, &pk, 0, &mut hd);
            decode_all(cfg.sr, cfg.ch, fs, &pk, 5, &mut hd);
            decode_all(8000, 1, 8 * cfg.fms10 / 10, &pk, 3, &mut hd);
            all_packets.extend(pk.into_iter().step_by(7));
        }
        println!("cfg{ci:03} enc={:016x} dec={:016x} {:?}", he.0, hd.0, cfg);
        total.bytes(&he.0.to_le_bytes());
        total.bytes(&hd.0.to_le_bytes());
    }
    if let Ok(p) = std::env::var("CORPUS_OUT") {
        std::fs::write(p, &corpus_out).unwrap();
    }
    if only.is_some() {
        return;
    }
    // Hostile packets: mutated real packets and random bytes.
    let mut rng = Rng(0x5eed);
    let mut hm = Fnv::new();
    let mut panics = 0;
    for &(sr, ch) in &[(48000u32, 2usize), (16000, 1), (8000, 2), (24000, 1)] {
        let mut dec = OpusDecoder::new(sr as i32, ch).unwrap();
        let mut pcm = vec![0f32; 5760 * 2];
        for round in 0..40000 {
            let mut p: Vec<u8> = if round % 4 == 0 {
                let n = 1 + rng.next() as usize % 300;
                (0..n).map(|_| rng.next() as u8).collect()
            } else {
                let mut p = all_packets[rng.next() as usize % all_packets.len()].clone();
                let flips = 1 + rng.next() % 6;
                for _ in 0..flips {
                    let i = rng.next() as usize % p.len();
                    p[i] ^= 1 << (rng.next() % 8);
                }
                if round % 5 == 1 && p.len() > 2 {
                    let n = 1 + rng.next() as usize % (p.len() - 1);
                    p.truncate(n);
                }
                p
            };
            if round % 3 == 0 {
                p[0] = rng.next() as u8;
            }
            let fs = sr as usize * 120 / 1000;
            let r = catch_unwind(AssertUnwindSafe(|| dec.decode(&p, fs, &mut pcm)));
            match r {
                Ok(Ok(n)) => {
                    hm.bytes(b"ok");
                    hm.f32s(&pcm[..(n * ch).min(pcm.len())]);
                }
                Ok(Err(e)) => hm.bytes(e.as_bytes()),
                Err(_) => {
                    panics += 1;
                    hm.bytes(b"panic");
                    dec = OpusDecoder::new(sr as i32, ch).unwrap();
                }
            }
        }
    }
    println!("hostile dec={:016x} panics={panics}", hm.0);
    total.bytes(&hm.0.to_le_bytes());
    println!("TOTAL {:016x}", total.0);
}

fn bench_cases() -> Vec<(&'static str, Cfg)> {
    vec![
        ("silk_nb", Cfg { sr: 8000, ch: 1, app: Application::Voip, br: 12000, cx: 10, fms10: 200, cbr: false, fec: false, hybrid: false, i16in: false }),
        ("silk_wb", Cfg { sr: 16000, ch: 1, app: Application::Voip, br: 24000, cx: 10, fms10: 200, cbr: false, fec: true, hybrid: false, i16in: false }),
        ("hybrid_fb", Cfg { sr: 48000, ch: 1, app: Application::Voip, br: 32000, cx: 10, fms10: 200, cbr: false, fec: false, hybrid: true, i16in: false }),
        ("celt_fb_st", Cfg { sr: 48000, ch: 2, app: Application::Audio, br: 128000, cx: 10, fms10: 200, cbr: false, fec: false, hybrid: false, i16in: false }),
        ("celt_fb_10ms", Cfg { sr: 48000, ch: 1, app: Application::Audio, br: 64000, cx: 10, fms10: 100, cbr: false, fec: false, hybrid: false, i16in: false }),
    ]
}

/// Run one case's encode or decode loop `iters` times, for counting
/// instructions with `perf stat` (setup cost is measured with iters = 0).
fn run_mode(wav: &str, case: &str, kind: &str, iters: usize) {
    let src = read_wav(wav);
    let (_, cfg) = bench_cases().into_iter().find(|(n, _)| *n == case).unwrap();
    let pcm = signal(&src, cfg.sr, cfg.ch, 10.0);
    let fs = cfg.sr as usize * cfg.fms10 / 10000;
    if kind == "enc" {
        for _ in 0..iters {
            std::hint::black_box(encode_all(&cfg, &pcm).unwrap());
        }
        return;
    }
    let pk = encode_all(&cfg, &pcm).unwrap();
    let mut out = vec![0f32; fs * cfg.ch];
    for _ in 0..iters {
        let mut dec = OpusDecoder::new(cfg.sr as i32, cfg.ch).unwrap();
        for (i, p) in pk.iter().enumerate() {
            let inp: &[u8] = if kind == "dec_plc" && i % 4 == 3 { &[] } else { p };
            std::hint::black_box(dec.decode(std::hint::black_box(inp), fs, &mut out).unwrap());
        }
    }
}

fn bench_mode(wav: &str, reps: usize) {
    let src = read_wav(wav);
    for (name, cfg) in &bench_cases() {
        let pcm = signal(&src, cfg.sr, cfg.ch, 10.0);
        let pk = encode_all(cfg, &pcm).unwrap();
        let fs = cfg.sr as usize * cfg.fms10 / 10000;
        let mut enc_t = Vec::new();
        let mut dec_t = Vec::new();
        let mut plc_t = Vec::new();
        for _ in 0..reps {
            let t = Instant::now();
            std::hint::black_box(encode_all(cfg, &pcm).unwrap());
            enc_t.push(t.elapsed().as_secs_f64());
            let mut out = vec![0f32; fs * cfg.ch];
            let t = Instant::now();
            let mut dec = OpusDecoder::new(cfg.sr as i32, cfg.ch).unwrap();
            for p in &pk {
                std::hint::black_box(dec.decode(std::hint::black_box(p), fs, &mut out).unwrap());
            }
            dec_t.push(t.elapsed().as_secs_f64());
            let t = Instant::now();
            let mut dec = OpusDecoder::new(cfg.sr as i32, cfg.ch).unwrap();
            for (i, p) in pk.iter().enumerate() {
                let inp: &[u8] = if i % 4 == 3 { &[] } else { p };
                std::hint::black_box(dec.decode(inp, fs, &mut out).unwrap());
            }
            plc_t.push(t.elapsed().as_secs_f64());
        }
        for (k, v) in [("enc", &mut enc_t), ("dec", &mut dec_t), ("dec_plc", &mut plc_t)] {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            println!("{name} {k} min_ms={:.3} med_ms={:.3}", v[0] * 1e3, v[v.len() / 2] * 1e3);
        }
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    match a[1].as_str() {
        "hash" => hash_mode(&a[2]),
        "bench" => bench_mode(&a[2], a[3].parse().unwrap()),
        "run" => run_mode(&a[2], &a[3], &a[4], a[5].parse().unwrap()),
        _ => panic!("usage"),
    }
}
